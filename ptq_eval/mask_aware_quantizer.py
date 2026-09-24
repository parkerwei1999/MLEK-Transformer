"""EthosUQuantizer variant that keeps additive attention masks out of the
softmax-input grid (option 2 of the Swin mask discussion).

Stock Arm annotation gives `add(scores, mask)` its own observer, so the -100
(or -255 after ReplaceInfValuesPass) mask values stretch the int8 grid of the
scores by 4-8x. Here the add output, the following amax and the x-max
subtraction get DerivedQuantizationSpecs computed from the scores' observer:
no observer ever sees the mask, and masked entries saturate at the grid
minimum at convert time, which the exp table maps to ~0. Model code and the
lowering are untouched; on device this is an ordinary int8 ADD.
"""
import functools

import torch
from executorch.backends.arm.quantizer import EthosUQuantizer
from torchao.quantization.pt2e.quantizer import DerivedQuantizationSpec

_SHAPE_OPS = {
    torch.ops.aten.view.default, torch.ops.aten.view_copy.default, torch.ops.aten.reshape.default,
    torch.ops.aten.unsqueeze.default, torch.ops.aten.expand.default, torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int, torch.ops.aten.squeeze.dim, torch.ops.aten.contiguous.default,
    torch.ops.aten.clone.default,
}
MASK_THRESHOLD = -50.0  # -100 (Swin) and -255 (ReplaceInfValuesPass output) qualify


def _constant_source(node, model):
    """Follow shape-only ops back to a constant; return its tensor or None."""
    while isinstance(node, torch.fx.Node) and node.op == "call_function" and node.target in _SHAPE_OPS:
        node = node.args[0]
    if not isinstance(node, torch.fx.Node) or node.op != "get_attr":
        return None
    try:
        return functools.reduce(getattr, node.target.split("."), model)
    except AttributeError:
        return None


def _root_qspec(spec):
    """Follow SharedQuantizationSpec links (shape ops share their input's
    qspec) to the QuantizationSpec that actually carries dtype and range."""
    from torchao.quantization.pt2e.quantizer import SharedQuantizationSpec
    seen = 0
    while isinstance(spec, SharedQuantizationSpec) and seen < 64:
        target = spec.edge_or_node
        if isinstance(target, tuple):  # (producer, consumer) edge -> consumer's input spec
            spec = target[1].meta["quantization_annotation"].input_qspec_map[target[0]]
        else:  # node -> its output spec
            spec = target.meta["quantization_annotation"].output_qspec
        seen += 1
    return spec


class MaskAwareQuantizer(EthosUQuantizer):
    def __init__(self, compile_spec, mask_threshold=MASK_THRESHOLD):
        super().__init__(compile_spec)
        self.mask_threshold = mask_threshold
        self.rewritten = 0

    def annotate(self, model):
        model = super().annotate(model)
        if self.mask_threshold is None:  # rules only, no mask rewrite
            return model
        for add in list(model.graph.nodes):
            if add.op != "call_function" or add.target != torch.ops.aten.add.Tensor:
                continue
            for scores_idx in (0, 1):
                mask_arg = add.args[1 - scores_idx]
                const = _constant_source(mask_arg, model)
                if const is None or not torch.is_floating_point(const) or const.min() > self.mask_threshold:
                    continue
                scores = add.args[scores_idx]
                self._rewrite(add, scores)
                self.rewritten += 1
                break
        return model

    def _rewrite(self, add, scores):
        grid = self._derived(scores, zp=None)
        self._set(add, {}, grid)
        self._propagate(add, scores, grid)

    def _propagate(self, node, scores, grid):
        """Push the derived grid through shape ops into the softmax decomposition."""
        for user in list(node.users):
            if user.target in _SHAPE_OPS:
                self._set(user, {node: grid}, grid)
                self._propagate(user, scores, grid)
            elif user.target == torch.ops.aten.amax.default:
                self._set(user, {node: grid}, grid)
            elif user.target == torch.ops.aten.sub.Tensor:
                shifted = self._derived(scores, zp=127)  # x - max lives in [-255*s, 0]
                self._set(user, {a: grid for a in user.args if isinstance(a, torch.fx.Node)}, shifted)
                for u2 in user.users:
                    if u2.target == torch.ops.aten.exp.default:
                        self._set(u2, {user: shifted}, None)

    @staticmethod
    def _derived(scores, zp):
        # Inherit dtype / range / scheme from the scores' own qspec so the rule
        # is correct whether the attention block is a8 or a16.
        base = _root_qspec(scores.meta["quantization_annotation"].output_qspec)
        dtype, qmin, qmax = base.dtype, base.quant_min, base.quant_max
        qscheme = base.qscheme

        def derive(obs_or_fqs):
            scale, zero_point = obs_or_fqs[0].calculate_qparams()
            if zp is not None:
                zero_point = torch.full_like(zero_point, qmax if zp == 127 else zp)
            return scale, zero_point
        return DerivedQuantizationSpec(derived_from=[scores], derive_qparams_fn=derive,
                                       dtype=dtype, quant_min=qmin, quant_max=qmax, qscheme=qscheme)

    @staticmethod
    def _set(node, inputs, output):
        ann = node.meta.get("quantization_annotation")
        if ann is None:
            return
        for arg, spec in inputs.items():
            ann.input_qspec_map[arg] = spec
        if output is not None:
            ann.output_qspec = output


import re  # noqa: E402
from torchao.quantization.pt2e.quantizer import QuantizationAnnotation  # noqa: E402


def _deepest_fqn(node, _depth=0):
    """Deepest module FQN of a node.

    Nodes created by the Arm decomposition passes (e.g. the mean chain of a
    decomposed LayerNorm: reshape/sum/mul/reshape/sub) carry no nn_module_stack,
    so they would escape every module rule and stay at the global int8 config,
    i.e. a hidden int8 chain inside a LayerNorm meant to be fp32/int16. Such a
    node inherits the FQN of the first user that has one (the decomposed ops
    feed the remaining ops of the same module).
    """
    stack = node.meta.get("nn_module_stack", {})
    if not stack:
        if _depth >= 8:
            return ""
        for user in node.users:
            fqn = _deepest_fqn(user, _depth + 1)
            if fqn:
                return fqn
        return ""
    name = list(stack.values())[-1][0]
    return name[len("L['self']."):] if name.startswith("L['self'].") else name


class MixedPrecisionQuantizer(MaskAwareQuantizer):
    """MaskAwareQuantizer plus ordered per-module precision rules.

    `rules` is a list of (regex, QuantizationConfig | None). A rule matches a
    node when the regex matches the *deepest* module in its nn_module_stack,
    so a block-level rule reaches the residual adds without sweeping the
    block's attention / MLP submodules (torchao's set_module_name matches
    ancestors). None means "leave in fp32": the nodes are marked annotated so
    no later pass quantizes them. Remaining nodes get the stock annotation.
    """

    def __init__(self, compile_spec, rules=(), mask_threshold=MASK_THRESHOLD):
        super().__init__(compile_spec, mask_threshold)
        self.rules = list(rules)

    def annotate(self, model):
        for rule in self.rules:
            regex, config = rule[0], rule[1]
            ops = rule[2] if len(rule) > 2 else None  # optional op-type filter, e.g. {"sum.dim_IntList", "mul"}
            pattern = re.compile(regex)

            def matches(n, p=pattern, ops=ops):
                if not p.search(_deepest_fqn(n)):
                    return False
                if ops is None:
                    return True
                name = str(n.target).replace("aten.", "").replace(".default", "").replace(".Tensor", "")
                return name in ops
            if config is None:
                for node in model.graph.nodes:
                    if node.op == "call_function" and matches(node) and "quantization_annotation" not in node.meta:
                        node.meta["quantization_annotation"] = QuantizationAnnotation(_annotated=True)
            else:
                # ExecuTorch >= 1.3 wraps the TOSA quantizer in a composable
                # quantizer; the static-pattern annotator lives on the inner one.
                inner = self if hasattr(self, "_annotate_all_static_patterns") else self.quantizer
                inner._annotate_all_static_patterns(model, config, matches)
        model = super().annotate(model)
        return self._widen_memory_ops(model)

    # Memory / shape ops (view, slice, cat, permute, ...) are annotated by the Arm annotator as
    # quantization boundaries. When their producer is per-channel (PTF) or int16/int32, the global
    # int8 per-tensor spec they receive becomes a real re-quantization of the tensor ("requant
    # leakage", e.g. Swin's PatchMerging slices/cat re-crushing the PTF residual). Deployment-wise a
    # memory op moves codes untouched and any expansion happens once at the consumer, so:
    #   * per-channel producer  -> the op's annotation is removed: the dequantized per-channel
    #     values pass through exactly and the consumer's own input spec (e.g. the int32 LayerNorm
    #     input) is the only re-quantization ("transparent");
    #   * int16 / int32 per-tensor producer -> int16 per-tensor MinMax carriers (near-lossless and
    #     still lowerable, since int16 shape ops are plain TOSA).
    # Ops already given a wider spec by an explicit rule are left alone. Applied globally, so no
    # per-model merge rule is needed.
    _MEMORY_OPS = _SHAPE_OPS | {
        torch.ops.aten.slice.Tensor, torch.ops.aten.slice_copy.Tensor, torch.ops.aten.cat.default,
        torch.ops.aten.select.int, torch.ops.aten.split.Tensor, torch.ops.aten.flatten.using_ints,
    }

    def _widen_memory_ops(self, model):
        import os
        self.widened_memory_ops = self.transparent_memory_ops = 0
        if os.environ.get("PTQ_MEMORY_OP_PASS", "on") == "off":  # A/B control: keep the Arm annotator's int8 boundaries
            return model
        from executorch.backends.arm.quantizer.arm_quantizer import get_symmetric_a16w8_quantization_config
        from torchao.quantization.pt2e import MinMaxObserver
        import dataclasses
        a16 = get_symmetric_a16w8_quantization_config()
        carrier = dataclasses.replace(a16.input_activation,
                                      observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2 ** -12))

        def out_spec(node):
            ann = node.meta.get("quantization_annotation")
            return _root_qspec(ann.output_qspec) if ann is not None and ann.output_qspec is not None else None

        def per_channel(spec):
            return spec is not None and "per_channel" in str(getattr(spec, "qscheme", ""))

        def wide_per_tensor(spec):
            return spec is not None and getattr(spec, "dtype", None) in (torch.int16, torch.int32)

        transparent = set()  # memory ops whose annotation was removed (per-channel pass-through)
        widened = 0
        for node in model.graph.nodes:
            if node.op != "call_function" or node.target not in self._MEMORY_OPS:
                continue
            ann = node.meta.get("quantization_annotation")
            if ann is None or not ann.input_qspec_map:
                continue
            inputs = [a for a in node.all_input_nodes if a in ann.input_qspec_map]
            specs = [out_spec(a) for a in inputs]
            has_pc = any(a in transparent or per_channel(s) for a, s in zip(inputs, specs))
            has_wide = any(wide_per_tensor(s) for s in specs)
            if not (has_pc or has_wide):
                continue
            own = _root_qspec(ann.output_qspec) if ann.output_qspec is not None else None
            if own is not None and getattr(own, "dtype", None) != torch.int8:
                continue  # already wide by an explicit rule
            if has_pc:
                del node.meta["quantization_annotation"]
                transparent.add(node)
            else:
                for a in inputs:
                    ann.input_qspec_map[a] = carrier
                ann.output_qspec = carrier
                widened += 1
        self.widened_memory_ops = widened
        self.transparent_memory_ops = len(transparent)
        return model
