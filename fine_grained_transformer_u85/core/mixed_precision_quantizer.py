"""The stock Arm EthosUQuantizer plus ordered per-module / per-op precision rules (`MixedPrecisionQuantizer`),
optional quantized embedding tables, and the memory-op pass (always on) that stops shape ops from narrowing wider
tensors back to int8. Rule syntax: (regex, QuantizationConfig | None, optional op set); first match on the deepest
nn_module_stack FQN wins; None leaves the matched nodes in fp32. The op set takes aten op names (`@sub,mul`) or
LayerNorm roles (`@square`, `@var`, ...: one tensor of the decomposed LayerNorm, see _ln_role).
"""
import re

import torch
from executorch.backends.arm.quantizer import EthosUQuantizer
from torchao.quantization.pt2e.quantizer import QuantizationAnnotation

_SHAPE_OPS = {
    torch.ops.aten.view.default, torch.ops.aten.view_copy.default, torch.ops.aten.reshape.default,
    torch.ops.aten.unsqueeze.default, torch.ops.aten.expand.default, torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int, torch.ops.aten.squeeze.dim, torch.ops.aten.contiguous.default,
    torch.ops.aten.clone.default,
}
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


def _op_name(n):
    return str(n.target).replace("aten.", "").replace(".default", "").replace(".Tensor", "")


def _ln_role(n):
    """The LayerNorm role of a node of the decomposed LayerNorm (the Arm default decomposition and GA-LN), found from
    its data dependencies, not its position, so a rule names a tensor (`@square`) rather than an op type (`@mul`):
        mean     sum / mul of the mean that feeds `center`        center  x - mean (sub)
        square   center * center                                   sumsq   sum(square)
        var      sumsq * (1 / C)                                   var_eps var + eps
        rsqrt    rsqrt(var_eps)                                    xhat    center * rsqrt
        gamma    xhat * weight                                     beta    gamma + bias
    Returns None for any other node (e.g. GA-LN's Newton step y * y: its operand is not a `center`)."""
    if n.op != "call_function":
        return None
    name = _op_name(n)
    srcs = [a for a in n.args if isinstance(a, torch.fx.Node)]
    names = [_op_name(a) if a.op == "call_function" else "" for a in srcs]
    if name == "sub":
        return "center"
    if name == "mul" and len(n.args) == 2 and n.args[0] is n.args[1] and names[:1] == ["sub"]:
        return "square"
    if name == "rsqrt":
        return "rsqrt"
    roles = [_ln_role(a) if a.op == "call_function" and _op_name(a) in ("mul", "sub", "sum.dim_IntList", "add", "rsqrt") else None
             for a in srcs] if name in ("sum.dim_IntList", "mul", "add") else []
    if name == "sum.dim_IntList" and "square" in roles:
        return "sumsq"
    if name == "mul" and "sumsq" in roles:
        return "var"
    if name == "add" and "var" in roles:
        return "var_eps"
    if name == "mul" and "rsqrt" in roles:
        return "xhat"
    if name == "mul" and "xhat" in roles:
        return "gamma"
    if name == "add" and "gamma" in roles:
        return "beta"
    if name in ("sum.dim_IntList", "mul") and any(_feeds_center(u) for u in n.users):
        return "mean"
    return None


def _feeds_center(n, _depth=0):
    """True when n reaches a `sub` (x - mean) through mean-chain ops (mul / reshape)."""
    if n.op != "call_function" or _depth > 4:
        return False
    name = _op_name(n)
    if name == "sub":
        return True
    return name in ("mul", "reshape", "view") and any(_feeds_center(u, _depth + 1) for u in n.users)


LN_ROLES = ("mean", "center", "square", "sumsq", "var", "var_eps", "rsqrt", "xhat", "gamma", "beta")


class MixedPrecisionQuantizer(EthosUQuantizer):
    """The stock Arm quantizer plus ordered per-module precision rules.

    `rules` is a list of (regex, QuantizationConfig | None). A rule matches a
    node when the regex matches the *deepest* module in its nn_module_stack,
    so a block-level rule reaches the residual adds without sweeping the
    block's attention / MLP submodules (torchao's set_module_name matches
    ancestors). None means "leave in fp32": the nodes are marked annotated so
    no later pass quantizes them. Remaining nodes get the stock annotation.
    """

    share_int8_pcs_operands = False  # see _share_int8_operands_of_pcs_adds; off by default

    def __init__(self, compile_spec, rules=(), embedding_bits=None):
        super().__init__(compile_spec)
        self.rules = list(rules)
        self.embedding_bits = embedding_bits  # 8 / 16: quantize aten.embedding tables (per-tensor symmetric); output left to the consumer

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
                # an op type ("mul", "sum.dim_IntList", ...) or a LayerNorm role ("square", "var", ..., see _ln_role)
                return _op_name(n) in ops or (bool(ops & set(LN_ROLES)) and _ln_role(n) in ops)
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
        if self.embedding_bits:
            self._annotate_embeddings(model)
        if self.share_int8_pcs_operands:
            self._share_int8_operands_of_pcs_adds(model)
        return self._widen_memory_ops(model)

    def _share_int8_operands_of_pcs_adds(self, model):
        """PCS residual adds (per-channel symmetric output): an operand that already is a per-tensor INT8 tensor
        (the attention out_proj / MLP fc2 output under the global A8W8) keeps that quantization instead of being
        re-quantized onto the add's INT16 input grid, which only re-rounds the same 256 codes. Lowered, the add then
        reads the INT8 codes through one INT8 -> INT32 RESCALE instead of INT8 -> INT16 -> INT32."""
        from torchao.quantization.pt2e.quantizer import SharedQuantizationSpec
        n = 0
        for node in model.graph.nodes:
            ann = node.meta.get("quantization_annotation")
            if node.op != "call_function" or ann is None or ann.output_qspec is None:
                continue
            if getattr(_root_qspec(ann.output_qspec), "qscheme", None) is not torch.per_channel_symmetric:
                continue
            for a in list(ann.input_qspec_map):
                pann = a.meta.get("quantization_annotation")
                spec = _root_qspec(pann.output_qspec) if pann is not None and pann.output_qspec is not None else None
                if (spec is not None and getattr(spec, "dtype", None) == torch.int8
                        and getattr(spec, "qscheme", None) in (torch.per_tensor_affine, torch.per_tensor_symmetric)):
                    ann.input_qspec_map[a] = SharedQuantizationSpec(a)
                    n += 1
        self.shared_int8_operands = n

    def _annotate_embeddings(self, model):
        """aten.embedding: the stock Arm annotator leaves the table in fp32. Quantize the table per-tensor symmetric
        (int8 / int16, MinMax) and leave the gathered output unannotated: the consumer's own input observer
        (e.g. the positional add at int16) re-quantizes it, which on the device is GATHER + RESCALE."""
        from torchao.quantization.pt2e import MinMaxObserver
        from torchao.quantization.pt2e.quantizer import QuantizationSpec
        bits = self.embedding_bits
        lo, hi = (-128, 127) if bits == 8 else (-32768, 32767)
        spec = QuantizationSpec(dtype=torch.int8 if bits == 8 else torch.int16, quant_min=lo, quant_max=hi,
                                qscheme=torch.per_tensor_symmetric, observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2 ** -16))
        n = 0
        for node in model.graph.nodes:
            if node.op == "call_function" and node.target is torch.ops.aten.embedding.default:
                weight = node.args[0]
                ann = QuantizationAnnotation(input_qspec_map={weight: spec}, output_qspec=None, _annotated=True)
                node.meta["quantization_annotation"] = ann
                custom = dict(node.meta.get("custom", {}))
                custom["_arm_annotation_info"] = {"quantized": True}
                node.meta["custom"] = custom
                n += 1
        self.annotated_embeddings = n

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
        self.widened_memory_ops = self.transparent_memory_ops = 0
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
