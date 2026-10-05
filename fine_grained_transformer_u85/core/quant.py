"""Shared PT2E quantization plumbing for the evaluation and lowering entry points: activation configs,
precision-rule parsing, CPU export + annotation, calibration + conversion, the compile spec handed to the
Arm quantizer, and moving a converted graph to CUDA.
"""
import dataclasses
import importlib.util
import time
from enum import Enum
from pathlib import Path

import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec
from executorch.backends.arm.quantizer import (
    EthosUQuantizer,
    QuantizationConfig,
    get_symmetric_quantization_config,
)
from executorch.backends.arm.quantizer.arm_quantizer import (
    get_symmetric_a16w8_quantization_config,
)
from torchao.quantization.pt2e import MinMaxObserver
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


from torchao.quantization.pt2e.quantizer import QuantizationSpec
from torchao.quantization.pt2e import PerChannelMinMaxObserver

from core.ptf_observer import PTFPerChannelObserver

class ActObserver(Enum):
    HISTOGRAM = "histogram"  # Arm default: searches a clipped range that minimizes quantization error
    MINMAX = "minmax"        # full observed range, as FQ-ViT's calibrate() folds min / max
    KMEDIAN = "kmedian"      # per-tensor grid sized by k * median |x| (k = the recipe's --ln-dual-k), saturating: the dual-range fine rsqrt table


class KeepFp32(Enum):
    """Module types left unquantized, to test whether they cause an accuracy loss."""
    LAYERNORM = "layernorm"

    def module_type(self):
        return {KeepFp32.LAYERNORM: torch.nn.LayerNorm}[self]


class QuantConfig(Enum):
    A8W8 = "a8w8"    # int8 activations + int8 weights: the aot_arm_compiler default
    A16W8 = "a16w8"  # int16 activations + int8 weights (TOSA int16 extension, Ethos-U85)
    A16W8E16 = "a16w8e16"  # a16w8 with the int16 scale floor lowered from Arm's 2^-12 to 2^-16; recipe-owned, for the
    # small-range tensors (GA-LN fine / mask tables) whose values sit below the 2^-12 grid's first INT16 TABLE sample
    A32W8 = "a32w8"  # int32 symmetric activations (for op-level rules on LayerNorm internals)
    A8IN16OUT = "a8in16out"        # int8 input, int16 output: e.g. fc1 whose output feeds GELU unquantized (FQ-ViT placement)
    A16IN8OUT = "a16in8out"        # int16 inputs, int8 per-tensor output: an int16 adder whose result is stored int8
    A16INPTF8OUT = "a16inptf8out"  # int16 inputs, int8 per-channel power-of-two (FQ-ViT PTF) output
    A16INPC8OUT = "a16inpc8out"    # int16 inputs, int8 per-channel with UNCONSTRAINED per-channel scales (arbitrary multiplier)
    A16INPC16OUT = "a16inpc16out"  # int16 inputs, int16 per-channel MinMax output (OFM per-channel scaling at 16 bit)
    A16INPC8SYMOUT = "a16inpc8symout"  # int16 inputs, int8 per-channel SYMMETRIC MinMax output (zero-point-free: lowering rewrite = per-channel MUL only)
    APTF8IN8OUT = "aptf8in8out"    # int8 PER-CHANNEL (PTF) input, int8 per-tensor output: per-channel LN output folded into the next Linear's weights

    def build(self, act_observer: ActObserver, dual_k):
        """dual_k: k of the kmedian observer (GA-LN dual-range fine table), None when no kmedian tensor exists."""
        if self is QuantConfig.A8IN16OUT:
            a16 = QuantConfig.A16W8.build(act_observer, dual_k); a8 = QuantConfig.A8W8.build(act_observer, dual_k)
            return QuantizationConfig(a8.input_activation, a16.output_activation, a8.weight, a8.bias)
        if self is QuantConfig.APTF8IN8OUT:
            a8 = QuantConfig.A8W8.build(act_observer, dual_k)
            inp = QuantizationSpec(dtype=torch.int8, quant_min=-128, quant_max=127, qscheme=torch.per_channel_affine, ch_axis=2,
                                   observer_or_fake_quant_ctr=PTFPerChannelObserver.with_args(ch_axis=2))

            class _LooseInputConfig(QuantizationConfig):
                def get_input_act_qspec(self, node=None):
                    return self.input_activation

                def get_output_act_qspec(self, node=None):
                    return self.output_activation
            return _LooseInputConfig(inp, a8.output_activation, a8.weight, a8.bias)
        if self in (QuantConfig.A16IN8OUT, QuantConfig.A16INPTF8OUT, QuantConfig.A16INPC8OUT, QuantConfig.A16INPC16OUT,
                    QuantConfig.A16INPC8SYMOUT):
            a16 = QuantConfig.A16W8.build(act_observer, dual_k)
            a8 = QuantConfig.A8W8.build(act_observer, dual_k)
            out = a8.output_activation
            if self in (QuantConfig.A16INPTF8OUT, QuantConfig.A16INPC8OUT, QuantConfig.A16INPC16OUT, QuantConfig.A16INPC8SYMOUT):
                dtype, lo, hi, eps = ((torch.int16, -32768, 32767, 2 ** -16) if self is QuantConfig.A16INPC16OUT
                                      else (torch.int8, -128, 127, 2 ** -12))
                qscheme = torch.per_channel_symmetric if self is QuantConfig.A16INPC8SYMOUT else torch.per_channel_affine
                obs = (PTFPerChannelObserver.with_args(ch_axis=2) if self is QuantConfig.A16INPTF8OUT
                       else PerChannelMinMaxObserver.with_args(ch_axis=2, dtype=dtype, qscheme=qscheme,
                                                               quant_min=lo, quant_max=hi, eps=eps))
                out = QuantizationSpec(dtype=dtype, quant_min=lo, quant_max=hi,
                                       qscheme=qscheme, ch_axis=2, observer_or_fake_quant_ctr=obs)
            if self in (QuantConfig.A16INPTF8OUT, QuantConfig.A16INPC8OUT, QuantConfig.A16INPC16OUT, QuantConfig.A16INPC8SYMOUT):
                # The Arm QuantizationConfig only admits per-tensor activation specs; PTF is per-channel
                # (fake-quant accuracy experiment, not a lowerable config), so bypass that validation.
                class _LooseQuantizationConfig(QuantizationConfig):
                    def get_input_act_qspec(self, node=None):
                        return self.input_activation

                    def get_output_act_qspec(self, node=None):
                        return self.output_activation
                return _LooseQuantizationConfig(a16.input_activation, out, a16.weight, a16.bias)
            return QuantizationConfig(a16.input_activation, out, a16.weight, a16.bias)
        config = (get_symmetric_quantization_config() if self is QuantConfig.A8W8  # Arm int8 default: eps 2^-16
                  else get_symmetric_a16w8_quantization_config(epsilon=2 ** -16) if self is QuantConfig.A16W8E16
                  else get_symmetric_a16w8_quantization_config())  # Arm int16 default: eps 2^-12 (arm_quantizer.py:297)
        if self is QuantConfig.A32W8:
            act = QuantizationSpec(dtype=torch.int32, quant_min=-(2 ** 31) + 1, quant_max=2 ** 31 - 1,
                                   qscheme=torch.per_tensor_symmetric,
                                   observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2 ** -20))
            return QuantizationConfig(act, act, config.weight, config.bias)
        eps = 2 ** -16 if self is QuantConfig.A16W8E16 else 2 ** -12  # the same floors as the Arm configs above. Under the
        # 2^-12 floor the INT16 TABLE's first sample (scale * 128) can sit above small per-token variances, which the
        # rsqrt table then cannot resolve; the e16 variant exists for the tensors that need it.
        if act_observer is ActObserver.MINMAX:
            act = dataclasses.replace(config.input_activation, observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=eps))
            config = QuantizationConfig(act, act, config.weight, config.bias)
        elif act_observer is ActObserver.KMEDIAN:
            from core.ptf_observer import KMedianObserver
            act = dataclasses.replace(config.input_activation,
                                      observer_or_fake_quant_ctr=KMedianObserver.with_args(k=dual_k, eps=eps))
            config = QuantizationConfig(act, act, config.weight, config.bias)
        return config


def load_module_from_path(name: str, path: Path):
    # Loading by path skips the FQ-ViT package __init__, which pulls in its
    # whole quantization stack.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_tensor_factory(target) -> bool:
    """True for ops like aten.full / arange: a `device` argument and no tensor input."""
    overload = getattr(target, "default", target)  # OpOverloadPacket -> its default overload
    schema = getattr(overload, "_schema", None)
    if schema is None:
        return False
    return (any(a.name == "device" for a in schema.arguments)
            and not any("Tensor" in str(a.type) for a in schema.arguments))


def move_graph_module(gm: torch.fx.GraphModule, device: str):
    """GraphModule.to() only moves parameters and buffers. Also retarget device
    kwargs baked into nodes at export time and move plain tensor attributes.
    The Arm decomposition passes insert aten.full constants with no device kwarg
    at all, which would be created on CPU, so factory ops get one added."""
    n_kwargs = 0
    for node in gm.graph.nodes:
        if "device" in node.kwargs or (
                node.op == "call_function" and is_tensor_factory(node.target)):
            node.kwargs = {**node.kwargs, "device": torch.device(device)}
            n_kwargs += 1
    gm.recompile()
    gm.to(device)
    n_attrs = 0
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        owner, _, leaf = node.target.rpartition(".")
        parent = gm.get_submodule(owner) if owner else gm
        value = getattr(parent, leaf)
        if isinstance(value, torch.Tensor) and value.device.type != torch.device(device).type:
            setattr(parent, leaf, value.to(device))
            n_attrs += 1
    return n_kwargs, n_attrs


def prepare_on_cpu(module, example_inputs, compile_spec, quant_config: QuantConfig,
                   act_observer: ActObserver, keep_fp32, quantizer_cls=EthosUQuantizer):
    """Export, Arm annotation passes and observer insertion: the recipe of
    aot_arm_compiler.quantize(). CPU only, because the Arm passes create CPU
    constants and their shape propagation rejects a CUDA graph."""
    exported = torch.export.export(module, example_inputs, strict=True)
    quantizer = quantizer_cls(compile_spec)
    assert act_observer is not ActObserver.KMEDIAN, "kmedian is a per-rule observer (GA-LN dual), not a global one"
    quantizer.set_global(quant_config.build(act_observer, None))
    for kept in keep_fp32:
        quantizer.set_module_type(kept.module_type(), None)
    # The default guard node is a call_module, which the Arm annotation passes reject.
    prepared = prepare_pt2e(exported.module(check_guards=False), quantizer)
    if hasattr(quantizer, "transparent_memory_ops"):
        print(f"memory-op pass: transparent={quantizer.transparent_memory_ops} int16-carrier={quantizer.widened_memory_ops}", flush=True)
    n_observers = sum(1 for n in prepared.graph.nodes
                      if n.op == "call_module" and str(n.target).startswith("activation_post_process"))
    print(f"  prepared graph: {n_observers} observer call sites "
          f"(keep_fp32={[k.value for k in keep_fp32]})", flush=True)
    return prepared


def parse_rules(rules_str, default_observer: ActObserver, dual_k):
    """'regex=config[:observer][@op,op];...' -> [(regex, QuantizationConfig | None, {ops} | None)], in order
    (MixedPrecisionQuantizer applies the first match). 'fp32' leaves the match unquantized.
    dual_k: k of the kmedian observer, None when the recipe has no dual-range LayerNorm."""
    rules = []
    for item in [r for r in (rules_str or "").split(";") if r]:
        regex, cfg = item.rsplit("=", 1)
        cfg, _, ops = cfg.partition("@")
        cfg, _, obs = cfg.partition(":")
        rules.append((regex, None if cfg == "fp32" else QuantConfig(cfg).build(ActObserver(obs) if obs else default_observer, dual_k),
                      set(ops.split(",")) if ops else None))
    return rules


def quantizer_compile_spec() -> EthosUCompileSpec:
    """The compile spec the accuracy runners hand to the Arm quantizer. They never lower to Ethos-U, so the spec
    is fixed; lower_probe.py builds its own for the Vela run."""
    kit = Path(__file__).resolve().parents[2]
    return EthosUCompileSpec("ethos-u85-256", system_config="Ethos_U85_SYS_DRAM_Low", memory_mode="Dedicated_Sram",
                             extra_flags=[], config_ini=str(kit / "scripts/vela/default_vela.ini"))


def calibrate_and_convert(name: str, prepared, calib_batches, device: str):
    """Runs the calibration loop and the conversion; `prepared` is already on `device`."""
    t0 = time.time()
    with torch.no_grad():
        for batch in calib_batches:
            prepared(*batch)
    t1 = time.time()
    converted = convert_pt2e(prepared)
    print(f"  quantize[{name}] on {device}: calibrate={t1 - t0:.1f}s over {len(calib_batches)} "
          f"batches ({(t1 - t0) / len(calib_batches):.2f}s/batch) convert={time.time() - t1:.1f}s",
          flush=True)
    return converted
