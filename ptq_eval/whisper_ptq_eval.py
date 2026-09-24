"""Post-training quantization (PTQ) with real-speech calibration, plus word /
character error rate (WER / CER) evaluation, for the Whisper Ethos-U wrappers
defined in ../whisper_executorch_wrapper.py.

`examples.arm.aot_arm_compiler` can only calibrate on the single example input,
so this script repeats its quantization recipe (EthosUQuantizer + the default
symmetric config, strict export) with a calibration loop over LibriSpeech.

Measurement protocol follows FQ-ViT (apo_inject/whisper_utils.py): calibrate on
the first N utterances of dev-clean, evaluate on the first N of test-clean,
normalize with Whisper's EnglishTextNormalizer, score with jiwer.

The decoder wrapper has a static length and no KV cache, so greedy decoding
re-runs the whole decoder per generated token and reads the logits at the last
filled position; the causal mask keeps the trailing padding from leaking back.
"""
import argparse
import dataclasses
import importlib.util
import json
import random
import time
from enum import Enum
from pathlib import Path

import jiwer
import torch
import whisper
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
PC_REWRITE = False  # set from --pc-rewrite: lower per-channel activation Q/DQ before evaluation
from transformers import GenerationConfig
from whisper.normalizers import EnglishTextNormalizer
from whisper.tokenizer import get_tokenizer

HERE = Path(__file__).resolve().parent


class CalibMode(Enum):
    REAL = "real"      # LibriSpeech dev-clean utterances
    RANDOM = "random"  # the wrapper's single random example input (aot_arm_compiler behaviour)


class CalibFeed(Enum):
    PARALLEL = "parallel"              # one teacher-forced decoder pass per utterance, plus one random prefix
    AUTOREGRESSIVE = "autoregressive"  # FQ-ViT: observers run inside the real greedy decode loop


class ActObserver(Enum):
    HISTOGRAM = "histogram"  # Arm default: searches a clipped range that minimizes quantization error
    MINMAX = "minmax"        # full observed range, as FQ-ViT's calibrate() folds min / max
    KMEDIAN = "kmedian"      # per-tensor grid sized by k * median |x| (env LN_DUAL_K, default 4), saturating: the dual-range fine rsqrt table


class KeepFp32(Enum):
    """Module types left unquantized, to test whether they cause an accuracy loss."""
    LAYERNORM = "layernorm"

    def module_type(self):
        return {KeepFp32.LAYERNORM: torch.nn.LayerNorm}[self]


class QuantConfig(Enum):
    A8W8 = "a8w8"    # int8 activations + int8 weights: the aot_arm_compiler default
    A16W8 = "a16w8"  # int16 activations + int8 weights (TOSA int16 extension, Ethos-U85)
    A32W8 = "a32w8"  # int32 symmetric activations (for op-level rules on LayerNorm internals)
    A16W8E16 = "a16w8e16"          # a16w8 with the int16 scale floor lowered from 2^-12 to 2^-16 (small-range tensors)
    A8IN16OUT = "a8in16out"        # int8 input, int16 output: e.g. fc1 whose output feeds GELU unquantized (FQ-ViT placement)
    A16IN8OUT = "a16in8out"        # int16 inputs, int8 per-tensor output: an int16 adder whose result is stored int8
    A16INPTF8OUT = "a16inptf8out"  # int16 inputs, int8 per-channel power-of-two (FQ-ViT PTF) output
    A16INPC8OUT = "a16inpc8out"    # int16 inputs, int8 per-channel with UNCONSTRAINED per-channel scales (arbitrary multiplier)
    A16INPC16OUT = "a16inpc16out"  # int16 inputs, int16 per-channel MinMax output (OFM per-channel scaling at 16 bit)
    A16INPC8SYMOUT = "a16inpc8symout"  # int16 inputs, int8 per-channel SYMMETRIC MinMax output (zero-point-free: lowering rewrite = per-channel MUL only)
    APTF8IN8OUT = "aptf8in8out"    # int8 PER-CHANNEL (PTF) input, int8 per-tensor output: per-channel LN output folded into the next Linear's weights

    def build(self, act_observer: ActObserver):
        if self is QuantConfig.A8IN16OUT:
            a16 = QuantConfig.A16W8.build(act_observer); a8 = QuantConfig.A8W8.build(act_observer)
            return QuantizationConfig(a8.input_activation, a16.output_activation, a8.weight, a8.bias)
        if self is QuantConfig.APTF8IN8OUT:
            from torchao.quantization.pt2e.quantizer import QuantizationSpec
            from ptf_observer import PTFPerChannelObserver
            a8 = QuantConfig.A8W8.build(act_observer)
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
            a16 = QuantConfig.A16W8.build(act_observer)
            a8 = QuantConfig.A8W8.build(act_observer)
            out = a8.output_activation
            if self in (QuantConfig.A16INPTF8OUT, QuantConfig.A16INPC8OUT, QuantConfig.A16INPC16OUT, QuantConfig.A16INPC8SYMOUT):
                from torchao.quantization.pt2e.quantizer import QuantizationSpec
                from torchao.quantization.pt2e import PerChannelMinMaxObserver
                from ptf_observer import PTFPerChannelObserver
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
        config = (get_symmetric_quantization_config() if self is QuantConfig.A8W8
                  else get_symmetric_a16w8_quantization_config(epsilon=2 ** -16) if self is QuantConfig.A16W8E16
                  else get_symmetric_a16w8_quantization_config())
        if self is QuantConfig.A32W8:
            from torchao.quantization.pt2e.quantizer import QuantizationSpec
            act = QuantizationSpec(dtype=torch.int32, quant_min=-(2 ** 31) + 1, quant_max=2 ** 31 - 1,
                                   qscheme=torch.per_tensor_symmetric,
                                   observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=2 ** -20))
            return QuantizationConfig(act, act, config.weight, config.bias)
        eps = 2 ** -16 if self is QuantConfig.A16W8E16 else 2 ** -12
        if act_observer is ActObserver.MINMAX:
            act = dataclasses.replace(config.input_activation, observer_or_fake_quant_ctr=MinMaxObserver.with_args(eps=eps))
            config = QuantizationConfig(act, act, config.weight, config.bias)
        elif act_observer is ActObserver.KMEDIAN:
            import os
            from ptf_observer import KMedianObserver
            act = dataclasses.replace(config.input_activation,
                                      observer_or_fake_quant_ctr=KMedianObserver.with_args(k=float(os.environ.get("LN_DUAL_K", 4)), eps=eps))
            config = QuantizationConfig(act, act, config.weight, config.bias)
        return config


class QuantPart(Enum):
    ENCODER = "encoder"
    DECODER = "decoder"


class Stage(Enum):
    FP32_OPENAI = "fp32-openai"  # openai-whisper transcribe(), FQ-ViT baseline protocol
    FP32_STATIC = "fp32-static"  # FP32 wrappers through this script's static-length decode
    INT8 = "int8"                # quantized wrappers through the same decode


def load_module_from_path(name: str, path: Path):
    # Loading by path skips the FQ-ViT package __init__, which pulls in its
    # whole quantization stack.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def log_mel(audio, device: str) -> torch.Tensor:
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=80)
    return mel.unsqueeze(0).to(device)


class GreedyDecoder:
    """Static-length greedy decoding shared by the FP32 and int8 decoders."""

    def __init__(self, model_id: str, dec_len: int, device: str):
        self.tokenizer = get_tokenizer(multilingual=True, language="en", task="transcribe")
        self.prompt = list(self.tokenizer.sot_sequence_including_notimestamps)
        self.eot = self.tokenizer.eot
        self.dec_len = dec_len
        self.device = device
        gen_cfg = GenerationConfig.from_pretrained(model_id)
        self.suppress = torch.tensor(gen_cfg.suppress_tokens, device=device)
        self.begin_suppress = torch.tensor(gen_cfg.begin_suppress_tokens, device=device)

    def padded_ids(self, tokens) -> torch.Tensor:
        ids = torch.full((1, self.dec_len), self.eot, dtype=torch.long, device=self.device)
        ids[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=self.device)
        return ids

    @torch.no_grad()
    def decode(self, decoder, enc_states: torch.Tensor):
        """Returns (generated token ids, whether the static length was hit)."""
        tokens = list(self.prompt)
        while len(tokens) < self.dec_len:
            logits = decoder(self.padded_ids(tokens), enc_states)[0, len(tokens) - 1].float()
            logits[self.suppress] = float("-inf")
            if len(tokens) == len(self.prompt):
                logits[self.begin_suppress] = float("-inf")
            token = int(logits.argmax())
            if token == self.eot:
                return tokens[len(self.prompt):], False
            tokens.append(token)
        return tokens[len(self.prompt):], True

    def text(self, tokens) -> str:
        return self.tokenizer.decode(tokens)


def score(refs, hyps, normalizer):
    refs_n = [normalizer(r.upper()) for r in refs]
    hyps_n = [normalizer(h.strip().upper()) for h in hyps]
    return jiwer.wer(refs_n, hyps_n) * 100, jiwer.cer(refs_n, hyps_n) * 100


def evaluate(label: str, transcribe_fn, pairs, load_audio, normalizer, out_dir: Path):
    """transcribe_fn(audio) -> (text, hit_limit, n_steps)."""
    refs, hyps, n_limit, n_steps = [], [], 0, 0
    t0 = time.time()
    # Long rule strings overflow the 255-byte filename limit: keep the full label in the summary, shorten the file name.
    import hashlib
    fname = label if len(label) <= 120 else label[:80] + "-" + hashlib.md5(label.encode()).hexdigest()[:8]
    with open(out_dir / f"hyps_{fname}.jsonl", "w") as hyp_file:
        for i, (audio_path, ref) in enumerate(pairs):
            text, hit_limit, steps = transcribe_fn(load_audio(audio_path))
            refs.append(ref)
            hyps.append(text)
            n_limit += int(hit_limit)
            n_steps += steps
            hyp_file.write(json.dumps({"audio": audio_path, "ref": ref, "hyp": text,
                                       "hit_limit": hit_limit}) + "\n")
            if (i + 1) % 20 == 0 or (i + 1) == len(pairs):
                wer, cer = score(refs, hyps, normalizer)
                print(f"  [{label}] {i + 1}/{len(pairs)} WER={wer:.2f}% CER={cer:.2f}% "
                      f"hit_limit={n_limit} ({time.time() - t0:.0f}s)", flush=True)
    wer, cer = score(refs, hyps, normalizer)
    return {"label": label, "n_eval": len(pairs), "wer_pct": wer, "cer_pct": cer,
            "n_hit_static_length": n_limit, "decode_steps": n_steps,
            "eval_time_sec": time.time() - t0}


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
                   act_observer: ActObserver, keep_fp32, quantizer_cls=EthosUQuantizer,
                   a16_module_regex=None):
    """Export, Arm annotation passes and observer insertion: the recipe of
    aot_arm_compiler.quantize(). CPU only, because the Arm passes create CPU
    constants and their shape propagation rejects a CUDA graph."""
    exported = torch.export.export(module, example_inputs, strict=True)
    quantizer = quantizer_cls(compile_spec)
    quantizer.set_global(quant_config.build(act_observer))
    for kept in keep_fp32:
        quantizer.set_module_type(kept.module_type(), None)
    if a16_module_regex:
        # Mixed precision: named modules matching the regex get int16 activations.
        import re
        a16 = QuantConfig.A16W8.build(act_observer)
        matched = [n for n, _ in module.named_modules() if n and re.search(a16_module_regex, n)]
        for n in matched:
            quantizer.set_module_name(n, a16)
        print(f"  a16w8 modules ({len(matched)}): {matched[:4]}{' ...' if len(matched) > 4 else ''}", flush=True)
    # The default guard node is a call_module, which the Arm annotation passes reject.
    prepared = prepare_pt2e(exported.module(check_guards=False), quantizer)
    if hasattr(quantizer, "transparent_memory_ops"):
        print(f"memory-op pass: transparent={quantizer.transparent_memory_ops} int16-carrier={quantizer.widened_memory_ops}", flush=True)
    n_observers = sum(1 for n in prepared.graph.nodes
                      if n.op == "call_module" and str(n.target).startswith("activation_post_process"))
    print(f"  prepared graph: {n_observers} observer call sites "
          f"(keep_fp32={[k.value for k in keep_fp32]})", flush=True)
    return prepared


def calibrate_and_convert(name: str, prepared, calib_batches, device: str):
    """Runs the calibration loop and the conversion; `prepared` is already on `device`."""
    t0 = time.time()
    with torch.no_grad():
        for batch in calib_batches:
            prepared(*batch)
    t1 = time.time()
    converted = convert_pt2e(prepared)
    if PC_REWRITE:
        from pc_rewrite import rewrite_per_channel_activations
        print(f"  pc_rewrite[{name}]: {rewrite_per_channel_activations(converted)} per-channel activation sites", flush=True)
    print(f"  quantize[{name}] on {device}: calibrate={t1 - t0:.1f}s over {len(calib_batches)} "
          f"batches ({(t1 - t0) / len(calib_batches):.2f}s/batch) convert={time.time() - t1:.1f}s",
          flush=True)
    return converted


def dump_ln_scales(path: Path, encoder_q, decoder_q, encoder, decoder, mels, greedy, device):
    """Per-LayerNorm grids of the converted graphs next to the fp32 per-token variance.

    For every LN module: the Q scale on the two `sum` outputs (sum(x) and
    sum((x-mean)^2)), on the rsqrt input and on the LN output, plus the fp32
    median / 5th-percentile of sum((x-mean)^2) over tokens, so a var grid whose
    step exceeds a typical token's variance sum is visible at a glance.
    """
    def out_scale(node):
        users = [u for u in node.users if u.op == "call_function" and "quantize_per" in str(u.target)
                 and "dequantize" not in str(u.target)]
        if not users or "per_tensor" not in str(users[0].target):
            return None
        return f"{str(users[0].args[5]).replace('torch.', '')} s={users[0].args[1]:.3g}"

    grids, census = {}, {}
    for part, gm in (("enc", encoder_q), ("dec", decoder_q)):
        if gm is None:
            continue
        for node in gm.graph.nodes:
            stack = node.meta.get("nn_module_stack")
            if node.op != "call_function" or not stack:
                continue
            fqn = list(stack.values())[-1][0]
            if not fqn.endswith("layer_norm"):
                continue
            op = str(node.target).replace("aten.", "").replace(".default", "").replace(".Tensor", "")
            if "quantize_per" not in op:  # dtype census of every LN-internal op output
                dt = (out_scale(node) or "fp32").split(" ")[0]
                census.setdefault((part, fqn), {}).setdefault(f"{op}:{dt}", 0)
                census[(part, fqn)][f"{op}:{dt}"] += 1
            if op in ("sum.dim_IntList", "rsqrt"):
                grids.setdefault((part, fqn), []).append((op, out_scale(node)))
                if op == "rsqrt":  # the grid the variance is read through
                    arg = node.args[0]
                    if isinstance(arg, torch.fx.Node) and "dequantize_per_tensor" in str(arg.target):
                        grids[(part, fqn)].append(("rsqrt_in", f"{str(arg.args[5]).replace('torch.', '')} s={arg.args[1]:.3g}"))
            elif op == "mul" and len(node.args) == 2 and node.args[0] is node.args[1]:
                grids.setdefault((part, fqn), []).append(("square", out_scale(node)))  # (x-mean)^2
    stats, sqs, vars_ = {}, {}, {}
    hooks = []
    for part, wrapper in (("enc", encoder), ("dec", decoder)):
        for name, mod in wrapper.named_modules():
            if isinstance(mod, torch.nn.LayerNorm):
                def hook(m, inp, out, key=(part, name)):
                    x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                    sq = (x - x.mean(-1, keepdim=True)) ** 2
                    stats.setdefault(key, []).append(sq.sum(-1).cpu())
                    vars_.setdefault(key, []).append(sq.mean(-1).cpu())
                    sqs.setdefault(key, []).append(sq.flatten()[:: max(1, sq.numel() // 20000)].cpu())
                hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for mel in mels:
            states = encoder(mel)
            greedy.decode(decoder, states)
    for h in hooks:
        h.remove()
    lines = []
    for key in sorted(grids):
        ss = torch.cat(stats.get(key, [torch.zeros(1)]))
        g = "  ".join(f"{op}:{sc}" for op, sc in grids[key])
        sq = torch.cat(sqs.get(key, [torch.zeros(1)]))
        va = torch.cat(vars_.get(key, [torch.zeros(1)]))
        lines.append(f"{key[0]} {key[1]:45s} {g}  | fp32 sum((x-mean)^2) median={ss.median():.3g} "
                     f"p5={ss.quantile(0.05):.3g} max={ss.max():.3g} | (x-mean)^2 elem median={sq.median():.3g} "
                     f"p90={sq.quantile(0.9):.3g} max={sq.max():.3g} | var median={va.median():.3g} p5={va.quantile(0.05):.3g}")
    lines.append("")
    for key in sorted(census):
        lines.append(f"{key[0]} {key[1]:45s} ops: " + ", ".join(f"{k}x{v}" for k, v in sorted(census[key].items())))
    path.write_text("\n".join(lines) + "\n")
    print(f"  LN scales -> {path}", flush=True)


def main(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    stages = {Stage(s) for s in args.stages}
    calib_mode = CalibMode(args.calib_mode)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    data = load_module_from_path("librispeech_data", HERE / "data/librispeech_data.py")
    protocol = load_module_from_path("whisper_protocol", HERE / "data/whisper_protocol.py")
    wrapper = load_module_from_path("whisper_executorch_wrapper", HERE / "whisper_executorch_wrapper.py")

    eval_pairs = data.gather_librispeech_files(args.librispeech_dir, "test-clean", args.n_eval)
    cal_pairs = data.gather_librispeech_files(args.librispeech_dir, "dev-clean", args.n_cal)
    normalizer = EnglishTextNormalizer()
    results = {"args": vars(args), "stages": []}

    def finish(stage_result):
        results["stages"].append(stage_result)
        with open(out_dir / "summary.json", "w") as f:
            json.dump(results, f, indent=2)

    if Stage.FP32_OPENAI in stages:
        model = whisper.load_model("tiny", device=device)

        def transcribe_openai(audio):
            return protocol.transcribe_whisper_deterministic(model, audio)["text"], False, 0

        finish(evaluate(Stage.FP32_OPENAI.value, transcribe_openai, eval_pairs,
                        data.load_audio_torchaudio, normalizer, out_dir))
        del model

    if not stages & {Stage.FP32_STATIC, Stage.INT8}:
        return

    encoder, enc_example = wrapper._build(args.model_id, wrapper.WhisperPart.ENCODER, args.dec_len)
    decoder, dec_example = wrapper._build(args.model_id, wrapper.WhisperPart.DECODER, args.dec_len)
    global PC_REWRITE
    PC_REWRITE = args.pc_rewrite
    if args.ln_newton_steps is not None or args.ln_dual_q is not None or args.ln_dual_k is not None:
        from newton_layernorm import swap_layernorms, collect_var_quantiles
        steps = args.ln_newton_steps or 0
        cmaps = {}
        if args.ln_dual_q is not None or args.ln_dual_k is not None:  # fp32 pass over the calibration utterances (real greedy decode) -> per-LN variance quantile
            encoder.to(device); decoder.to(device)
            greedy_c = GreedyDecoder(args.model_id, args.dec_len, device)
            def run_c():
                for audio_path, _ in cal_pairs:
                    states = encoder(log_mel(data.load_audio_torchaudio(audio_path), device))
                    greedy_c.decode(decoder, states)
            combined = torch.nn.ModuleDict({"enc": encoder, "dec": decoder})  # one pass collects both
            cm = collect_var_quantiles(combined, run_c, args.ln_dual_q or 0.9, k=args.ln_dual_k)
            cmaps = {"encoder": {k[4:]: v for k, v in cm.items() if k.startswith("enc.")},
                     "decoder": {k[4:]: v for k, v in cm.items() if k.startswith("dec.")}}
            encoder.cpu(); decoder.cpu()
            print(f"  dual-range rsqrt: q={args.ln_dual_q} k={args.ln_dual_k}, {len(cm)} LayerNorms, c range {min(cm.values()):.3g}..{max(cm.values()):.3g}", flush=True)
        n_swapped = swap_layernorms(encoder, steps, cmaps.get("encoder")) + swap_layernorms(decoder, steps, cmaps.get("decoder"))
        print(f"  NewtonLayerNorm: {n_swapped} LayerNorms swapped, {steps} step(s), dual q={args.ln_dual_q} k={args.ln_dual_k}", flush=True)
    compile_spec = EthosUCompileSpec(
        args.target, system_config=args.system_config, memory_mode=args.memory_mode,
        extra_flags=["--verbose-operators", "--verbose-cycle-estimate"], config_ini=args.vela_config)
    # Parts left out of --quant-parts stay FP32, to localize an accuracy loss.
    quant_parts = {QuantPart(p) for p in args.quant_parts}
    quant_config = QuantConfig(args.quant_config)
    act_observer = ActObserver(args.act_observer)
    feed = CalibFeed(args.calib_feed)
    keep_fp32 = [KeepFp32(k) for k in args.keep_fp32]
    quantizer_cls = EthosUQuantizer
    if args.mask_aware or args.prec_rules:
        maq = load_module_from_path("mask_aware_quantizer", HERE / "mask_aware_quantizer.py")
        quantizer_cls = maq.MaskAwareQuantizer
    if args.prec_rules:
        import functools
        rules = []
        for item in args.prec_rules.split(";"):
            regex, cfg = item.rsplit("=", 1)
            cfg, _, ops = cfg.partition("@")
            cfg, _, obs = cfg.partition(":")
            rules.append((regex, None if cfg == "fp32" else QuantConfig(cfg).build(
                ActObserver(obs or args.act_observer)), set(ops.split(",")) if ops else None))
        quantizer_cls = functools.partial(maq.MixedPrecisionQuantizer, rules=rules,
                                          mask_threshold=maq.MASK_THRESHOLD if args.mask_aware else None,
                                          embedding_bits=args.embedding_bits)
    prepared = {}
    if Stage.INT8 in stages:
        t0 = time.time()
        if QuantPart.ENCODER in quant_parts:
            prepared[QuantPart.ENCODER] = prepare_on_cpu(
                encoder, enc_example, compile_spec, quant_config, act_observer, keep_fp32,
                quantizer_cls=quantizer_cls)
        if QuantPart.DECODER in quant_parts:
            prepared[QuantPart.DECODER] = prepare_on_cpu(
                decoder, dec_example, compile_spec, quant_config, act_observer, keep_fp32,
                quantizer_cls=quantizer_cls)
        print(f"export+prepare on cpu: {time.time() - t0:.0f}s", flush=True)

    if device != "cpu":
        # Keep GPU float math close to the CPU reference.
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    encoder, decoder = encoder.to(device), decoder.to(device)
    if device != "cpu":
        for graph in prepared.values():
            move_graph_module(graph, device)
    enc_example = tuple(t.to(device) for t in enc_example)
    dec_example = tuple(t.to(device) for t in dec_example)
    greedy = GreedyDecoder(args.model_id, args.dec_len, device)

    def make_transcribe(enc, dec):
        @torch.no_grad()
        def transcribe(audio):
            tokens, hit_limit = greedy.decode(dec, enc(log_mel(audio, device)))
            return greedy.text(tokens), hit_limit, len(tokens) + 1
        return transcribe

    if Stage.FP32_STATIC in stages:
        finish(evaluate(Stage.FP32_STATIC.value, make_transcribe(encoder, decoder), eval_pairs,
                        data.load_audio_torchaudio, normalizer, out_dir))

    if Stage.INT8 not in stages:
        return

    encoder_q, decoder_q, n_dec_batches = encoder, decoder, 0
    t0 = time.time()
    if calib_mode is CalibMode.RANDOM:
        if QuantPart.ENCODER in quant_parts:
            encoder_q = calibrate_and_convert("encoder", prepared[QuantPart.ENCODER], [enc_example], device)
        if QuantPart.DECODER in quant_parts:
            decoder_q = calibrate_and_convert("decoder", prepared[QuantPart.DECODER], [dec_example], device)
            n_dec_batches = 1
    elif feed is CalibFeed.AUTOREGRESSIVE:
        # FQ-ViT's procedure (apo_build._whisper_calibrator): drive the observers
        # through the real greedy decode of the unquantized model. The
        # observer-instrumented graphs still compute in FP32, so they generate the
        # FP32 transcript while every decode step is observed, and the decoder
        # is fed unquantized encoder states.
        enc_prep, dec_prep = prepared.get(QuantPart.ENCODER), prepared.get(QuantPart.DECODER)
        t_cal = time.time()
        with torch.no_grad():
            for audio_path, _ in cal_pairs:
                mel = log_mel(data.load_audio_torchaudio(audio_path), device)
                states = enc_prep(mel) if enc_prep is not None else encoder(mel)
                if dec_prep is not None:
                    tokens, _ = greedy.decode(dec_prep, states)
                    n_dec_batches += len(tokens) + 1
        print(f"  calibrate[autoregressive] on {device}: {time.time() - t_cal:.1f}s over "
              f"{len(cal_pairs)} utterances, {n_dec_batches} decoder steps", flush=True)
        if enc_prep is not None:
            encoder_q = convert_pt2e(enc_prep)
        if dec_prep is not None:
            decoder_q = convert_pt2e(dec_prep)
        if PC_REWRITE:
            from pc_rewrite import rewrite_per_channel_activations
            for nm, g in (("encoder", encoder_q), ("decoder", decoder_q)):
                if g is not None:
                    print(f"  pc_rewrite[{nm}]: {rewrite_per_channel_activations(g)} per-channel activation sites", flush=True)
    else:
        mels = [log_mel(data.load_audio_torchaudio(p), device) for p, _ in cal_pairs]
        if QuantPart.ENCODER in quant_parts:
            encoder_q = calibrate_and_convert("encoder", prepared[QuantPart.ENCODER], [(m,) for m in mels], device)
        # Decoder calibration inputs: the FP32 model's own transcript (what the
        # decoder is fed at run time) on top of the *quantized* encoder's output
        # (what it will see on device). Each utterance contributes the full
        # sequence and one random prefix, since during decoding most calls see a
        # partially filled buffer.
        if QuantPart.DECODER in quant_parts:
            dec_batches = []
            with torch.no_grad():
                for mel in mels:
                    tokens, _ = greedy.decode(decoder, encoder(mel))
                    tokens = greedy.prompt + tokens
                    enc_q = encoder_q(mel)
                    prefix = rng.randint(len(greedy.prompt), len(tokens))
                    dec_batches.append((greedy.padded_ids(tokens), enc_q))
                    dec_batches.append((greedy.padded_ids(tokens[:prefix]), enc_q))
            decoder_q = calibrate_and_convert("decoder", prepared[QuantPart.DECODER], dec_batches, device)
            n_dec_batches = len(dec_batches)
    parts_label = "+".join(sorted(p.value for p in quant_parts))
    print(f"quantized {parts_label} ({calib_mode.value}): {time.time() - t0:.0f}s, "
          f"decoder calibration batches={n_dec_batches}", flush=True)

    label = (f"{quant_config.value}-{act_observer.value}-{calib_mode.value}-"
             f"{feed.value}-{parts_label}")
    if keep_fp32:
        label += "-fp32_" + "+".join(k.value for k in keep_fp32)
    if args.mask_aware:
        label += "-maskaware"
    if args.prec_rules:
        label += f"-rules[{args.prec_rules}]"
    if args.ln_newton_steps is not None:
        label += f"-newton{args.ln_newton_steps}"
    if args.ln_dual_q is not None:
        label += f"-dual{args.ln_dual_q}"
    if args.ln_dual_k is not None:
        label += f"-dualk{args.ln_dual_k:g}"
    if args.embedding_bits:
        label += f"-emb{args.embedding_bits}"
    if args.pc_rewrite:
        label += "-pcrw"
    if args.dump_ln_scales:
        dump_ln_scales(out_dir / f"ln_scales_{label}.txt", encoder_q, decoder_q, encoder, decoder,
                       [log_mel(data.load_audio_torchaudio(a), device) for a, _ in cal_pairs[:8]], greedy, device)
    result = evaluate(label, make_transcribe(encoder_q, decoder_q), eval_pairs,
                      data.load_audio_torchaudio, normalizer, out_dir)
    result.update({"calib_mode": calib_mode.value, "quant_parts": parts_label,
                   "quant_config": quant_config.value, "device": device,
                   "act_observer": act_observer.value, "calib_feed": feed.value,
                   "keep_fp32": [k.value for k in keep_fp32],
                   "n_cal": len(cal_pairs), "cal_split": "dev-clean", "eval_split": "test-clean"})
    finish(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stages", nargs="+", choices=[s.value for s in Stage],
                        default=[s.value for s in Stage])
    parser.add_argument("--calib-mode", choices=[m.value for m in CalibMode],
                        default=CalibMode.REAL.value)
    parser.add_argument("--n-cal", type=int, default=200)
    parser.add_argument("--n-eval", type=int, default=200)
    parser.add_argument("--dec-len", type=int, default=128,
                        help="Static decoder length (prompt + generated tokens).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quant-config", choices=[c.value for c in QuantConfig],
                        default=QuantConfig.A8W8.value)
    parser.add_argument("--act-observer", choices=[o.value for o in ActObserver],
                        default=ActObserver.HISTOGRAM.value)
    parser.add_argument("--calib-feed", choices=[f.value for f in CalibFeed],
                        default=CalibFeed.PARALLEL.value)
    parser.add_argument("--keep-fp32", nargs="*", choices=[k.value for k in KeepFp32], default=[])
    parser.add_argument("--pc-rewrite", action="store_true", default=False,
                        help="Rewrite per-channel activation Q/DQ into per-tensor + int32 MULs (pc_rewrite.py) before evaluation.")
    parser.add_argument("--embedding-bits", type=int, default=None, choices=[8, 16],
                        help="Quantize the decoder token-embedding table (per-tensor symmetric) instead of leaving it fp32.")
    parser.add_argument("--ln-dual-k", type=float, default=None,
                        help="Dual-range rsqrt with the fine table sized by K x median per-token variance (rules: layer_norm\\.fine$=a32w8@clamp;layer_norm\\.fine$=a16w8e16).")
    parser.add_argument("--ln-dual-q", type=float, default=None,
                        help="Dual-range rsqrt in NewtonLayerNorm: fine table sized by this per-token variance quantile (e.g. 0.995).")
    parser.add_argument("--ln-newton-steps", type=int, default=None,
                        help="Swap nn.LayerNorm for NewtonLayerNorm (keepdim formulation): rsqrt table seed + N int32 Newton steps; 0 = swap only.")
    parser.add_argument("--dump-ln-scales", action="store_true", default=False,
                        help="Write per-LayerNorm quantization grids vs fp32 token variance to the out dir.")
    parser.add_argument("--mask-aware", action="store_true", default=False,
                        help="Keep additive attention masks out of the softmax-input grid (MaskAwareQuantizer).")
    parser.add_argument("--prec-rules", default=None,
                        help='Ordered "regex=a16w8|a8w8|fp32[:observer];..." rules on the deepest module FQN.')
    parser.add_argument("--quant-parts", nargs="+", choices=[p.value for p in QuantPart],
                        default=[p.value for p in QuantPart])
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
    parser.add_argument("--target", default="ethos-u85-256")
    parser.add_argument("--system-config", default="Ethos_U85_SYS_DRAM_Low")
    parser.add_argument("--memory-mode", default="Dedicated_Sram")
    parser.add_argument("--vela-config",
                        default=str(HERE.parent / "scripts/vela/default_vela.ini"))
    main(parser.parse_args())
