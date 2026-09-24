"""Calibration-only precision advisor: how many quantization levels does the *ordinary* token get?

For every quantize node of a calibrated + converted graph, capture the fp32 tensor entering it and
report, per tensor: the grid step s (per-channel: the per-channel steps), the per-token max |x| over
the channel axis, and derived
    levels_med = median over tokens of (row max / s)   -- resolution left for a typical token
    levels_p5  = 5th percentile of the same             -- resolution of the weakest tokens
    zero_frac  = fraction of elements that round to 0
    span       = global max / median row max            -- how much a few tokens inflate the grid
No labels are used: everything comes from the calibration images / utterances, so the flags can
be raised before any test-set evaluation. Tensors with few levels for the median token are where
int8 per-tensor grids fail (LN internals, residual streams with massive activations, rsqrt inputs).
"""
import argparse, functools, statistics, sys, torch
from pathlib import Path
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import whisper_ptq_eval as pe, mask_aware_quantizer as maq

Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
Q_PC = torch.ops.quantized_decomposed.quantize_per_channel.default


def parse_rules(rules_str, observer):
    rules = []
    for item in [r for r in rules_str.split(";") if r]:
        rx, cfg = item.rsplit("=", 1); cfg, _, ops = cfg.partition("@"); cfg, _, obs = cfg.partition(":")
        rules.append((rx, None if cfg == "fp32" else pe.QuantConfig(cfg).build(pe.ActObserver(obs or observer)),
                      set(ops.split(",")) if ops else None))
    return rules


def _short(n):
    return str(n.target).replace("torch.ops.", "").replace("aten.", "").replace(".default", "").replace(".Tensor", "")


def fqn(node):
    """Module path + op of the first consumer that is not a quantize/dequantize node (Q nodes carry no stack)."""
    seen, frontier = set(), [node]
    while frontier:
        n = frontier.pop(0)
        if n in seen:
            continue
        seen.add(n)
        if n is not node and "quantize" not in str(n.target):
            stack = n.meta.get("nn_module_stack")
            path = list(stack.values())[-1][0] if stack else "?"
            return f"{path} -> {_short(n)}"
        frontier.extend(n.users)
    return "?"


def main(args):
    if args.model.startswith("whisper"):
        wrapper = pe.load_module_from_path("whisper_executorch_wrapper", HERE / "whisper_executorch_wrapper.py")
        data = pe.load_module_from_path("librispeech_data", HERE / "data/librispeech_data.py")
        model, example = wrapper._build(f"openai/{args.model}", wrapper.WhisperPart.ENCODER, 128)
        pairs = data.gather_librispeech_files(args.librispeech_dir, "dev-clean", args.n_cal)
        batches = [(pe.log_mel(data.load_audio_torchaudio(p), args.device),) for p, _ in pairs]
        model_name = args.model
    else:
        import fqvit_models
        from fqvit_models import imagenet_data as data
        model = fqvit_models.build_model(args.model)
        example = (torch.randn(args.batch_size, 3, 224, 224),)
        batches = [(b.to(args.device),) for b in data._load_imgs(args.n_cal, None, "train", shuffle=True, seed=0,
                                                                  data_dir=args.imagenet_dir, model_name=args.model, batch_size=args.batch_size)]
        model_name = args.model
    if args.ln_newton_steps is not None:
        from newton_layernorm import swap_layernorms
        swap_layernorms(model, args.ln_newton_steps)
    cs = pe.EthosUCompileSpec("ethos-u85-256", system_config="Ethos_U85_SYS_DRAM_Low", memory_mode="Dedicated_Sram")
    rules = parse_rules(args.prec_rules, args.act_observer)
    qcls = functools.partial(maq.MixedPrecisionQuantizer, rules=rules) if rules else pe.EthosUQuantizer
    prepared = pe.prepare_on_cpu(model, example, cs, pe.QuantConfig(args.quant_config), pe.ActObserver(args.act_observer), [], quantizer_cls=qcls)
    pe.move_graph_module(prepared, args.device)
    with torch.no_grad():
        for b in batches:
            prepared(*b)
    gm = convert_pt2e(prepared)
    pe.move_graph_module(gm, args.device)

    rows = []
    class Cap(torch.fx.Interpreter):
        def run_node(self, n):
            if n.op == "call_function" and n.target in (Q, Q_PC) and n.args[0].op != "get_attr":
                x = self.env[n.args[0]].detach().float()
                if n.target is Q:
                    s = torch.tensor(float(n.args[1]), device=x.device); dtype = n.args[5]
                else:
                    s = getattr(gm, n.args[1].target).float().to(x.device); dtype = n.args[6]
                c = x.shape[-1] if x.ndim >= 2 else 1
                flat = x.reshape(-1, c).abs()
                rowmax = flat.max(1).values
                step = s if n.target is Q else s.reshape(1, -1)
                lv = (flat.max(1).values.unsqueeze(1) / step).min(1).values if n.target is Q_PC else rowmax / s
                rows.append(dict(name=n.name, fqn=fqn(n), dtype=str(dtype).replace("torch.", ""),
                                 scheme="pc" if n.target is Q_PC else "pt", shape=tuple(x.shape),
                                 step=float(s.min()), levels_med=float(lv.median()), levels_p5=float(lv.kthvalue(max(1, int(0.05 * lv.numel()))).values),
                                 zero_frac=float((flat < step / 2).float().mean()), span=float(rowmax.max() / rowmax.median().clamp(min=1e-12))))
            return super().run_node(n)
    with torch.no_grad():
        Cap(gm).run(*batches[0])
    bits = {"int8": 8, "int16": 16, "int32": 32}
    for r in rows:
        r["eff_bits"] = max(0.0, torch.log2(torch.tensor(max(r["levels_med"], 1e-9))).item() + 1)  # signed
    rows.sort(key=lambda r: r["levels_med"])
    print(f"=== {model_name} {args.quant_config} rules=[{args.prec_rules}] n_cal={args.n_cal}: {len(rows)} activation quantize nodes")
    print(f"{'levels_med':>10s} {'p5':>8s} {'eff_bits':>8s} {'zero%':>6s} {'span':>7s} {'dtype':>6s} sch  fqn / node")
    shown = 0
    for r in rows:
        if r["levels_med"] >= args.threshold and shown >= args.top:
            break
        flag = "!!" if r["levels_med"] < args.threshold else "  "
        print(f"{flag}{r['levels_med']:8.1f} {r['levels_p5']:8.1f} {r['eff_bits']:8.1f} {100 * r['zero_frac']:6.1f} {r['span']:7.1f} {r['dtype']:>6s} {r['scheme']}  {r['fqn']} / {r['name']}")
        shown += 1
    n_flag = sum(1 for r in rows if r["levels_med"] < args.threshold)
    print(f"flagged (< {args.threshold} levels for the median token): {n_flag} / {len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="FQ-ViT model name or whisper-tiny/base/small (encoder)")
    ap.add_argument("--quant-config", default="a8w8")
    ap.add_argument("--act-observer", default="histogram")
    ap.add_argument("--prec-rules", default="")
    ap.add_argument("--ln-newton-steps", type=int, default=None)
    ap.add_argument("--n-cal", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=16.0, help="flag tensors whose median token has fewer levels than this")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imagenet-dir", default="/home/shared/ImageNet")
    ap.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
    main(ap.parse_args())
