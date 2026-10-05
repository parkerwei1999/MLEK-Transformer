"""Calibration-only precision advisor: how many quantization levels does the *ordinary* token get?

For every quantize node of a calibrated + converted graph, capture the fp32 tensor entering it and
report, per tensor: the grid step s (per-channel: the per-channel steps), the per-token max |x| over
the channel axis, and derived
    levels_med = median over tokens of (row max / s)   -- resolution left for a typical token
    levels_p5  = 5th percentile of the same             -- resolution of the weakest tokens
    zero_frac  = fraction of elements that round to 0
    span       = global max / median row max            -- how much a few tokens inflate the grid
No labels are used: everything comes from the calibration images / utterances, so the flags can
be raised before any test-set evaluation. Tensors with few levels for the median token are the
candidates for a wider grid (e.g. LayerNorm internals, residual streams, rsqrt inputs).
"""
import argparse, functools, statistics, sys, torch
DEC_LEN = 128  # whisper_eval.py
from pathlib import Path
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # fine_grained_transformer_u85/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "helper"))  # fqvit_models: adapter over the official FQ-ViT in 3rdparty/FQ-ViT
import core.quant as cq, core.mixed_precision_quantizer as maq
from helper import whisper_io as wio

Q = torch.ops.quantized_decomposed.quantize_per_tensor.default
Q_PC = torch.ops.quantized_decomposed.quantize_per_channel.default


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


def calibrate(model, example, batches, args, rules):
    cs = cq.quantizer_compile_spec()
    qcls = functools.partial(maq.MixedPrecisionQuantizer, rules=rules) if rules else cq.EthosUQuantizer
    prepared = cq.prepare_on_cpu(model, example, cs, cq.QuantConfig(args.quant_config), cq.ActObserver(args.act_observer), [], quantizer_cls=qcls)
    cq.move_graph_module(prepared, args.device)
    with torch.no_grad():
        for b in batches:
            prepared(*b)
    gm = convert_pt2e(prepared)
    cq.move_graph_module(gm, args.device)
    return gm


def capture(gm, batches, args, real_len=None):
    """Per quantize node: the per-token row max of |x| and a random subsample of the elements, both from the fp32
    tensor entering the node, over `batches`. real_len[i]: for batch i, keep only the first real_len[i] positions of
    tensors laid out (1, DEC_LEN, ...) (the static decoder's decoded positions; padding positions are dropped)."""
    acc = {}
    gen = torch.Generator(device="cpu").manual_seed(0)
    cur = {"n": None}

    class Cap(torch.fx.Interpreter):
        def run_node(self, n):
            if n.op == "call_function" and n.target in (Q, Q_PC) and n.args[0].op != "get_attr":
                x = self.env[n.args[0]].detach().float()
                if cur["n"] is not None:  # the position axis: the first DEC_LEN-long axis preceded only by size-1 axes
                    for ax in range(x.ndim - 1):
                        if x.shape[ax] == DEC_LEN:
                            x = x.narrow(ax, 0, cur["n"]); break
                        if x.shape[ax] != 1:
                            break
                if n.target is Q:
                    s = torch.tensor(float(n.args[1]), device=x.device); dtype = n.args[5]
                else:
                    s = getattr(gm, n.args[1].target).float().to(x.device); dtype = n.args[6]
                c = x.shape[-1] if x.ndim >= 2 else 1
                flat = x.reshape(-1, c).abs()
                step = s if n.target is Q else s.reshape(1, -1)
                a = acc.setdefault(n.name, dict(node=n, s=s, dtype=dtype, shape=tuple(x.shape), rowmax=[], elems=[], n_zero=0, n_el=0))
                a["rowmax"].append(flat.max(1).values.cpu())
                idx = torch.randperm(flat.numel(), generator=gen)[: args.sample_per_batch]
                a["elems"].append(flat.reshape(-1)[idx.to(flat.device)].cpu())
                a["n_zero"] += int((flat < step / 2).sum()); a["n_el"] += flat.numel()
            return super().run_node(n)
    with torch.no_grad():
        for i, b in enumerate(batches):
            cur["n"] = real_len[i] if real_len is not None else None
            Cap(gm).run(*b)
    return acc


def main(args):
    rules = cq.parse_rules(args.prec_rules, cq.ActObserver(args.act_observer), None)
    parts = []  # (label, gm, capture batches, real_len)
    if args.model.startswith("whisper"):
        import random
        from transformers import AutoConfig
        wrapper = cq.load_module_from_path("whisper_executorch_wrapper", ROOT / "helper/whisper_executorch_wrapper.py")
        data = cq.load_module_from_path("librispeech_data", ROOT / "helper/data/librispeech_data.py")
        model_id = f"openai/{args.model}"
        wio.N_MELS = AutoConfig.from_pretrained(model_id).num_mel_bins
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False  # as whisper_eval.py
        encoder, enc_example = wrapper._build(model_id, wrapper.WhisperPart.ENCODER, DEC_LEN)
        pairs = data.gather_librispeech_files(args.librispeech_dir, "dev-clean", args.n_cal)
        mels = [wio.log_mel(data.load_audio_torchaudio(p), args.device) for p, _ in pairs]
        enc_q = calibrate(encoder, enc_example, [(m,) for m in mels], args, rules)
        if args.whisper_part in ("encoder", "both"):
            parts.append(("encoder", enc_q, [(m,) for m in mels[: args.cap_batches]], None))
        if args.whisper_part in ("decoder", "both"):
            # Decoder calibration exactly as whisper_eval.py: fp32 greedy transcript on the fp32 encoder; per utterance the
            # full transcript and one random prefix (Random(seed)), both EOT-padded to DEC_LEN, on the quantized encoder's states.
            decoder, dec_example = wrapper._build(model_id, wrapper.WhisperPart.DECODER, DEC_LEN)
            enc_fp, _ = wrapper._build(model_id, wrapper.WhisperPart.ENCODER, DEC_LEN)
            enc_fp.to(args.device).eval()
            greedy = wio.GreedyDecoder(model_id, DEC_LEN, args.device)
            rng = random.Random(args.seed)
            cal, cap, real_len = [], [], []
            dec_fp = None
            with torch.no_grad():
                for i, mel in enumerate(mels):
                    if dec_fp is None:
                        dec_fp, _ = wrapper._build(model_id, wrapper.WhisperPart.DECODER, DEC_LEN); dec_fp.to(args.device).eval()
                    tokens, _ = greedy.decode(dec_fp, enc_fp(mel))
                    tokens = list(greedy.prompt) + tokens
                    states = enc_q(mel)
                    prefix = rng.randint(len(greedy.prompt), len(tokens))
                    full = (greedy.padded_ids(tokens), states)
                    cal += [full, (greedy.padded_ids(tokens[:prefix]), states)]
                    if i < args.cap_batches:
                        cap.append(full); real_len.append(len(tokens))
            del enc_fp, dec_fp
            dec_q = calibrate(decoder, dec_example, cal, args, rules)
            parts.append(("decoder", dec_q, cap, real_len))
    else:
        import fqvit_models
        from fqvit_models import imagenet_data as data
        model = fqvit_models.build_model(args.model)
        example = (torch.randn(args.batch_size, 3, 224, 224),)
        batches = [(b.to(args.device),) for b in data._load_imgs(args.n_cal, None, "train", shuffle=True, seed=0,
                                                                  data_dir=args.imagenet_dir, model_name=args.model, batch_size=args.batch_size)]
        if args.ln_newton_steps is not None:
            from core.newton_layernorm import swap_layernorms
            swap_layernorms(model, args.ln_newton_steps)
        parts.append(("", calibrate(model, example, batches, args, rules), batches[: args.cap_batches], None))
    model_name = args.model

    all_rows = []
    for part, gm, cap, real_len in parts:
        acc = capture(gm, cap, args, real_len)
        rows = []
        for name, a in acc.items():
            n, s = a["node"], a["s"]
            src = n.args[0]
            producer = _short(src) if src.op == "call_function" else src.op
            is_square = src.op == "call_function" and _short(src) == "mul" and len(src.args) == 2 and src.args[0] is src.args[1]
            rowmax = torch.cat(a["rowmax"]); el = torch.cat(a["elems"])
            if n.target is Q_PC:
                lv = rowmax / float(s.min())  # conservative: the smallest channel step
            else:
                lv = rowmax / s.cpu()
            stack = src.meta.get("nn_module_stack") if src.op == "call_function" else None
            path = list(stack.values())[-1][0] if stack else "?"
            rows.append(dict(name=name, fqn=fqn(n), path=path, producer=producer, is_square=int(is_square),
                             dtype=str(a["dtype"]).replace("torch.", ""), scheme="pc" if n.target is Q_PC else "pt", shape=a["shape"],
                             step=float(s.min()), levels_med=float(lv.median()), levels_p5=float(lv.kthvalue(max(1, int(0.05 * lv.numel()))).values),
                             levels_elem=float(el.median() / s.min()), r_elem=float(el.max() / el.median().clamp(min=1e-12)),
                             zero_frac=a["n_zero"] / max(a["n_el"], 1), span=float(rowmax.max() / rowmax.median().clamp(min=1e-12)),
                             _elems=el))
        # x_c^2 tensors: also the levels the fp32 (x - mean)^2 would get on this grid, from the x - mean tensor of the same
        # LayerNorm, so the x_c grid's own rounding does not hide the x_c^2 grid's.
        by_path = {}
        for r in rows:
            if r["producer"] == "sub":
                by_path.setdefault(r["path"], r)
        for r in rows:
            r["levels_sq_fp32"] = float((by_path[r["path"]]["_elems"] ** 2).median() / r["step"]) if r["is_square"] and r["path"] in by_path else float("nan")
            r["zero_sq_fp32"] = float(((by_path[r["path"]]["_elems"] ** 2) < r["step"] / 2).float().mean()) if r["is_square"] and r["path"] in by_path else float("nan")
        for r in rows:
            del r["_elems"]
        for r in rows:
            r["eff_bits"] = max(0.0, torch.log2(torch.tensor(max(r["levels_med"], 1e-9))).item() + 1)  # signed
        rows.sort(key=lambda r: r["levels_med"])
        print(f"=== {model_name}{(" " + part) if part else ""} {args.quant_config} rules=[{args.prec_rules}] n_cal={args.n_cal} captured batches={len(cap)}{" (decoded positions only)" if real_len else ""}: {len(rows)} activation quantize nodes")
        print(f"{'levels_med':>10s} {'p5':>8s} {'elem_med':>8s} {'eff_bits':>8s} {'zero%':>6s} {'span':>7s} {'dtype':>6s} sch  fqn / node")
        shown = 0
        for r in rows:
            if r["levels_med"] >= args.threshold and shown >= args.top:
                break
            flag = "!!" if r["levels_med"] < args.threshold else "  "
            print(f"{flag}{r['levels_med']:8.1f} {r['levels_p5']:8.1f} {r['levels_elem']:8.2f} {r['eff_bits']:8.1f} {100 * r['zero_frac']:6.1f} {r['span']:7.1f} {r['dtype']:>6s} {r['scheme']}  {r['fqn']} / {r['name']}")
            shown += 1
        n_flag = sum(1 for r in rows if r["levels_med"] < args.threshold)
        print(f"flagged (< {args.threshold} levels for the median token): {n_flag} / {len(rows)}")
        for r in rows:
            r["part"] = part
        all_rows += rows
    rows = all_rows
    if args.csv:
        import csv
        keys = [k for k in rows[0] if k != "shape"] + ["shape"]
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["model", "quant_config", "act_observer", "prec_rules", "n_cal"] + keys)
            w.writeheader()
            for r in rows:
                w.writerow(dict(model=model_name, quant_config=args.quant_config, act_observer=args.act_observer, prec_rules=args.prec_rules, n_cal=args.n_cal, **{k: (str(r[k]) if k == "shape" else r[k]) for k in keys}))
        print(f"rows -> {args.csv}")


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
    ap.add_argument("--cap-batches", type=int, default=10, help="calibration batches to capture statistics from (the grid itself is calibrated on all --n-cal)")
    ap.add_argument("--sample-per-batch", type=int, default=20000, help="elements sampled per tensor per captured batch for the element-wise median / max")
    ap.add_argument("--csv", default="", help="write every quantize node's row here (levels_med, levels_elem, levels_sq_fp32, zero_frac, r_elem, ...)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imagenet-dir", default="/home/shared/ImageNet")
    ap.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
    ap.add_argument("--whisper-part", default="encoder", choices=["encoder", "decoder", "both"],
                    help="whisper: which graph to report; the decoder is calibrated as whisper_eval.py does (needs the quantized encoder)")
    ap.add_argument("--seed", type=int, default=0, help="whisper decoder calibration: the random-prefix draw (whisper_eval.py --seed)")
    main(ap.parse_args())
