"""Calibrate-once check (MinMax recipes): for several recipes of one vision model, the quantization parameters after
(a) the normal calibration over N batches and (b) core.calib_stats.apply() of states collected once over the same
N batches must be identical, and so must the top-1 on a few validation batches.

    python3 tests/calib_stats_check.py --model deit_tiny_patch16_224 --n-cal 50
"""
import argparse
import copy
import functools
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "helper"))
import core.calib_stats as cs  # noqa: E402
import core.mixed_precision_quantizer as maq  # noqa: E402
import core.quant as cq  # noqa: E402
from core.pcs_decompose import rewrite_per_channel_activations  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e  # noqa: E402

LN = r"norm\d?$=a32w8@sub,mul,sum.dim_IntList,add;norm\d?$=a16w8"
PCS = r"blocks\.\d+$=a16inpc8symout@add"
RECIPES = [("Default A8W8", "a8w8", "minmax", ""), ("Default A16W8", "a16w8", "minmax", ""),
           ("A8W8 + LN-32", "a8w8", "minmax", LN), ("Ours A8W8 (PCS)", "a8w8", "minmax", f"{PCS};{LN}"),
           ("A8W8 + LN-16", "a8w8", "minmax", r"norm\d?$=a16w8"), ("A16W8 + LN-32", "a16w8", "minmax", LN)]
HISTOGRAM_RAISES = ("Default A8W8", "a8w8", "histogram", "")  # out of scope: apply() must refuse it


def qparams(gm):
    out = []
    for n in gm.graph.nodes:
        if n.op == "call_function" and "quantize_per" in str(n.target) and "dequantize" not in str(n.target):
            s, z = n.args[1], n.args[2]
            if isinstance(s, torch.fx.Node):
                s, z = getattr(gm, s.target), getattr(gm, z.target)
            out.append((n.name, torch.as_tensor(s).double().flatten().cpu(), torch.as_tensor(z).double().flatten().cpu()))
    return out


def main(args):
    import fqvit_models
    from fqvit_models import imagenet_data as data
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    model = fqvit_models.build_model(args.model).eval()
    example = (torch.randn(args.batch_size, 3, 224, 224),)
    spec = cq.quantizer_compile_spec()
    cal = [(b,) for b in data._load_imgs(args.n_cal, None, "train", shuffle=True, seed=0, data_dir=args.imagenet_dir,
                                         model_name=args.model, batch_size=args.batch_size)]
    val = [b for _, b in zip(range(args.val_batches), data.val_loader(None, args.batch_size, deterministic_order=False,
                                                                       data_dir=args.imagenet_dir, model_name=args.model, num_workers=4))]
    t0 = time.time()
    stats = cs.collect(model, example, spec, cal, args.device)
    print(f"collect: {len(stats['nodes'])} nodes, {time.time() - t0:.0f}s", flush=True)
    ok = True
    for name, qc, obs, rules in RECIPES:
        qcls = functools.partial(maq.MixedPrecisionQuantizer, rules=cq.parse_rules(rules, cq.ActObserver(obs), None)) if rules else cq.EthosUQuantizer
        graphs = {}
        for mode in ("calibrated", "applied"):
            prep = cq.prepare_on_cpu(copy.deepcopy(model), example, spec, cq.QuantConfig(qc), cq.ActObserver(obs), [], quantizer_cls=qcls)
            cq.move_graph_module(prep, args.device)
            t0 = time.time()
            with torch.no_grad():
                if mode == "calibrated":
                    for b in cal:
                        prep(b[0].to(args.device))
                    counts = ""
                else:
                    counts = cs.apply(prep, stats)
            gm = convert_pt2e(prep); rewrite_per_channel_activations(gm, skipped=[])
            graphs[mode] = (gm, time.time() - t0, counts)
        a, b = qparams(graphs["calibrated"][0]), qparams(graphs["applied"][0])
        names_ok = [x[0] for x in a] == [x[0] for x in b]
        diff = max(((x[1] - y[1]).abs().max() / x[1].abs().max().clamp(min=1e-30)).item() for x, y in zip(a, b)) if names_ok else float("nan")
        zdiff = max((x[2] - y[2]).abs().max().item() for x, y in zip(a, b)) if names_ok else float("nan")
        hits = {}
        with torch.no_grad():
            for mode, (gm, _, _) in graphs.items():
                hits[mode] = sum((gm(x.to(args.device)).argmax(-1).cpu() == y).sum().item() for x, y in val)
        same = names_ok and diff == 0 and zdiff == 0 and hits["calibrated"] == hits["applied"]
        ok &= same
        print(f"{'OK  ' if same else 'DIFF'} {name:26s} quantize nodes {len(a)} | max rel scale diff {diff:.2e}, zp diff {zdiff:.0f} | "
              f"top-1 hits {hits['calibrated']} vs {hits['applied']} on {sum(len(y) for _, y in val)} | "
              f"calibrate {graphs['calibrated'][1]:.1f}s, apply {graphs['applied'][1]:.1f}s | {graphs['applied'][2]}", flush=True)
    name, qc, obs, rules = HISTOGRAM_RAISES
    prep = cq.prepare_on_cpu(copy.deepcopy(model), example, spec, cq.QuantConfig(qc), cq.ActObserver(obs), [])
    try:
        cs.apply(prep, stats); ok = False; print("DIFF histogram recipe was accepted (must raise)", flush=True)
    except NotImplementedError:
        print("OK   histogram recipe refused", flush=True)
    print("ALL IDENTICAL" if ok else "MISMATCH", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deit_tiny_patch16_224")
    ap.add_argument("--n-cal", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--val-batches", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imagenet-dir", default="/home/shared/ImageNet")
    main(ap.parse_args())
