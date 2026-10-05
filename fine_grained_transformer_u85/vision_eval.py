"""Post-training quantization of an FQ-ViT vision model through the stock
ExecuTorch Arm quantizer, with ImageNet Top-1 / Top-5 of the framework fake-quant
graph, for the DeiT / Swin models of FQ-ViT.

The model classes are FQ-ViT's own (3rdparty/FQ-ViT through helper/fqvit_models, which also holds the
ImageNet loader), so the FP32 network is the one behind the FQ-ViT results for that model. The
quantization recipe and the CPU-prepare / CUDA-run split come from core/quant.py.

All quantized variants are evaluated in one pass over the validation set, so
each image is decoded once and every model_variant sees identical inputs.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import functools

import torch
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "helper"))  # fqvit_models: adapter over the official FQ-ViT in 3rdparty/FQ-ViT
import fqvit_models
from fqvit_models import imagenet_data as data

import core.quant as cq
from core.pcs_decompose import rewrite_per_channel_activations
import core.recipes as recipes

from core.newton_layernorm import swap_layernorms, collect_var_quantiles
import core.mixed_precision_quantizer as maq
import core.calib_stats as calib_stats

MODEL_NAMES = ["deit_tiny_patch16_224", "deit_small_patch16_224", "deit_base_patch16_224",
               "swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224", "swin_base_patch4_window7_224"]


def main(args) -> None:
    # Model and directory setup
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if args.n_eval % args.batch_size or args.n_cal % args.batch_size:
        raise ValueError("n_cal and n_eval must be multiples of the static batch size")
    model = fqvit_models.build_model(args.model_name)


    # LayerNorm module swap for LN-Newton and LN-FineLUT
    if args.ln_newton_steps is not None or args.ln_dual_k is not None:
        steps = args.ln_newton_steps or 0
        cmap = None
        if args.ln_dual_k is not None:
            model.to(device)
            cal_c = data._load_imgs(args.n_cal, None, "train", shuffle=True, seed=args.seed, data_dir=args.imagenet_dir,
                                    model_name=args.model_name, batch_size=args.batch_size)
            def calibrate():
                for b in cal_c:
                    model(b.to(device))
            cmap = collect_var_quantiles(model, calibrate, None, k=args.ln_dual_k)
            model.cpu()
            print(f"  fine-lut: k={args.ln_dual_k}, {len(cmap)} LayerNorms, c range {min(cmap.values()):.3g}..{max(cmap.values()):.3g}", flush=True)
        print(f"  NewtonLayerNorm: {swap_layernorms(model, steps, cmap)} LayerNorms swapped, {steps} step(s), fine-lut k={args.ln_dual_k}", flush=True)
    
    # Export preparation
    torch.manual_seed(args.seed)
    example = (torch.randn(args.batch_size, 3, 224, 224),)
    compile_spec = cq.quantizer_compile_spec()

    # Fine-grained quantization preparation
    quantizer_cls = cq.EthosUQuantizer
    if args.prec_rules:
        rules = cq.parse_rules(args.prec_rules, cq.ActObserver(args.act_observers[0]), args.ln_dual_k)
        quantizer_cls = functools.partial(maq.MixedPrecisionQuantizer, rules=rules)
    variants = {}
    t0 = time.time()
    for quant, observer in [(q, o) for q in args.quant_configs for o in args.act_observers]:
        label = (f"{quant}-{observer}" + (f"-rules[{args.prec_rules}]" if args.prec_rules else ""))
        print(f"preparing {label}", flush=True)
        variants[label] = {
            "graph": cq.prepare_on_cpu(model, example, compile_spec, cq.QuantConfig(quant),
                                       cq.ActObserver(observer), [], quantizer_cls=quantizer_cls),
            "quant_config": quant, "act_observer": observer,
        }
    print(f"export+prepare on cpu: {time.time() - t0:.0f}s for {len(variants)} variants", flush=True)

    # Make sure no TF32 is used
    if device != "cpu":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    # Mimicking FQ-ViT calibration set: shuffled train images, seed 0, batches of 100.
    cal_batches = data._load_imgs(args.n_cal, None, "train", shuffle=True, seed=args.seed,
                                  data_dir=args.imagenet_dir, model_name=args.model_name,
                                  batch_size=args.batch_size)

    # Reuse calibration stats
    stats = None
    if args.calib_stats:  # calibrate once per (model, calibration set); every recipe reuses the observer states
        meta = {"model": args.model_name, "n_cal": args.n_cal, "seed": args.seed, "batch_size": args.batch_size,
                "ln_newton_steps": args.ln_newton_steps, "ln_dual_k": args.ln_dual_k}
        path = Path(args.calib_stats)
        if path.exists():
            stats = torch.load(path, weights_only=False)
            assert stats.get("meta") == meta, f"{path} was collected for {stats.get('meta')}, this run is {meta}"
            print(f"calib-stats: loaded {path} ({len(stats['nodes'])} nodes)", flush=True)
        else:
            t0 = time.time()
            stats = calib_stats.collect(model, example, compile_spec, [(b,) for b in cal_batches], device)
            stats["meta"] = meta
            path.parent.mkdir(parents=True, exist_ok=True); torch.save(stats, path)
            print(f"calib-stats: collected {len(stats['nodes'])} nodes in {time.time() - t0:.0f}s -> {path}", flush=True)
    
    # To GPU for calibration and evaluation
    if device != "cpu":
        for model_variant in variants.values():
            cq.move_graph_module(model_variant["graph"], device)

    # Apply calibration stats or run calibration
    model = model.to(device)
    t0 = time.time()
    with torch.no_grad():
        for label, model_variant in variants.items():
            if stats is not None:
                counts = calib_stats.apply(model_variant["graph"], stats)
                print(f"  calib-stats[{label}]: {counts}", flush=True)
            else:
                for batch in cal_batches:
                    model_variant["graph"](batch.to(device))

            # PT2E
            model_variant["graph"] = convert_pt2e(model_variant["graph"])

            # Rewrite per-channel activations
            ptf = []  # PTF sites
            n_pc = rewrite_per_channel_activations(model_variant["graph"], skipped=ptf)
            if n_pc or ptf:
                print(f"  pcs_decompose[{label}]: {n_pc} per-channel activation sites rewritten, "
                      f"{len(ptf)} PTF sites left as fake-quant (not lowerable)", flush=True)
    print(f"calibrate+convert on {device}: {time.time() - t0:.0f}s, "
          f"{len(cal_batches)} batches of {args.batch_size}", flush=True)

    # Validation
    loader = data.val_loader(None, args.batch_size, deterministic_order=(args.n_eval == 50000),
                             data_dir=args.imagenet_dir, model_name=args.model_name,
                             num_workers=args.num_workers)
    runners = {"fp32": model, **{label: v["graph"] for label, v in variants.items()}}
    correct = {label: [0, 0] for label in runners}
    seen, skipped, t0 = 0, 0, time.time()
    assert args.eval_skip % args.batch_size == 0, "--eval-skip must be a multiple of --batch-size"
    with torch.no_grad():
        for images, targets in loader:
            # Resume in the same shuffled order if --eval-skip is used
            if skipped < args.eval_skip:  # the same shuffled order, a later block of it (a second / third 1k subset)
                skipped += images.shape[0]
                continue
            if seen >= args.n_eval:
                break
            images, targets = images.to(device), targets.to(device)
            for label, runner in runners.items():
                top5 = runner(images).topk(5, dim=-1).indices
                correct[label][0] += (top5[:, 0] == targets).sum().item()
                correct[label][1] += (top5 == targets[:, None]).any(-1).sum().item()
            seen += images.shape[0]
            if seen % 5000 == 0 or seen == args.n_eval:
                line = " ".join(f"{label}={c[0] / seen * 100:.2f}" for label, c in correct.items())
                print(f"  {seen}/{args.n_eval} ({time.time() - t0:.0f}s) top1: {line}", flush=True)

    results = {"args": vars(args), "n_eval": seen, "stages": []}
    for label, (top1, top5) in correct.items():
        meta = {k: v for k, v in variants.get(label, {}).items() if k != "graph"}
        results["stages"].append({"label": label, "top1_pct": top1 / seen * 100,
                                  "top5_pct": top5 / seen * 100, "n_eval": seen,
                                  "n_cal": 0 if label == "fp32" else args.n_cal, **meta})
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", required=True, choices=MODEL_NAMES)
    parser.add_argument("--recipe", default=None, help="named recipe from core/recipes.py (e.g. \"Ours A8W8\"); sets the quantization flags itself")
    parser.add_argument("--quant-configs", nargs="+", choices=["a8w8", "a16w8"], default=["a8w8"])
    parser.add_argument("--act-observers", nargs="+", choices=["histogram", "minmax"],
                        default=["histogram"])
    parser.add_argument("--prec-rules", default=None,
                        help='Ordered "regex=config[:observer][@op,...]" rules matched on the deepest module FQN')
    parser.add_argument("--n-cal", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=50000)
    parser.add_argument("--eval-skip", type=int, default=0,
                        help="skip the first N images of the evaluation order (a subset: the seed-0 shuffle), e.g. 1000 for a second 1k subset")
    parser.add_argument("--calib-stats", default=None,
                        help="observer-state file (core/calib_stats.py): loaded when it exists, else collected from the calibration "
                             "images and saved; every recipe of the run is then calibrated from it without running the images again")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Static batch size of the exported graphs (FQ-ViT calibrates in batches of 100).")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ln-dual-k", type=float, default=None, help="fine table sized by K x median per-token variance; the rules must give the fine table and mask the kmedian observer (core/recipes.py)")
    parser.add_argument("--ln-newton-steps", type=int, default=None,
                        help="Swap LayerNorm modules for NewtonLayerNorm with N Newton steps (0 = swap only).")
    parser.add_argument("--imagenet-dir", default="/home/shared/ImageNet")
    main(recipes.parse_args(parser, "vision", lambda a: a.model_name))
