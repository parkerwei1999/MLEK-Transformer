"""Post-training quantization of an FQ-ViT vision model through the stock
ExecuTorch Arm quantizer, with ImageNet Top-1 / Top-5 of the framework fake-quant
graph. Model-selectable successor of ../../deit_ethosu/ptq_eval/deit_ptq_eval.py,
which stays frozen because its hash is pinned in the Access-Quant Fill Log.

The model class, checkpoint, preprocessing and image selection are FQ-ViT's own
(models/, experiments/imagenet_data.py), loaded from its checkout, so the FP32
network is the one behind the FQ-ViT receipts for that model. The
quantization recipe and the CPU-prepare / CUDA-run split are shared with
../../whisper_ethosu/ptq_eval/whisper_ptq_eval.py.

All quantized variants are evaluated in one pass over the validation set, so
each image is decoded once and every variant sees identical inputs.
"""
import argparse
import importlib.util
import itertools
import json
import sys
import time
from enum import Enum
from pathlib import Path

import torch
from executorch.backends.arm.ethosu import EthosUCompileSpec
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e

HERE = Path(__file__).resolve().parent
MODEL_NAMES = ["deit_tiny_patch16_224", "deit_small_patch16_224", "deit_base_patch16_224",
               "swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224", "swin_base_patch4_window7_224"]


def load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class KeepFp32(Enum):
    """Module types left unquantized, to test whether they cause an accuracy loss."""
    LAYERNORM = "layernorm"

    def module_type(self):
        # FQ-ViT builds its models with its nn.LayerNorm subclass; the Arm filter matches
        # the exact class recorded in nn_module_stack, not base classes.
        from models.ptq import QIntLayerNorm
        return {KeepFp32.LAYERNORM: QIntLayerNorm}[self]


def main(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if args.n_eval % args.batch_size or args.n_cal % args.batch_size:
        raise ValueError("n_cal and n_eval must be multiples of the static batch size")

    shared = load_by_path("whisper_ptq_eval", Path(args.shared_script))
    if args.fqvit_dir:  # optional: build from a live FQ-ViT checkout instead of the vendored copy
        fqvit = Path(args.fqvit_dir)
        sys.path.insert(0, str(fqvit))
        from config import Config
        import models as fqvit_models
        data = load_by_path("imagenet_data", fqvit / "experiments/imagenet_data.py")
        constructor = getattr(fqvit_models, args.model_name)
        model = constructor(pretrained=True, quant=False, calibrate=False, cfg=Config()).eval()
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fqvit_models
        from fqvit_models import imagenet_data as data
        model = fqvit_models.build_model(args.model_name)

    if args.ln_newton_steps is not None or args.ln_dual_q is not None:
        from newton_layernorm import swap_layernorms, collect_var_quantiles
        steps = args.ln_newton_steps or 0
        cmap = None
        if args.ln_dual_q is not None:
            model.to(device)
            cal_c = data._load_imgs(args.n_cal, None, "train", shuffle=True, seed=args.seed, data_dir=args.imagenet_dir,
                                    model_name=args.model_name, batch_size=args.batch_size)
            def run_c():
                for b in cal_c:
                    model(b.to(device))
            cmap = collect_var_quantiles(model, run_c, args.ln_dual_q)
            model.cpu()
            print(f"  dual-range rsqrt: q={args.ln_dual_q}, {len(cmap)} LayerNorms, c range {min(cmap.values()):.3g}..{max(cmap.values()):.3g}", flush=True)
        print(f"  NewtonLayerNorm: {swap_layernorms(model, steps, cmap)} LayerNorms swapped, {steps} step(s), dual={args.ln_dual_q}", flush=True)
    torch.manual_seed(args.seed)
    example = (torch.randn(args.batch_size, 3, 224, 224),)
    compile_spec = EthosUCompileSpec(
        args.target, system_config=args.system_config, memory_mode=args.memory_mode,
        extra_flags=[], config_ini=args.vela_config)

    quantizer_cls = shared.EthosUQuantizer
    if args.mask_aware or args.prec_rules:
        maq = load_by_path("mask_aware_quantizer", Path(args.shared_script).parent / "mask_aware_quantizer.py")
        quantizer_cls = maq.MaskAwareQuantizer
    if args.prec_rules:
        import functools
        rules = []
        for item in args.prec_rules.split(";"):
            regex, cfg = item.rsplit("=", 1)
            # "a16w8:minmax@sum.dim_IntList,mul" = observer and an optional op-type filter.
            cfg, _, ops = cfg.partition("@")
            cfg, _, obs = cfg.partition(":")
            rules.append((regex, None if cfg == "fp32" else shared.QuantConfig(cfg).build(
                shared.ActObserver(obs or args.act_observers[0])), set(ops.split(",")) if ops else None))
        quantizer_cls = functools.partial(maq.MixedPrecisionQuantizer, rules=rules,
                                          mask_threshold=maq.MASK_THRESHOLD if args.mask_aware else None)
    variants = {}
    t0 = time.time()
    for quant, observer, keep in itertools.product(args.quant_configs, args.act_observers,
                                                   args.keep_fp32_sets):
        keep_fp32 = [KeepFp32(k) for k in keep.split("+") if k != "none"]
        label = (f"{quant}-{observer}-fp32_{keep}" + ("-maskaware" if args.mask_aware else "")
                 + (f"-a16[{args.a16_modules}]" if args.a16_modules else "")
                 + (f"-rules[{args.prec_rules}]" if args.prec_rules else ""))
        print(f"preparing {label}", flush=True)
        variants[label] = {
            "graph": shared.prepare_on_cpu(model, example, compile_spec, shared.QuantConfig(quant),
                                           shared.ActObserver(observer), keep_fp32,
                                           quantizer_cls=quantizer_cls,
                                           a16_module_regex=args.a16_modules),
            "quant_config": quant, "act_observer": observer, "keep_fp32": [k.value for k in keep_fp32],
        }
    print(f"export+prepare on cpu: {time.time() - t0:.0f}s for {len(variants)} variants", flush=True)

    if device != "cpu":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        for variant in variants.values():
            shared.move_graph_module(variant["graph"], device)
    model = model.to(device)

    # FQ-ViT calibration set: shuffled train images, seed 0, batches of 100.
    cal_batches = data._load_imgs(args.n_cal, None, "train", shuffle=True, seed=args.seed,
                                  data_dir=args.imagenet_dir, model_name=args.model_name,
                                  batch_size=args.batch_size)
    t0 = time.time()
    with torch.no_grad():
        for label, variant in variants.items():
            for batch in cal_batches:
                variant["graph"](batch.to(device))
            variant["graph"] = convert_pt2e(variant["graph"])
            if args.pc_rewrite:
                from pc_rewrite import rewrite_per_channel_activations
                print(f"  pc_rewrite[{label}]: {rewrite_per_channel_activations(variant['graph'], x_bits_below=args.pc_rewrite_bits)} per-channel activation sites (x grid s_base/2^{args.pc_rewrite_bits})", flush=True)
    print(f"calibrate+convert on {device}: {time.time() - t0:.0f}s, "
          f"{len(cal_batches)} batches of {args.batch_size}", flush=True)

    # Full validation set in file order; a subset uses FQ-ViT's seed-0 shuffle so it
    # is not just the first alphabetical classes.
    loader = data.val_loader(None, args.batch_size, deterministic_order=(args.n_eval == 50000),
                             data_dir=args.imagenet_dir, model_name=args.model_name,
                             num_workers=args.num_workers)
    runners = {"fp32": model, **{label: v["graph"] for label, v in variants.items()}}
    correct = {label: [0, 0] for label in runners}
    seen, t0 = 0, time.time()
    with torch.no_grad():
        for images, targets in loader:
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
    parser.add_argument("--quant-configs", nargs="+", choices=["a8w8", "a16w8"], default=["a8w8", "a16w8"])
    parser.add_argument("--act-observers", nargs="+", choices=["histogram", "minmax"],
                        default=["histogram", "minmax"])
    parser.add_argument("--keep-fp32-sets", nargs="+", choices=["none", "layernorm"],
                        default=["none", "layernorm"])
    parser.add_argument("--prec-rules", default=None,
                        help='Ordered "regex=a16w8|a8w8|fp32;..." rules matched on the deepest module FQN '
                             "(MixedPrecisionQuantizer); mask-aware is implied.")
    parser.add_argument("--a16-modules", default=None,
                        help="Regex over module FQNs; matching modules are annotated with the a16w8 config.")
    parser.add_argument("--mask-aware", action="store_true", default=False,
                        help="Keep additive attention masks out of the softmax-input grid (MaskAwareQuantizer).")
    parser.add_argument("--n-cal", type=int, default=1000)
    parser.add_argument("--n-eval", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Static batch size of the exported graphs (FQ-ViT calibrates in batches of 100).")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pc-rewrite-bits", type=int, default=None, help="pc_rewrite int32 activation grid = s_base / 2^bits (default adaptive)")
    parser.add_argument("--ln-dual-q", type=float, default=None,
                        help="Dual-range rsqrt in NewtonLayerNorm: fine table sized by this per-token variance quantile.")
    parser.add_argument("--ln-newton-steps", type=int, default=None,
                        help="Swap LayerNorm modules for NewtonLayerNorm with N Newton steps (0 = swap only).")
    parser.add_argument("--pc-rewrite", action="store_true", default=False,
                        help="Rewrite per-channel activation Q/DQ into per-tensor + int32 MULs (pc_rewrite.py) before evaluation.")
    parser.add_argument("--imagenet-dir", default="/home/shared/ImageNet")
    parser.add_argument("--fqvit-dir", default=None,
                        help="Optional FQ-ViT checkout; default uses the vendored fqvit_models package.")
    parser.add_argument("--shared-script",
                        default=str(HERE / "whisper_ptq_eval.py"))
    parser.add_argument("--target", default="ethos-u85-256")
    parser.add_argument("--system-config", default="Ethos_U85_SYS_DRAM_Low")
    parser.add_argument("--memory-mode", default="Dedicated_Sram")
    parser.add_argument("--vela-config",
                        default=str(HERE.parent / "scripts/vela/default_vela.ini"))
    main(parser.parse_args())
