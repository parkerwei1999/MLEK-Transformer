"""Median-grid span (MGS = M / s) of every block-level activation site of a vision model, calibration set only.

One fp32 pass over the calibration images (no quantizer, no labels). Sites, per transformer block:
    fc2          the MLP's fc2 output
    res_attn     the residual add after attention (= norm2's input)
    res_mlp      the residual add after the MLP (= the block output)
    ln_in / ln_xc / ln_sq / ln_var / ln_out   per LayerNorm (norm1, norm2, PatchMerging norm, final norm): its input,
                 x - mean, (x - mean)^2, the per-token variance, its output
M = median |x| over every element (per-token tensors: every token); s = the scale a MinMax observer with the Arm
default floor assigns at the Default width, INT8 per-tensor affine: (max(max, 0) - min(min, 0)) / 255, floored at
2^-16 (arm_quantizer.py:119). Also MGS_ch = median over channels of the channel's max |x| / s (the median channel's
peak, the PCS view). Vision Default graphs use Arm's HistogramObserver, whose scale is <= this one (MGS_hist >= MGS).
Output: CSV, one row per site.
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # fine_grained_transformer_u85/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "helper"))

LO, HI, NBINS = -60.0, 30.0, 9000  # log2 |x| histogram, 0.01 octave bins


class Stat:
    def __init__(self, c, device):
        self.mn, self.mx = math.inf, -math.inf
        self.ch = torch.zeros(c, device=device)
        self.hist = torch.zeros(NBINS, dtype=torch.float64, device=device)
        self.n_zero = 0
        self.n = 0

    def update(self, x):
        x = x.detach().float()
        a = x.abs()
        self.mn = min(self.mn, x.min().item()); self.mx = max(self.mx, x.max().item())
        self.ch = torch.maximum(self.ch, a.reshape(-1, a.shape[-1]).amax(0))
        v = a.reshape(-1).double()
        nz = v > 0
        self.n_zero += int((~nz).sum()); self.n += v.numel()
        self.hist += torch.histc(v[nz].log2().clamp(LO, HI - 1e-9), bins=NBINS, min=LO, max=HI)

    def median(self):
        k = 0.5 * self.n - self.n_zero
        if k <= 0:
            return 0.0
        c = torch.cumsum(self.hist, 0)
        i = int(torch.searchsorted(c, torch.tensor(k, dtype=c.dtype, device=c.device)))
        return 2 ** (LO + (min(i, NBINS - 1) + 0.5) * (HI - LO) / NBINS)


def where(name):
    """(stage, block) from a module path: [layers.<i>.]blocks.<j> (DeiT: stage 0) / layers.<i>.downsample / patch_embed / norm."""
    m = re.match(r"(?:layers\.(\d+)\.)?blocks\.(\d+)", name)
    if m:
        return int(m.group(1) or 0), int(m.group(2))
    m = re.match(r"layers\.(\d+)\.downsample", name)
    if m:
        return int(m.group(1)), -1
    return (-1, -1) if name.startswith("patch_embed") else (99, -1)


def main(args):
    import fqvit_models
    from fqvit_models import imagenet_data as data
    model = fqvit_models.build_model(args.model).to(args.device).eval()
    stats = {}  # (module, kind) -> Stat

    def upd(module, kind, x):
        stats.setdefault((module, kind), Stat(x.shape[-1], x.device)).update(x)

    def ln_hook(name):
        def fn(mod, inp, out):
            x = inp[0].detach().float()
            mean = x.mean(-1, keepdim=True); xc = x - mean; sq = xc * xc
            upd(name, "ln_in", x); upd(name, "ln_xc", xc); upd(name, "ln_sq", sq)
            upd(name, "ln_var", sq.mean(-1, keepdim=True)); upd(name, "ln_out", out)
            if name.endswith(".norm2"):  # norm2's input is the residual add after attention
                upd(name[: -len(".norm2")], "res_attn", x)
        return fn

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.register_forward_hook(ln_hook(name))
        elif name.endswith(".mlp.fc2") and isinstance(m, torch.nn.Linear):
            m.register_forward_hook(lambda mod, i, o, k=name[: -len(".mlp.fc2")]: upd(k, "fc2", o))
        elif re.fullmatch(r"(?:layers\.\d+\.)?blocks\.\d+", name):
            m.register_forward_hook(lambda mod, i, o, k=name: upd(k, "res_mlp", o))

    with torch.no_grad():
        for b in data._load_imgs(args.n_cal, None, "train", shuffle=True, seed=args.seed, data_dir=args.imagenet_dir,
                                 model_name=args.model, batch_size=args.batch_size):
            model(b.to(args.device))

    out = Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "stage", "block", "module", "site", "min", "max", "M", "s_int8", "floored", "MGS", "MGS_ch"])
        for (module, kind), st in sorted(stats.items(), key=lambda kv: (where(kv[0][0]), kv[0][0], kv[0][1])):
            raw = (max(st.mx, 0.0) - min(st.mn, 0.0)) / 255
            s = max(raw, 2 ** -16)
            med = st.median()
            stage, block = where(module)
            w.writerow([args.model, stage, block, module, kind, f"{st.mn:.6g}", f"{st.mx:.6g}", f"{med:.6g}", f"{s:.6g}",
                        int(raw < 2 ** -16), f"{med / s:.6g}", f"{st.ch.median().item() / s:.6g}"])
    print(f"{args.model}: {len(stats)} sites -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="FQ-ViT model name (swin_* / deit_*)")
    ap.add_argument("--n-cal", type=int, required=True, help="calibration images (production: 1000)")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed of the ImageNet train split: which calibration images")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imagenet-dir", default="/home/shared/ImageNet")
    ap.add_argument("--csv", required=True)
    main(ap.parse_args())
