"""Per-channel range of every residual-stream read (LayerNorm inputs, Swin PatchMerging inputs) on calibration
images, fp32, no accuracy run. The vision counterpart of the Whisper per-token variance census, on the channel
axis: a per-tensor INT8 grid sized by the max channel leaves the median channel 127 / R_c codes.

Sites: every LayerNorm input ("ln"), every residual add input = each block's output stream ("res", the tensor the
PCS residual spec quantizes), Swin PatchMerging input, and every fc2 output ("fc2", the MLP output added into the
residual stream). Output: <out-dir>/<model>_channel_census.csv (per-site summary) and
<out-dir>/<model>_channel_max.pt ({site: per-channel max|x| tensor}) for plotting the channel profile.
"""
import argparse, csv, sys
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "helper"))
import fqvit_models
from fqvit_models import imagenet_data as data

ap = argparse.ArgumentParser()
ap.add_argument("--model-name", required=True); ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--imagenet-dir", default="/home/shared/ImageNet"); ap.add_argument("--out-dir", required=True); ap.add_argument("--device", default="cuda")
args = ap.parse_args()
model = fqvit_models.build_model(args.model_name).to(args.device).eval()
chan_max = {}
def pre(name):
    def fn(mod, inp):
        x = inp[0].detach().float().abs().reshape(-1, inp[0].shape[-1])   # [tokens, C]
        m = x.amax(0).cpu(); chan_max[name] = torch.maximum(chan_max[name], m) if name in chan_max else m
    return fn
def post(name):
    def fn(mod, inp, out):
        x = out.detach().float().abs().reshape(-1, out.shape[-1])
        m = x.amax(0).cpu(); chan_max[name] = torch.maximum(chan_max[name], m) if name in chan_max else m
    return fn
for name, m in model.named_modules():
    kind = type(m).__name__
    if isinstance(m, torch.nn.LayerNorm) or kind == "PatchMerging":
        m.register_forward_pre_hook(pre(name + "|ln"))
    if kind in ("Block", "SwinTransformerBlock"):           # block output = the residual stream after both adds
        m.register_forward_hook(post(name + "|res"))
    if name.endswith("mlp.fc2"):
        m.register_forward_hook(post(name + "|fc2"))
with torch.no_grad():
    for b in data._load_imgs(args.n, None, "train", shuffle=True, seed=0, data_dir=args.imagenet_dir, model_name=args.model_name, batch_size=50):
        model(b.to(args.device))
out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True); rows = []
torch.save(chan_max, out / f"{args.model_name}_channel_max.pt")
for name, m in chan_max.items():
    mx, med = m.max().item(), m.median().item(); R = mx / max(med, 1e-12)
    rows.append((args.model_name, name, m.numel(), f"{mx:.4g}", f"{med:.4g}", f"{R:.3g}", f"{127 / R:.2f}"))
with open(out / f"{args.model_name}_channel_census.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "site", "n_channels", "max_abs", "median_channel_max", "R_c", "codes_median_channel"]); w.writerows(rows)
print("  by kind:", {k: f"worst R_c {max(float(r[5]) for r in rows if r[1].endswith('|'+k)):.3g}, <8-code sites {sum(float(r[6])<8 for r in rows if r[1].endswith('|'+k))}/{sum(r[1].endswith('|'+k) for r in rows)}" for k in ("ln","res","fc2")})
Rs = [float(r[5]) for r in rows]; codes = [float(r[6]) for r in rows]
worst = max(rows, key=lambda r: float(r[5]))
print(f"{args.model_name}: {len(rows)} sites on {args.n} cal images | worst R_c {worst[5]} at {worst[1]} (median channel gets {worst[6]} codes) | "
      f"sites with < 8 codes for the median channel: {sum(c < 8 for c in codes)}/{len(rows)} | median R_c over sites {sorted(Rs)[len(Rs)//2]:.3g}", flush=True)
