"""One-pass, calibration-set-only resolution of every LayerNorm's internal tensors.

The fp32 model runs once over the calibration set (no export, no quantizer, no labels). A hook on every LayerNorm
rebuilds, from the LayerNorm input, the tensors the Arm backend's decomposition quantizes (DecomposeLayerNormPass /
DecomposeVarPass / DecomposeMeanDimPass):

    sum_x = sum(x)   mean = sum_x / C   xc = x - mean   sq = xc * xc   sum_sq = sum(sq)   var = sum_sq / C
    var_eps = var + eps   rsqrt = 1 / sqrt(var_eps)   xhat = xc * rsqrt   yw = xhat * weight   out = yw + bias

and keeps, per tensor, the min / max over the observer population and a log2 histogram of |value| over the median
population. The scale a MinMax observer would assign is then computed for each activation width, with the floors
of the Arm configs:
    int8  per-tensor affine    s = (max(max, 0) - min(min, 0)) / 255, floored at 2^-16 (arm_quantizer.py:119)
    int16 per-tensor symmetric s = max|x| / 32767,                     floored at 2^-12 (arm_quantizer.py:304)
    int32 per-tensor symmetric s = max|x| / (2^31 - 1),                floored at 2^-20 (core/quant.py a32w8)
and N = median|x| / s (levels between zero and the median value), zero% = share of values with |x| < s / 2.
This equals the calibrated scale for every MinMax-observed tensor (Whisper);
the vision Default graphs use Arm's HistogramObserver, whose scale is <= the MinMax one (N_hist >= N here).

Populations (what the observer sees vs. what the median is taken over), as helper/diagnostics/levels_report.py:
  vision    both: every token of every calibration image.
  Whisper   encoder: both = every frame (padding frames included).
            decoder: observer = both calibration passes of whisper_eval.py (full transcript + one random prefix,
            EOT-padded to 128 positions); median / zero% = the decoded positions of the full-transcript pass.
Whisper decoder inputs are the fp32 encoder's states (the runner feeds the quantized encoder's; fp32 here).

Output: one CSV row per (part, LayerNorm, tensor) with min, max, median_abs, R = max|x| / median|x|, and for
w in 8 / 16 / 32: s_w, N_w, zero_w, floored_w (the scale sits on the eps floor).
"""
import argparse
import csv
import math
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # fine_grained_transformer_u85/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "helper"))
import core.quant as cq  # noqa: E402

DEC_LEN = 128  # whisper_eval.py
TENSORS = ("sum_x", "mean", "xc", "sq", "sum_sq", "var", "var_eps", "rsqrt", "xhat", "yw", "out")
LO, HI, NBINS = -60.0, 30.0, 9000  # log2 |x| histogram: 0.01 octave bins (0.7 % relative)
GRIDS = {8: ("affine", 255, 2 ** -16), 16: ("symmetric", 32767, 2 ** -12), 32: ("symmetric", 2 ** 31 - 1, 2 ** -20)}


class Stat:
    def __init__(self, device):
        self.mn, self.mx = math.inf, -math.inf
        self.hist = torch.zeros(NBINS, dtype=torch.float64, device=device)
        self.n_zero = 0  # exact zeros in the median population (log2 undefined)
        self.n = 0

    def observe(self, t):
        self.mn = min(self.mn, t.min().item()); self.mx = max(self.mx, t.max().item())

    def sample(self, t):
        a = t.abs().reshape(-1).double()
        nz = a > 0
        self.n_zero += int((~nz).sum()); self.n += a.numel()
        self.hist += torch.histc(a[nz].log2().clamp(LO, HI - 1e-9), bins=NBINS, min=LO, max=HI)

    def quantile_abs(self, q):
        """|x| at quantile q of the median population (bin centre)."""
        k = q * self.n - self.n_zero
        if k <= 0:
            return 0.0
        c = torch.cumsum(self.hist, 0)
        i = int(torch.searchsorted(c, torch.tensor(k, dtype=c.dtype, device=c.device)))
        return 2 ** (LO + (min(i, NBINS - 1) + 0.5) * (HI - LO) / NBINS)

    def frac_below(self, v):
        if v <= 0:
            return 0.0
        i = int((math.log2(v) - LO) / (HI - LO) * NBINS)
        below = self.n_zero + float(self.hist[: max(0, min(i, NBINS))].sum())
        return below / max(self.n, 1)


def scale(width, mn, mx):
    kind, q, eps = GRIDS[width]
    s = (max(mx, 0.0) - min(mn, 0.0)) / q if kind == "affine" else max(abs(mn), abs(mx)) / q
    return max(s, eps), s < eps


def main(args):
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    stats = {}  # (part, ln name, tensor) -> Stat
    cur = {"part": "", "observe": True, "sample": True, "n": None}

    def hook(part, name):
        def fn(mod, inp, out):
            if not (cur["observe"] or cur["sample"]):
                return
            x = inp[0].detach().float()
            c = x.shape[-1]
            t = {}
            t["sum_x"] = x.sum(-1, keepdim=True); t["mean"] = t["sum_x"] / c
            t["xc"] = x - t["mean"]; t["sq"] = t["xc"] * t["xc"]
            t["sum_sq"] = t["sq"].sum(-1, keepdim=True); t["var"] = t["sum_sq"] / c
            t["var_eps"] = t["var"] + mod.eps; t["rsqrt"] = torch.rsqrt(t["var_eps"])
            t["xhat"] = t["xc"] * t["rsqrt"]
            t["yw"] = t["xhat"] * mod.weight.float() if mod.weight is not None else t["xhat"]
            t["out"] = t["yw"] + mod.bias.float() if mod.bias is not None else t["yw"]
            for k in TENSORS:
                st = stats.setdefault((part, name, k), Stat(x.device))
                if cur["observe"]:
                    st.observe(t[k])
                if cur["sample"]:
                    v = t[k]
                    if cur["n"] is not None:  # decoder: the decoded positions (axis 1 = DEC_LEN)
                        v = v[:, : cur["n"]]
                    st.sample(v)
        return fn

    def register(module, part):
        n = 0
        for name, m in module.named_modules():
            if isinstance(m, torch.nn.LayerNorm):
                m.register_forward_hook(hook(part, name)); n += 1
        return n

    with torch.no_grad():
        if args.model.startswith("whisper"):
            from transformers import AutoConfig
            from helper import whisper_io as wio
            wrapper = cq.load_module_from_path("whisper_executorch_wrapper", ROOT / "helper/whisper_executorch_wrapper.py")
            data = cq.load_module_from_path("librispeech_data", ROOT / "helper/data/librispeech_data.py")
            model_id = f"openai/{args.model}"
            wio.N_MELS = AutoConfig.from_pretrained(model_id).num_mel_bins
            encoder, _ = wrapper._build(model_id, wrapper.WhisperPart.ENCODER, DEC_LEN)
            decoder, _ = wrapper._build(model_id, wrapper.WhisperPart.DECODER, DEC_LEN)
            encoder.to(args.device).eval(); decoder.to(args.device).eval()
            register(encoder, "encoder"); register(decoder, "decoder")
            greedy = wio.GreedyDecoder(model_id, DEC_LEN, args.device)
            rng = random.Random(args.seed)
            pairs = data.gather_librispeech_files(args.librispeech_dir, "dev-clean", args.n_cal)
            for path, _ in pairs:
                mel = wio.log_mel(data.load_audio_torchaudio(path), args.device)
                cur.update(observe=True, sample=True, n=None)
                states = encoder(mel)
                cur.update(observe=False, sample=False)
                tokens, _ = greedy.decode(decoder, states)
                tokens = list(greedy.prompt) + tokens
                prefix = rng.randint(len(greedy.prompt), len(tokens))
                cur.update(observe=True, sample=True, n=len(tokens))
                decoder(greedy.padded_ids(tokens), states)
                cur.update(observe=True, sample=False, n=None)
                decoder(greedy.padded_ids(tokens[:prefix]), states)
        else:
            import fqvit_models
            from fqvit_models import imagenet_data as data
            model = fqvit_models.build_model(args.model).to(args.device).eval()
            register(model, "")
            for b in data._load_imgs(args.n_cal, None, "train", shuffle=True, seed=0, data_dir=args.imagenet_dir,
                                     model_name=args.model, batch_size=args.batch_size):
                cur.update(observe=True, sample=True, n=None)
                model(b.to(args.device))

    out = Path(args.csv); out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "part", "layernorm", "tensor", "n_median_pop", "min", "max", "median_abs", "p5_abs", "R"]
    for w in GRIDS:
        cols += [f"s_{w}", f"N_{w}", f"zero_{w}", f"floored_{w}"]
    with open(out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader()
        for (part, name, k), st in stats.items():
            med = st.quantile_abs(0.5)
            row = dict(model=args.model, part=part, layernorm=name, tensor=k, n_median_pop=st.n, min=st.mn, max=st.mx,
                       median_abs=med, p5_abs=st.quantile_abs(0.05), R=max(abs(st.mn), abs(st.mx)) / med if med > 0 else math.inf)
            for w in GRIDS:
                s, floored = scale(w, st.mn, st.mx)
                row.update({f"s_{w}": s, f"N_{w}": med / s, f"zero_{w}": st.frac_below(s / 2), f"floored_{w}": int(floored)})
            wr.writerow(row)
    print(f"{args.model}: {len({(p, n) for p, n, _ in stats})} LayerNorms x {len(TENSORS)} tensors -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="FQ-ViT model name (deit_*, swin_*) or whisper-<size>")
    ap.add_argument("--n-cal", type=int, required=True, help="calibration images / utterances (production: 1000 / 200)")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="whisper decoder calibration: the random-prefix draw (whisper_eval.py --seed)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imagenet-dir", default="/home/shared/ImageNet")
    ap.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
    ap.add_argument("--csv", required=True)
    main(ap.parse_args())
