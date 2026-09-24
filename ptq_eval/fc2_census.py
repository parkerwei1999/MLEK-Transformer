"""fp32 census of Whisper fc2 outputs (and layer outputs = residual stream, for contrast).

Answers whether fc2's output outliers are token-type (a few tokens carry huge rows, so a
per-channel int8 grid still crushes everything else) or channel-type (a few channels are large
for every token, which per-channel scales absorb). Per layer, over N greedy-decoded dev-clean
utterances:
  max        largest |x|
  R_tok      max over tokens of row-Linf / median over tokens of row-Linf
  R_ch       max over channels of col-Linf / median over channels of col-Linf
  z_pc8      fraction of elements that round to 0 on a per-channel int8 grid (s_c = colmax_c/127)
  z_pt16     same on a per-tensor int16 grid (s = max/32767)
  tok@max    position of the max row (encoder: frame; decoder: generated step, 0 = first prompt token)
"""
import argparse, statistics, sys
from collections import defaultdict
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import whisper_ptq_eval as pe  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model-id", default="openai/whisper-base")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--device", default="cuda")
ap.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
args = ap.parse_args()

data = pe.load_module_from_path("librispeech_data", HERE / "data/librispeech_data.py")
model = WhisperForConditionalGeneration.from_pretrained(args.model_id).to(args.device).eval()
pairs = data.gather_librispeech_files(args.librispeech_dir, "dev-clean", args.n)

captured = defaultdict(list)  # name -> list of [T, C] per utterance (accumulated across decode steps)
current = {}
def hook(name):
    def fn(mod, inp, out):
        y = out[0] if isinstance(out, tuple) else out
        current.setdefault(name, []).append(y.detach().float().reshape(-1, y.shape[-1]).cpu())
    return fn
for i, layer in enumerate(model.model.encoder.layers):
    layer.fc2.register_forward_hook(hook(f"enc{i}.fc2")); layer.register_forward_hook(hook(f"enc{i}.out"))
for i, layer in enumerate(model.model.decoder.layers):
    layer.fc2.register_forward_hook(hook(f"dec{i}.fc2")); layer.register_forward_hook(hook(f"dec{i}.out"))

with torch.no_grad():
    for path, _ in pairs:
        current.clear()
        mel = pe.log_mel(data.load_audio_torchaudio(path), args.device)
        model.generate(mel, max_new_tokens=128, num_beams=1, do_sample=False, language="en", task="transcribe")
        for k, chunks in current.items():
            captured[k].append(torch.cat(chunks, 0))

def stats(mats):
    rows = []
    for x in mats:
        a = x.abs()
        rmax, cmax = a.max(1).values, a.max(0).values
        r_tok = (rmax.max() / rmax.median()).item()
        r_ch = (cmax.max() / cmax.median()).item()
        z_pc8 = (a < (cmax / 127 / 2)).float().mean().item()
        z_pt16 = (a < (a.max() / 32767 / 2)).float().mean().item()
        # PTF as implemented (ptf_observer.py): base = smallest per-channel scale, alpha_c clamped to [0, 12],
        # so a channel whose max exceeds cmax.min() * 2^12 is CLIPPED at that ceiling.
        # ptf_observer.py also floors the base scale at eps = 2^-12, i.e. the ceiling is at least 127.0.
        ceiling = max(cmax.min().item(), 127 * 2 ** -12) * 4096
        span = (cmax.max() / cmax.min()).item()
        n_clip = int((a > ceiling).sum())
        clip_ratio = (a.max() / ceiling).item()
        rows.append((a.max().item(), r_tok, r_ch, z_pc8, z_pt16, int(rmax.argmax()), span, n_clip, clip_ratio, cmax.min().item()))
    med = lambda i: statistics.median(r[i] for r in rows)
    return (max(r[0] for r in rows), med(1), med(2), med(3), med(4), sorted(r[5] for r in rows),
            med(6), sum(r[7] for r in rows), max(r[8] for r in rows), min(r[9] for r in rows))
print(f"=== {args.model_id} n={len(pairs)}  (medians over utterances; tok@max lists all utterances; "
      f"span = max/min per-channel max; clip = elements above the alpha=12 PTF ceiling, max/ceiling)")
print(f"{'layer':10s} {'max':>8s} {'R_tok':>7s} {'R_ch':>7s} {'z_pc8':>6s} {'z_pt16':>6s} {'span':>9s} {'cmin':>7s} {'n_clip':>6s} {'max/ceil':>8s}  tok@max")
for name in sorted(captured, key=lambda n: (n[:3], int(''.join(c for c in n.split('.')[0] if c.isdigit())), n)):
    mx, rt, rc, z8, z16, toks, span, nclip, cr, cmin = stats(captured[name])
    print(f"{name:10s} {mx:8.1f} {rt:7.1f} {rc:7.1f} {z8:6.2f} {z16:6.3f} {span:9.0f} {cmin:7.4f} {nclip:6d} {cr:8.2f}  {toks}")
