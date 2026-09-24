"""fp32 census of per-token LayerNorm variance across Whisper sizes (calibration-only, no quantization).

Every integer LayerNorm has to hold sigma^2 (or 1/sqrt) of *all* tokens in one fixed-point format
(int16 TABLE input here, fixed-headroom accumulator in a FISR design). With an attention sink the
per-token variance range R = var_max / var_median can reach 1e4, so a 16-bit grid sized by the sink
leaves the typical token 32767 / R levels. Per LayerNorm we report var median / p5 / max, R, the
implied int16 levels for the median token, and the relative rsqrt error of a table seed at that
resolution (half a level), which is what one Newton step has to remove.
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
ap.add_argument("--model-id", default="openai/whisper-small")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--device", default="cuda")
ap.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
ap.add_argument("--top", type=int, default=6)
args = ap.parse_args()

data = pe.load_module_from_path("librispeech_data", HERE / "data/librispeech_data.py")
model = WhisperForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.float32).to(args.device).eval()
pairs = data.gather_librispeech_files(args.librispeech_dir, "dev-clean", args.n)
var_by_ln = defaultdict(list)   # name -> list of per-token variances (all utterances, all decode steps)
resid_max = defaultdict(float)
def hook(name):
    def fn(mod, inp, out):
        x = inp[0].detach().float()
        v = x.var(-1, unbiased=False).reshape(-1)
        var_by_ln[name].append(v.cpu())
        resid_max[name] = max(resid_max[name], x.abs().max().item())
    return fn
for name, m in model.named_modules():
    if isinstance(m, torch.nn.LayerNorm):
        m.register_forward_hook(hook(name))
with torch.no_grad():
    for path, _ in pairs:
        import whisper
        audio = whisper.pad_or_trim(data.load_audio_torchaudio(path))
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.config.num_mel_bins).unsqueeze(0).to(args.device)
        model.generate(mel, max_new_tokens=128, num_beams=1, do_sample=False, language="en", task="transcribe")
rows = []
for name, chunks in var_by_ln.items():
    v = torch.cat(chunks)
    med, p5, mx = v.median().item(), v.kthvalue(max(1, int(0.05 * v.numel()))).values.item(), v.max().item()
    R = mx / max(med, 1e-12)
    levels = 32767 / R
    seed_err = 0.5 / max(levels, 1e-9) / 2  # half a level on var -> relative error of 1/sqrt is half of that
    rows.append((levels, name, med, p5, mx, R, seed_err, resid_max[name]))
rows.sort()
d = model.config.d_model
print(f"=== {args.model_id} d={d} n={len(pairs)}: {len(rows)} LayerNorms; worst {args.top} by int16 levels for the median token")
print(f"{'levels_med':>10s} {'R=max/med':>10s} {'var_med':>8s} {'var_p5':>8s} {'var_max':>9s} {'seed_err':>8s} {'|x|max':>7s}  layernorm")
for levels, name, med, p5, mx, R, se, xm in rows[:args.top]:
    print(f"{levels:10.1f} {R:10.0f} {med:8.3g} {p5:8.3g} {mx:9.3g} {100 * se:7.1f}% {xm:7.1f}  {name}")
print(f"LNs with < 16 levels: {sum(1 for r in rows if r[0] < 16)} / {len(rows)};  < 4 levels: {sum(1 for r in rows if r[0] < 4)}")
