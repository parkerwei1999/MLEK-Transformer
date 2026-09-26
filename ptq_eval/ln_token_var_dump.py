"""Dump the fp32 per-token variance at every LayerNorm input of a Whisper model.

Same inference as ln_var_profile.py (HF fp32 model, first N dev-clean utterances,
greedy generate with the KV cache), so the per-LN statistics reproduce the census.
var = mean over d of (x - mean)^2 of the LayerNorm input, before + eps.

Outputs (in --out-dir):
  whisper-<size>.csv          model,layernorm,var_fp32   (one row per token per LayerNorm)
  whisper-<size>_summary.csv  model,layernorm,n_tokens,var_max,var_median,var_p99,r_max_over_median,
                              int16_levels_median_token,argmax_utt_id,argmax_token_idx
Decoder token_idx is the position in the decoded sequence (0 = <|startoftranscript|>); the
first generate step feeds the whole prompt, later steps one token each. Encoder token_idx is
the frame index (0..1499, padding frames included).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import whisper_ptq_eval as pe  # noqa: E402

NAMES = {"tiny": "Whisper-Tiny", "base": "Whisper-Base", "small": "Whisper-Small",
         "medium": "Whisper-Medium", "large-v3": "Whisper-Large-v3"}

ap = argparse.ArgumentParser()
ap.add_argument("--model-id", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--device", default="cuda")
ap.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
ap.add_argument("--out-dir", required=True)
args = ap.parse_args()

size = args.model_id.split("whisper-")[-1]
label = NAMES.get(size, args.model_id)
data = pe.load_module_from_path("librispeech_data", HERE / "data/librispeech_data.py")
model = WhisperForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.float32).to(args.device).eval()
pairs = data.gather_librispeech_files(args.librispeech_dir, "dev-clean", args.n)

var_by_ln = defaultdict(list)      # name -> list of 1-D tensors (per-token variance)
pos_by_ln = defaultdict(list)      # name -> list of (utt_id, token_idx) in the same order
seen = defaultdict(int)            # name -> tokens already seen in the current utterance
cur_utt = [None]


def hook(name):
    def fn(mod, inp, out):
        x = inp[0].detach().float()
        v = x.var(-1, unbiased=False).reshape(-1).cpu()
        start = seen[name]
        var_by_ln[name].append(v)
        pos_by_ln[name].extend((cur_utt[0], start + i) for i in range(v.numel()))
        seen[name] = start + v.numel()
    return fn


for name, m in model.named_modules():
    if isinstance(m, torch.nn.LayerNorm):
        m.register_forward_hook(hook(name))

import whisper  # noqa: E402

with torch.no_grad():
    for path, _ in pairs:
        cur_utt[0] = Path(path).stem
        seen.clear()
        audio = whisper.pad_or_trim(data.load_audio_torchaudio(path))
        mel = whisper.log_mel_spectrogram(audio, n_mels=model.config.num_mel_bins).unsqueeze(0).to(args.device)
        model.generate(mel, max_new_tokens=128, num_beams=1, do_sample=False, language="en", task="transcribe")

out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
with open(out / f"whisper-{size}.csv", "w", newline="") as fh, open(out / f"whisper-{size}_summary.csv", "w", newline="") as fs:
    w = csv.writer(fh); w.writerow(["model", "layernorm", "var_fp32"])
    s = csv.writer(fs); s.writerow(["model", "layernorm", "n_tokens", "var_max", "var_median", "var_p99",
                                    "r_max_over_median", "int16_levels_median_token", "argmax_utt_id", "argmax_token_idx"])
    for name in var_by_ln:
        v = torch.cat(var_by_ln[name])
        for x in v.tolist():
            w.writerow([label, name, f"{x:.6g}"])
        med, mx = v.median().item(), v.max().item()
        p99 = v.kthvalue(max(1, int(0.99 * v.numel()))).values.item()
        utt, tok = pos_by_ln[name][int(v.argmax())]
        r = mx / max(med, 1e-12)
        s.writerow([label, name, v.numel(), f"{mx:.6g}", f"{med:.6g}", f"{p99:.6g}", f"{r:.6g}", f"{32767 / r:.4g}", utt, tok])
print(f"{label}: {len(var_by_ln)} LayerNorms, {sum(t.numel() for c in var_by_ln.values() for t in c)} rows -> {out}", flush=True)
