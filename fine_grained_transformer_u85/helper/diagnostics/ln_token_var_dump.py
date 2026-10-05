"""Dump the fp32 per-token variance at every LayerNorm input of a Whisper model.

Default: same inference as ln_var_profile.py (HF fp32 model, first N utterances of a LibriSpeech split,
greedy generate with the KV cache), so the per-LN statistics reproduce the census. Each decoder row is a
position that was actually decoded (the first generate step feeds the whole prompt, later steps one token).

--static: the graphs the accuracy runner and the device execute (helper/whisper_executorch_wrapper: encoder on
1500 mel frames, decoder on DEC_LEN = 128 EOT-padded positions, no KV cache), fp32 on the GPU. Per utterance the
decoder is recorded on the two passes whisper_eval.py calibrates with: the full transcript and one random prefix,
both padded to 128, so the rows include the padded positions the observers see (frame = token | padding).
var = mean over d of (x - mean)^2 of the LayerNorm input, before + eps.

Outputs (in --out-dir):
  whisper-<size>.csv          model,layernorm,var_fp32,split,utt_id,token_idx,frame,pass   (one row per token per LayerNorm;
                              frame = speech | padding for encoder tokens (beyond the utterance's last mel frame);
                              decoder: '' (default) or token | padding (--static); pass = full | prefix (--static decoder) else '')
  whisper-<size>_summary.csv  model,layernorm,n_tokens,var_max,var_median,var_p99,r_max_over_median,
                              int16_levels_median_token,argmax_utt_id,argmax_token_idx   (over every row, padding included)
  whisper-<size>_padding_summary.csv  (--static) per decoder LayerNorm: token rows vs padding rows
Decoder token_idx is the position in the decoded sequence (0 = <|startoftranscript|>). Encoder token_idx is
the frame index (0..1499, padding frames included).
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # fine_grained_transformer_u85/
sys.path.insert(0, str(ROOT))
import core.quant as cq  # noqa: E402

NAMES = {"tiny": "Whisper-Tiny", "base": "Whisper-Base", "small": "Whisper-Small",
         "medium": "Whisper-Medium", "large-v3": "Whisper-Large-v3"}
DEC_LEN = 128  # whisper_eval.py

ap = argparse.ArgumentParser()
ap.add_argument("--model-id", required=True)
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--split", default="dev-clean", help="LibriSpeech split: dev-clean (calibration) or test-clean (evaluation)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--static", action="store_true", default=False,
                help="record the exported static graphs (128-position EOT-padded decoder) on the two calibration passes "
                     "per utterance instead of HF generate; see the module docstring")
ap.add_argument("--seed", type=int, default=0, help="--static: the random-prefix pass (whisper_eval.py uses the same draw)")
args = ap.parse_args()

size = args.model_id.split("whisper-")[-1]
label = NAMES.get(size, args.model_id)
data = cq.load_module_from_path("librispeech_data", ROOT / "helper/data/librispeech_data.py")
pairs = data.gather_librispeech_files(args.librispeech_dir, args.split, args.n)

var_by_ln = defaultdict(list)      # name -> list of 1-D tensors (per-token variance)
pos_by_ln = defaultdict(list)      # name -> list of (utt_id, token_idx, n_real | None, pass) in the same order
seen = defaultdict(int)            # name -> tokens already seen in the current utterance (HF generate mode)
cur = {"utt": None, "pass": "", "len": None, "record": True}
speech_frames = {}                 # utt_id -> encoder frames covering speech (30 s window = 1500 frames)


def hook(name):
    enc = ".encoder." in name

    def fn(mod, inp, out):
        if not cur["record"]:
            return
        x = inp[0].detach().float()
        v = x.var(-1, unbiased=False).reshape(-1).cpu()
        var_by_ln[name].append(v)
        if args.static:
            n_real = speech_frames[cur["utt"]] if enc else cur["len"]
            pos_by_ln[name].extend((cur["utt"], i, n_real, "" if enc else cur["pass"]) for i in range(v.numel()))
        else:
            start = seen[name]
            n_real = speech_frames[cur["utt"]] if enc else None
            pos_by_ln[name].extend((cur["utt"], start + i, n_real, "") for i in range(v.numel()))
            seen[name] = start + v.numel()
    return fn


def register(module, prefix=""):
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.register_forward_hook(hook(prefix + name))


def load(path):
    raw = data.load_audio_torchaudio(path)
    cur["utt"] = Path(path).stem
    speech_frames[cur["utt"]] = min(1500, -(-len(raw) // (16000 * 20 // 1000)))  # 20 ms per encoder frame
    return raw


with torch.no_grad():
    if args.static:
        from transformers import AutoConfig
        from helper import whisper_io as wio
        wrapper = cq.load_module_from_path("whisper_executorch_wrapper", ROOT / "helper/whisper_executorch_wrapper.py")
        wio.N_MELS = AutoConfig.from_pretrained(args.model_id).num_mel_bins
        encoder, _ = wrapper._build(args.model_id, wrapper.WhisperPart.ENCODER, DEC_LEN)
        decoder, _ = wrapper._build(args.model_id, wrapper.WhisperPart.DECODER, DEC_LEN)
        encoder.to(args.device).eval(); decoder.to(args.device).eval()
        register(encoder, "model."); register(decoder, "model.")  # the HF module paths, as in the default mode
        greedy = wio.GreedyDecoder(args.model_id, DEC_LEN, args.device)
        rng = random.Random(args.seed)
        for path, _ in pairs:
            raw = load(path)
            cur["pass"], cur["record"] = "", True
            enc_states = encoder(wio.log_mel(raw, args.device))
            cur["record"] = False  # the greedy steps themselves are not recorded (one 128-position pass per token)
            tokens, _ = greedy.decode(decoder, enc_states)
            tokens = list(greedy.prompt) + tokens
            cur["record"] = True
            for pass_, n in (("full", len(tokens)), ("prefix", rng.randint(len(greedy.prompt), len(tokens)))):
                cur["pass"], cur["len"] = pass_, n
                decoder(greedy.padded_ids(tokens[:n]), enc_states)
    else:
        import whisper
        from transformers import WhisperForConditionalGeneration
        model = WhisperForConditionalGeneration.from_pretrained(args.model_id, dtype=torch.float32).to(args.device).eval()
        register(model)
        for path, _ in pairs:
            raw = load(path)
            seen.clear()
            audio = whisper.pad_or_trim(raw)
            mel = whisper.log_mel_spectrogram(audio, n_mels=model.config.num_mel_bins).unsqueeze(0).to(args.device)
            model.generate(mel, max_new_tokens=DEC_LEN, num_beams=1, do_sample=False, language="en", task="transcribe")


def frame_of(name, tok, n_real):
    if n_real is None:
        return ""
    return ("speech" if ".encoder." in name else "token") if tok < n_real else "padding"


out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
with open(out / f"whisper-{size}.csv", "w", newline="") as fh, open(out / f"whisper-{size}_summary.csv", "w", newline="") as fs:
    w = csv.writer(fh); w.writerow(["model", "layernorm", "var_fp32", "split", "utt_id", "token_idx", "frame", "pass"])
    s = csv.writer(fs); s.writerow(["model", "layernorm", "n_tokens", "var_max", "var_median", "var_p99",
                                    "r_max_over_median", "int16_levels_median_token", "argmax_utt_id", "argmax_token_idx"])
    pad_rows = []
    for name in var_by_ln:
        v = torch.cat(var_by_ln[name])
        frames = [frame_of(name, tok, n_real) for (_, tok, n_real, _) in pos_by_ln[name]]
        for x, (utt, tok, _, pass_), fr in zip(v.tolist(), pos_by_ln[name], frames):
            w.writerow([label, name, f"{x:.6g}", args.split, utt, tok, fr, pass_])
        med, mx = v.median().item(), v.max().item()
        p99 = v.kthvalue(max(1, int(0.99 * v.numel()))).values.item()
        utt, tok = pos_by_ln[name][int(v.argmax())][:2]
        r = mx / max(med, 1e-12)
        s.writerow([label, name, v.numel(), f"{mx:.6g}", f"{med:.6g}", f"{p99:.6g}", f"{r:.6g}", f"{32767 / r:.4g}", utt, tok])
        if args.static and ".decoder." in name:
            is_tok = torch.tensor([fr == "token" for fr in frames])
            t, p = v[is_tok], v[~is_tok]
            pad_rows.append([label, name, t.numel(), p.numel(), f"{t.median():.6g}", f"{p.median():.6g}" if p.numel() else "",
                             f"{t.max():.6g}", f"{p.max():.6g}" if p.numel() else "",
                             f"{(p < t.median()).float().mean() * 100:.1f}" if p.numel() else ""])
if args.static:
    with open(out / f"whisper-{size}_padding_summary.csv", "w", newline="") as fp:
        c = csv.writer(fp)
        c.writerow(["model", "layernorm", "n_token", "n_padding", "median_token", "median_padding", "max_token", "max_padding",
                    "padding_below_token_median_pct"])
        c.writerows(pad_rows)
print(f"{label}: {len(var_by_ln)} LayerNorms, {sum(t.numel() for c in var_by_ln.values() for t in c)} rows"
      f"{' (static graphs, padded decoder)' if args.static else ''} -> {out}", flush=True)
