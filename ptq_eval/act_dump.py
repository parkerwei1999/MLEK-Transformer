"""Dump fp32 activations (token x channel) of the tensors the quantization story is about, for
plotting per-channel ranges, token x channel magnitude maps and quantization-error maps:

  * every LayerNorm input (forward pre-hook)            -> "<module>.in"
  * every fc2 / mlp.fc2 output (the stream writer)       -> "<module>.out"
  * every block / encoder-decoder layer output (stream)  -> "<module>.out"

Whisper: encoder on log-mel of N dev-clean utterances; decoder captured on the last step of a
real fp32 greedy decode (static 128 positions). Vision: N ImageNet train images.
Output: <out-dir>/<tag>.npz (float16, key = tensor name, value = [N, tokens, channels]) and
<tag>.json (shapes, sample ids, decoded lengths). Filter tensors with --targets (regex on the
module name) to keep large models small.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import whisper_ptq_eval as pe  # noqa: E402


def attach(model, targets_re, store):
    """Register hooks; returns the list of handles."""
    handles = []
    for name, m in model.named_modules():
        if not re.search(targets_re, name):
            continue
        if isinstance(m, torch.nn.LayerNorm):
            handles.append(m.register_forward_pre_hook(
                lambda mod, inp, n=name: store.__setitem__(n + ".in", inp[0].detach())))
        if name.endswith("fc2") or name.endswith("mlp.fc2"):
            handles.append(m.register_forward_hook(
                lambda mod, inp, out, n=name: store.__setitem__(n + ".out", out.detach())))
        if re.search(r"(^|\.)(blocks|layers)\.\d+$", name):  # DeiT blocks.N, Swin layers.L.blocks.N, Whisper layers.N
            handles.append(m.register_forward_hook(
                lambda mod, inp, out, n=name: store.__setitem__(n + ".out", (out[0] if isinstance(out, tuple) else out).detach())))
    return handles


def to_np(t):
    t = t.detach().float().cpu()
    if t.dim() == 4:  # Swin [B, H, W, C] -> [B, H*W, C]
        t = t.reshape(t.shape[0], -1, t.shape[-1])
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t.half().numpy()


def dump_whisper(args, out):
    data = pe.load_module_from_path("librispeech_data", HERE / "data/librispeech_data.py")
    wrapper = pe.load_module_from_path("whisper_executorch_wrapper", HERE / "whisper_executorch_wrapper.py")
    from transformers import AutoConfig
    pe.N_MELS = AutoConfig.from_pretrained(args.model_id).num_mel_bins
    encoder, _ = wrapper._build(args.model_id, wrapper.WhisperPart.ENCODER, 128)
    decoder, _ = wrapper._build(args.model_id, wrapper.WhisperPart.DECODER, 128)
    encoder.to(args.device).eval(); decoder.to(args.device).eval()
    greedy = pe.GreedyDecoder(args.model_id, 128, args.device)
    pairs = data.gather_librispeech_files(args.librispeech_dir, "dev-clean", args.n)
    acc, meta = {}, {"samples": [], "decoded_len": []}
    with torch.no_grad():
        for path, _ in pairs:
            store = {}
            h = attach(encoder, args.targets, store)
            states = encoder(pe.log_mel(data.load_audio_torchaudio(path), args.device))
            for x in h: x.remove()
            enc_store = {"enc." + k: v for k, v in store.items()}
            store = {}
            h = attach(decoder, args.targets, store)
            tokens, _ = greedy.decode(decoder, states)  # hooks keep the LAST step: full sequence
            for x in h: x.remove()
            dec_store = {"dec." + k: v for k, v in store.items()}
            for k, v in {**enc_store, **dec_store}.items():
                acc.setdefault(k, []).append(to_np(v))
            meta["samples"].append(str(path))
            meta["decoded_len"].append(int(len(tokens)) if hasattr(tokens, "__len__") else -1)
    return acc, meta


def dump_vision(args, out):
    import fqvit_models
    from fqvit_models import imagenet_data as data
    model = fqvit_models.build_model(args.model_name).to(args.device).eval()
    batches = data._load_imgs(args.n, None, "train", shuffle=True, seed=0, data_dir=args.imagenet_dir,
                              model_name=args.model_name, batch_size=1)
    acc, meta = {}, {"samples": [f"imagenet-train seed0 #{i}" for i in range(args.n)]}
    with torch.no_grad():
        for b in batches:
            x = (b[0] if isinstance(b, (tuple, list)) else b).to(args.device)
            for i in range(x.shape[0]):
                store = {}
                h = attach(model, args.targets, store)
                model(x[i:i + 1])
                for hh in h: hh.remove()
                for k, v in store.items():
                    acc.setdefault(k, []).append(to_np(v))
    return acc, meta


def main(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.model_id:
        acc, meta = dump_whisper(args, out); tag = args.model_id.split("/")[-1]
    else:
        acc, meta = dump_vision(args, out); tag = args.model_name
    arrays = {k: np.concatenate(v, axis=0) for k, v in acc.items()}
    np.savez_compressed(out / f"{tag}.npz", **arrays)
    meta["tensors"] = {k: list(v.shape) for k, v in arrays.items()}
    (out / f"{tag}.json").write_text(json.dumps(meta, indent=1))
    mb = sum(v.nbytes for v in arrays.values()) / 1e6
    print(f"{tag}: {len(arrays)} tensors, {mb:.0f} MB fp16 -> {out / (tag + '.npz')}", flush=True)
    for k in list(arrays)[:6]:
        print(f"  {k:60s} {arrays[k].shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-id", default=None, help="openai/whisper-<size> (Whisper mode)")
    ap.add_argument("--model-name", default=None, help="FQ-ViT vision model name (vision mode)")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--targets", default=".", help="regex on module names to keep (e.g. 'layers\\.(2|3)\\.' or 'layers\\.2[0-3]\\.')")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--librispeech-dir", default="/home/shared/LibriSpeech")
    ap.add_argument("--imagenet-dir", default="/home/shared/ImageNet")
    main(ap.parse_args())
