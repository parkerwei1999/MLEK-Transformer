"""LibriSpeech dataset loading utilities (library SSoT).

Extracted from apo_inject/whisper_utils.py so data-loading consumers
(experiments/probe_act_distribution.py, experiments/apo_build.py) can
depend on the dataset layer without pulling whisper/jiwer metric deps.
whisper_utils re-exports these names for backward compatibility.
"""
import os

import numpy as np
import soundfile as sf


def load_audio_torchaudio(path, target_sr=16000):
    """Load audio via soundfile (FLAC/WAV); fall back to whisper.load_audio
    if soundfile can't decode. LibriSpeech is 16kHz FLAC mono — no resample
    needed in practice, but we still verify SR and resample defensively.
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr != target_sr:
        # Use numpy linear resampling (good enough; LibriSpeech is already 16k).
        import scipy.signal  # lazy import; scipy is in numpy stack
        n_out = int(round(wav.shape[0] * target_sr / sr))
        wav = scipy.signal.resample(wav, n_out).astype(np.float32)
    return wav.astype(np.float32)


def gather_librispeech_files(root, split="test-clean", n_max=4000):
    pairs = []
    split_root = os.path.join(root, split)
    for speaker in sorted(os.listdir(split_root)):
        speaker_dir = os.path.join(split_root, speaker)
        if not os.path.isdir(speaker_dir):
            continue
        for chapter in sorted(os.listdir(speaker_dir)):
            chapter_dir = os.path.join(speaker_dir, chapter)
            if not os.path.isdir(chapter_dir):
                continue
            trans_file = os.path.join(chapter_dir, f"{speaker}-{chapter}.trans.txt")
            if not os.path.exists(trans_file):
                continue
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utt_id, transcript = parts
                    audio = os.path.join(chapter_dir, f"{utt_id}.flac")
                    if os.path.exists(audio):
                        pairs.append((audio, transcript))
                        if len(pairs) >= n_max:
                            return pairs
    return pairs
