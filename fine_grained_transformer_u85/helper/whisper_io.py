"""Whisper input features and static-length greedy decoding, shared by the Whisper evaluation, the
lowering probe and the diagnostics.
Set N_MELS (module attribute) from the model config before calling log_mel: large-v3 uses 128 mel bins.
"""
import torch
import whisper
from transformers import AutoConfig, GenerationConfig
from whisper.tokenizer import get_tokenizer

N_MELS = 80  # callers set it from the model config: whisper-large-v3 uses 128 mel bins


def log_mel(audio, device: str) -> torch.Tensor:
    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio), n_mels=N_MELS)
    return mel.unsqueeze(0).to(device)


class GreedyDecoder:
    """Static-length greedy decoding shared by the FP32 and int8 decoders."""

    def __init__(self, model_id: str, dec_len: int, device: str):
        # large-v3 adds the <|yue|> language token (vocab 51866): with 100 languages the
        # <|transcribe|> / <|notimestamps|> ids shift by one, so the prompt must be built
        # for the model's own language count or the decoder is asked to translate instead.
        vocab_size = AutoConfig.from_pretrained(model_id).vocab_size
        self.tokenizer = get_tokenizer(multilingual=True, num_languages=100 if vocab_size == 51866 else 99,
                                       language="en", task="transcribe")
        self.prompt = list(self.tokenizer.sot_sequence_including_notimestamps)
        self.eot = self.tokenizer.eot
        self.dec_len = dec_len
        self.device = device
        gen_cfg = GenerationConfig.from_pretrained(model_id)
        self.suppress = torch.tensor(gen_cfg.suppress_tokens, device=device)
        self.begin_suppress = torch.tensor(gen_cfg.begin_suppress_tokens, device=device)

    def padded_ids(self, tokens) -> torch.Tensor:
        ids = torch.full((1, self.dec_len), self.eot, dtype=torch.long, device=self.device)
        ids[0, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=self.device)
        return ids

    @torch.no_grad()
    def decode(self, decoder, enc_states: torch.Tensor):
        """Returns (generated token ids, whether the static length was hit)."""
        tokens = list(self.prompt)
        while len(tokens) < self.dec_len:
            logits = decoder(self.padded_ids(tokens), enc_states)[0, len(tokens) - 1].float()
            logits[self.suppress] = float("-inf")
            if len(tokens) == len(self.prompt):
                logits[self.begin_suppress] = float("-inf")
            token = int(logits.argmax())
            if token == self.eot:
                return tokens[len(self.prompt):], False
            tokens.append(token)
        return tokens[len(self.prompt):], True

    def text(self, tokens) -> str:
        return self.tokenizer.decode(tokens)
