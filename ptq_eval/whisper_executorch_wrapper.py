"""Whisper wrappers for ExecuTorch `examples.arm.aot_arm_compiler`.

The compiler imports this file and reads the module-level `ModelUnderTest`
and `ModelInputs`, so the variant is selected through environment variables:

  WHISPER_MODEL    Hugging Face model id          (openai/whisper-tiny)
  WHISPER_PART     encoder | decoder              (encoder)
  WHISPER_DEC_LEN  static decoder sequence length (8)

Inputs are random tensors: good enough to study export / partitioning, not
meaningful as int8 calibration data for accuracy.
"""

import os
from enum import Enum

import torch
from transformers import WhisperForConditionalGeneration


class WhisperPart(Enum):
    ENCODER = "encoder"
    DECODER = "decoder"


class WhisperEncoder(torch.nn.Module):
    """log-mel [1, n_mels, 3000] -> encoder states [1, 1500, d_model]."""

    def __init__(self, whisper: WhisperForConditionalGeneration):
        super().__init__()
        self.encoder = whisper.model.encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_features, return_dict=False)[0]


class WhisperDecoder(torch.nn.Module):
    """Decoder + LM head at a static length, without KV cache."""

    def __init__(self, whisper: WhisperForConditionalGeneration, dec_len: int):
        super().__init__()
        self.decoder = whisper.model.decoder
        self.proj_out = whisper.proj_out
        # A ready-made 4D mask makes transformers skip its vmap-based mask
        # builder, which leaves a non-serializable node in the exported graph.
        # The Arm ReplaceInfValues pass rewrites -inf to a quantizable -255.
        causal = torch.full((dec_len, dec_len), float("-inf")).triu(diagonal=1)
        self.register_buffer("causal_mask", causal[None, None], persistent=False)

    def forward(
        self, decoder_input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=self.causal_mask,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=False,
        )[0]
        return self.proj_out(hidden)


def _build(model_id: str, part: WhisperPart, dec_len: int):
    # Eager attention keeps matmul/softmax explicit instead of fused SDPA.
    whisper = WhisperForConditionalGeneration.from_pretrained(
        model_id, attn_implementation="eager", torch_dtype=torch.float32
    ).eval()
    cfg = whisper.config
    torch.manual_seed(0)
    if part is WhisperPart.ENCODER:
        frames = cfg.max_source_positions * 2
        return WhisperEncoder(whisper), (torch.randn(1, cfg.num_mel_bins, frames),)
    ids = torch.randint(0, cfg.vocab_size, (1, dec_len), dtype=torch.long)
    enc = torch.randn(1, cfg.max_source_positions, cfg.d_model)
    return WhisperDecoder(whisper, dec_len), (ids, enc)


ModelUnderTest, ModelInputs = _build(
    os.environ.get("WHISPER_MODEL", "openai/whisper-tiny"),
    WhisperPart(os.environ.get("WHISPER_PART", WhisperPart.ENCODER.value)),
    int(os.environ.get("WHISPER_DEC_LEN", "8")),
)
