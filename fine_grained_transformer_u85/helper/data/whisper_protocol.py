"""Shared deterministic transcription protocol for Whisper measurements."""


WHISPER_DECODE_TEMPERATURE = 0.0


def transcribe_whisper_deterministic(model, audio):
    """Transcribe one waveform with the repository measurement contract."""
    return model.transcribe(
        audio,
        language="en",
        verbose=False,
        fp16=False,
        temperature=WHISPER_DECODE_TEMPERATURE,
    )
