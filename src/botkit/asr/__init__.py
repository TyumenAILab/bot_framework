"""ASR module: speech → text behind one interface, provider chosen by configuration."""

from .base import (
    AUDIO_EXTENSIONS,
    ASRError,
    AudioTranscriptionProvider,
    audio_filename,
    transcribe_attachment,
)
from .factory import available_asr, load_asr, register_asr
from .moonshine import MoonshineError, MoonshineTranscriber

__all__ = [
    "AUDIO_EXTENSIONS",
    "ASRError",
    "AudioTranscriptionProvider",
    "MoonshineError",
    "MoonshineTranscriber",
    "audio_filename",
    "available_asr",
    "load_asr",
    "register_asr",
    "transcribe_attachment",
]
