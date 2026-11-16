"""TTS contract: text → audio, independent of the provider and of the messenger."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = [
    "MAX_AUDIO_BYTES",
    "SpeechSynthesizer",
    "SynthesizedAudio",
    "TTSError",
    "audio_extension",
]

MAX_AUDIO_BYTES = 10 * 1024 * 1024
_EXTENSIONS = {"audio/ogg": "ogg", "audio/mpeg": "mp3", "audio/wav": "wav"}


class TTSError(RuntimeError):
    """Safe-to-log failure: a short reason, never text, audio or credentials."""

    def __init__(self, message: str, *, retry: bool = True):
        super().__init__(message)
        self.retry = retry


@dataclass(frozen=True)
class SynthesizedAudio:
    data: bytes
    mime_type: str = "audio/ogg"
    seconds: float | None = None  # playing time, when the provider knows it

    def __post_init__(self) -> None:
        if not 0 < len(self.data) <= MAX_AUDIO_BYTES:
            raise ValueError("Synthesized audio must be nonempty and at most 10 MiB")
        if not self.mime_type.startswith("audio/"):
            raise ValueError("Synthesized audio needs an audio/* MIME type")

    @property
    def extension(self) -> str | None:
        return _EXTENSIONS.get(self.mime_type)

    @property
    def is_voice_message(self) -> bool:
        """OGG/Opus is the one format messengers play as a voice bubble."""
        return self.mime_type == "audio/ogg"


@runtime_checkable
class SpeechSynthesizer(Protocol):
    """Anything that turns text into audio. ``voice`` overrides the configured voice."""

    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio: ...

    async def aclose(self) -> None: ...


def audio_extension(audio: SynthesizedAudio) -> str:
    """File extension for sending; ``ValueError`` for types messengers cannot take."""
    if audio.extension is None:
        raise ValueError(f"Unsupported TTS audio type: {audio.mime_type}")
    return audio.extension
