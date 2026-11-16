"""ASR contract: speech → text, independent of the provider and of the messenger.

The interface is the one the framework already had — ``AudioTranscriptionProvider``
(``atranscribe(...) -> LLMResponse``). Keeping a single contract means any provider
registered here also works inside the A5 ``AudioExtractor`` without adapters:

* ``result.response`` — final text (after optional LLM post-processing);
* ``result.raw``      — text exactly as the ASR engine returned it;
* ``result.meta``     — provider details, among them ``audio_duration_seconds`` if known.
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING

from botkit.llm.base import AudioTranscriptionProvider, LLMResponse
from botkit.llm.transcript import plain_transcript

if TYPE_CHECKING:
    from botkit.transport.base import Attachment

__all__ = [
    "AUDIO_EXTENSIONS",
    "ASRError",
    "AudioTranscriptionProvider",
    "audio_filename",
    "transcribe_attachment",
]

# MIME → file extension the engines understand. Messenger voice notes are OGG/Opus.
AUDIO_EXTENSIONS = {
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/webm": "webm",
    "audio/flac": "flac",
    "audio/aac": "aac",
}


class ASRError(Exception):
    """Safe-to-log ASR failure: a short reason, never audio, transcripts or credentials."""


def audio_filename(mime_type: str, stem: str = "voice") -> str:
    """``voice.ogg`` for ``audio/ogg`` etc. Unknown audio types fall back to ``.ogg``."""
    base = mime_type.split(";", 1)[0].strip().lower()
    return f"{posixpath.basename(stem)}.{AUDIO_EXTENSIONS.get(base, 'ogg')}"


async def transcribe_attachment(
    recognizer: AudioTranscriptionProvider,
    attachment: Attachment,
    *,
    timeout: float | None = None,
    language: str | None = None,
    strip_speaker_prefixes: bool = True,
) -> LLMResponse:
    """Download a messenger attachment and run it through any recognizer.

    ``strip_speaker_prefixes`` removes the ``[00:00:00.000 - 00:00:01.000] Speaker 1:``
    prefixes some engines (Pneuma) add, so the bot sees plain text. The raw engine text
    stays in ``result.raw``. The size limit is the attachment's own ``max_bytes``.
    """
    if attachment.kind not in {"audio", "voice"} and not attachment.mime_type.startswith("audio/"):
        raise ASRError("The attachment is not audio")
    data = await attachment.read_bytes()
    result = await recognizer.atranscribe(
        data,
        mime_type=attachment.mime_type,
        filename=audio_filename(attachment.mime_type),
        timeout=timeout,
        language=language,
    )
    if strip_speaker_prefixes:
        from dataclasses import replace

        result = replace(result, response=plain_transcript(result.response))
    return result
