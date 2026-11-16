"""Speak a text to a chat: synthesize with any provider, send through any adapter.

Messenger-specific details stay in the adapter (``send_attachment(kind="voice")``); this
module only picks voice-bubble vs. plain audio file and never lets a TTS failure break the
bot: text is the fallback and the caller decides how to show it.
"""

from __future__ import annotations

import logging

from botkit.transport.base import MessengerAdapter

from .base import SpeechSynthesizer, SynthesizedAudio, audio_extension

logger = logging.getLogger(__name__)


async def send_audio(
    adapter: MessengerAdapter,
    chat_id: str,
    audio: SynthesizedAudio,
    *,
    caption: str = "",
    basename: str = "voice",
    **options: object,
) -> str:
    """OGG/Opus goes as a voice message, MP3/WAV as an audio file. Returns the message ID."""
    kind = "voice" if audio.is_voice_message else "audio"
    return await adapter.send_attachment(  # type: ignore[attr-defined]
        chat_id,
        audio.data,
        filename=f"{basename}.{audio_extension(audio)}",
        kind=kind,
        caption=caption,
        **options,
    )


async def speak(
    adapter: MessengerAdapter,
    chat_id: str,
    text: str,
    synthesizer: SpeechSynthesizer,
    *,
    caption: str = "",
    voice: str | None = None,
    basename: str = "voice",
    typing: bool = True,
) -> SynthesizedAudio | None:
    """Synthesize ``text`` and send it as a voice message.

    Returns the sent audio (``.seconds`` is the playing time when the provider knows it),
    or ``None`` if audio could not be produced or delivered. The reason is logged without
    content; the caller should then send the text instead.
    """
    try:
        if typing:
            async with await adapter.start_typing(chat_id):
                audio = await synthesizer.synthesize(text, voice=voice)
        else:
            audio = await synthesizer.synthesize(text, voice=voice)
        await send_audio(adapter, chat_id, audio, caption=caption, basename=basename)
        return audio
    except Exception as exc:
        # Platform errors (e.g. VK error 100) name the rejected parameter but carry no user content.
        detail = getattr(exc, "detail", None) or getattr(exc, "error_msg", None)
        if detail is None and isinstance(exc, RuntimeError):
            detail = str(exc)  # TTSError messages are content-free by contract
        logger.warning("TTS failed (%s%s)", type(exc).__name__, f": {str(detail)[:160]}" if detail else "")
        return None
