"""TTS module: text → speech behind one interface, provider chosen by configuration."""

from .base import MAX_AUDIO_BYTES, SpeechSynthesizer, SynthesizedAudio, TTSError, audio_extension
from .cloud import GoogleTTS, YandexTTS
from .delivery import send_audio, speak
from .factory import available_tts, load_tts, register_tts
from .kokoro import KokoroError, KokoroTTS
from .service import CachedSynthesizer

__all__ = [
    "MAX_AUDIO_BYTES",
    "CachedSynthesizer",
    "GoogleTTS",
    "KokoroError",
    "KokoroTTS",
    "SpeechSynthesizer",
    "SynthesizedAudio",
    "TTSError",
    "YandexTTS",
    "audio_extension",
    "available_tts",
    "load_tts",
    "register_tts",
    "send_audio",
    "speak",
]
