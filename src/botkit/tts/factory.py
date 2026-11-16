"""TTS provider registry. The provider is chosen by configuration, not by code.

``TTS_PROVIDER`` is one of:

* ``disabled`` / ``none`` / empty — TTS off, ``load_tts()`` returns ``None``;
* ``yandex`` | ``google`` | ``kokoro`` — built-in providers (see their ``from_env``);
* ``module:factory`` — your own: ``factory()`` returns an object with
  ``async synthesize(text, *, voice=None) -> SynthesizedAudio`` and ``async aclose()``.
  Operator-controlled only: never read it from user input.

More engines can be registered in code with :func:`register_tts`. ``TTS_TIMEOUT`` (default 6 s)
is the overall budget of one synthesis; ``TTS_CACHE_TTL`` (default 300 s, ``0`` = off) keeps
audio in RAM so a replayed text is not synthesized twice.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

from botkit.usage import UsageTracker

from .base import SpeechSynthesizer
from .service import CachedSynthesizer

TTSFactory = Callable[..., SpeechSynthesizer]
_REGISTRY: dict[str, TTSFactory] = {}
_DISABLED = {"", "none", "disabled"}


def register_tts(name: str, factory: TTSFactory, *, replace: bool = False) -> None:
    """Make ``TTS_PROVIDER=<name>`` build ``factory(timeout=...)``."""
    key = name.strip().lower()
    if not key or ":" in key or key in _DISABLED:
        raise ValueError("TTS provider name must be a nonempty word without ':'")
    if key in _REGISTRY and not replace:
        raise ValueError(f"TTS provider already registered: {key}")
    _REGISTRY[key] = factory


def available_tts() -> list[str]:
    return sorted(_REGISTRY)


def _yandex(**kwargs: Any) -> SpeechSynthesizer:
    from .cloud import YandexTTS

    return YandexTTS.from_env(**kwargs)


def _google(**kwargs: Any) -> SpeechSynthesizer:
    from .cloud import GoogleTTS

    return GoogleTTS.from_env(**kwargs)


def _kokoro(**kwargs: Any) -> SpeechSynthesizer:
    from .kokoro import KokoroTTS

    return KokoroTTS.from_env(timeout=kwargs["timeout"])


_REGISTRY.update({"yandex": _yandex, "google": _google, "kokoro": _kokoro})


def load_tts(
    provider: str | None = None, *, tracker: UsageTracker | None = None, cache: bool = True
) -> SpeechSynthesizer | None:
    """Build the configured synthesizer, or ``None`` when TTS is disabled.

    ``provider`` overrides ``TTS_PROVIDER``. By default the result is wrapped in
    :class:`CachedSynthesizer` (timeout, RAM cache, usage events). The caller owns it
    and must ``await aclose()``.
    """
    name = (os.getenv("TTS_PROVIDER", "") if provider is None else provider).strip()
    if name.lower() in _DISABLED:
        return None
    timeout = float(os.getenv("TTS_TIMEOUT", "6"))
    if ":" in name:
        module, _, attribute = name.partition(":")
        if not module or not attribute:
            raise ValueError("TTS_PROVIDER must be module:factory")
        inner = getattr(importlib.import_module(module), attribute)()
        label = name
    else:
        label = name.lower()
        if label not in _REGISTRY:
            raise ValueError(
                f"TTS_PROVIDER must be disabled, module:factory or one of: {', '.join(available_tts())}"
            )
        inner = _REGISTRY[label](timeout=timeout)
    if not cache:
        return inner
    return CachedSynthesizer(
        inner,
        timeout=timeout,
        cache_ttl=float(os.getenv("TTS_CACHE_TTL", "300")),
        tracker=tracker,
        provider=label,
    )
