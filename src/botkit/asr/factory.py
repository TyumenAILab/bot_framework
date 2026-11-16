"""ASR provider registry. The provider is chosen by configuration, not by code.

``ASR_PROVIDER`` is one of:

* ``disabled`` / ``none`` / empty — ASR off, ``load_asr()`` returns ``None``;
* ``pneuma``      — Pneuma queued ASR (legacy or chunked protocol);
* ``moonshine``   — self-hosted Moonshine-compatible HTTP server;
* ``module:factory`` — your own provider: ``factory(**options)`` returns an object with
  ``atranscribe(data, *, mime_type, filename, timeout=None, language=None) -> LLMResponse``
  and ``aclose()``. Operator-controlled only: never read it from user input.

New engines can also be registered in code with :func:`register_asr`.
``ASR_POSTPROCESS_ENABLED=true`` wraps any built-in provider in the LLM post-processing
pipeline (punctuation/case restoration); custom factories are returned unchanged.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from typing import Any

from botkit.llm.base import AudioTranscriptionProvider
from botkit.llm.config import _env_bool
from botkit.usage import UsageTracker

ASRFactory = Callable[..., AudioTranscriptionProvider]
_REGISTRY: dict[str, ASRFactory] = {}
_DISABLED = {"", "none", "disabled"}


def register_asr(name: str, factory: ASRFactory, *, replace: bool = False) -> None:
    """Make ``ASR_PROVIDER=<name>`` build ``factory(tracker=..., **options)``."""
    key = name.strip().lower()
    if not key or ":" in key or key in _DISABLED:
        raise ValueError("ASR provider name must be a nonempty word without ':'")
    if key in _REGISTRY and not replace:
        raise ValueError(f"ASR provider already registered: {key}")
    _REGISTRY[key] = factory


def available_asr() -> list[str]:
    return sorted(_REGISTRY)


def _build_pneuma(*, tracker: UsageTracker | None = None, store: Any = None, **_: Any):
    from botkit.llm.pneuma import PneumaConfig, PneumaTranscriber
    from botkit.llm.pneuma_upload import ChunkedPneumaTranscriber, UploadConfig
    from botkit.llm.upload_state import SQLiteUploadStore

    protocol = os.getenv("PNEUMA_UPLOAD_PROTOCOL", "legacy").strip().lower()
    if protocol not in {"legacy", "chunked"}:
        raise ValueError("PNEUMA_UPLOAD_PROTOCOL must be legacy or chunked")
    if protocol == "legacy" and store is not None:
        raise ValueError("Resumable upload stores require PNEUMA_UPLOAD_PROTOCOL=chunked")
    config = UploadConfig.from_env() if protocol == "chunked" else PneumaConfig.from_env()
    if protocol == "chunked":
        if store is None and (path := os.getenv("PNEUMA_UPLOAD_STATE_DB", "").strip()):
            store = SQLiteUploadStore(path)
        return ChunkedPneumaTranscriber(config, tracker=tracker, store=store)
    return PneumaTranscriber(config, tracker=tracker)


def _build_moonshine(*, tracker: UsageTracker | None = None, **_: Any):
    from .moonshine import MoonshineTranscriber

    return MoonshineTranscriber.from_env(tracker=tracker)


_REGISTRY.update({"pneuma": _build_pneuma, "moonshine": _build_moonshine})


def load_asr(
    provider: str | None = None, *, tracker: UsageTracker | None = None, store: Any = None
) -> AudioTranscriptionProvider | None:
    """Build the configured recognizer, or ``None`` when ASR is disabled.

    ``provider`` overrides ``ASR_PROVIDER``. The caller owns the result and must
    ``await aclose()`` it (or use ``async with`` where the provider supports it).
    """
    name = (os.getenv("ASR_PROVIDER", "") if provider is None else provider).strip()
    if name.lower() in _DISABLED:
        return None
    if ":" in name:
        module, _, attribute = name.partition(":")
        if not module or not attribute:
            raise ValueError("ASR_PROVIDER must be module:factory")
        return getattr(importlib.import_module(module), attribute)(tracker=tracker)
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"ASR_PROVIDER must be disabled, module:factory or one of: {', '.join(available_asr())}")
    # Build the processor first: a bad post-processing setting must fail before any
    # recognizer (and its HTTP client) exists.
    processor = None
    if _env_bool("ASR_POSTPROCESS_ENABLED", False):
        from botkit.llm.transcript import load_transcript_processor

        processor = load_transcript_processor(tracker=tracker)
    transcriber = _REGISTRY[key](tracker=tracker, store=store)
    if processor is None:
        return transcriber
    from botkit.llm.transcript import TranscriptionPipeline

    return TranscriptionPipeline(transcriber, processor, owns_components=True)
