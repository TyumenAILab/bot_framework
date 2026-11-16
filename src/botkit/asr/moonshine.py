"""Client for a self-hosted Moonshine-style ASR HTTP server (speech → text).

Contract: ``GET /health`` and ``POST /v1/audio/transcriptions`` with multipart field
``file`` → JSON ``{"text", "duration", "segments"}``. Any server that speaks this
contract works. It is meant for a private network, so proxy variables from the
environment are ignored. Audio lives only in memory and is never logged.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from botkit.llm.base import LLMResponse
from botkit.usage import UsageEvent, UsageTracker, current_context, safe_log

from .base import ASRError

logger = logging.getLogger(__name__)


class MoonshineError(ASRError):
    """Safe-to-log failure: carries a short reason, never audio, credentials or transcripts."""


class MoonshineTranscriber:
    """``AudioTranscriptionProvider`` for the Moonshine HTTP API.

    The server has a fixed number of workers, so at most ``max_concurrent`` recordings
    are sent at once and the rest wait here instead of timing out there.
    ``on_queued(ahead)`` (optional, Moonshine-specific) is awaited once if a recording
    has to wait, so a bot can tell the user "you are Nth in the queue".
    """

    provider = "moonshine"

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        language: str = "en",
        model_name: str = "moonshine",
        timeout: float = 180,
        connect_timeout: float = 10,
        max_concurrent: int = 2,
        max_bytes: int = 20 * 1024 * 1024,
        client: httpx.AsyncClient | None = None,
        tracker: UsageTracker | None = None,
    ):
        parts = urlsplit(base_url or "")
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            raise ValueError("MOONSHINE_API_URL must be an http(s) URL without credentials, e.g. http://host:8181")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Moonshine timeout must be finite and positive")
        if type(max_concurrent) is not int or max_concurrent < 1:
            raise ValueError("MOONSHINE_MAX_CONCURRENT must be a positive integer")
        self.base_url = base_url.rstrip("/")
        self.language, self.model_name, self.timeout = language, model_name, timeout
        self.max_bytes, self.tracker = max_bytes, tracker
        self.max_concurrent = max_concurrent
        self._slots = asyncio.Semaphore(max_concurrent)
        self.active = self.waiting = 0
        self._headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            trust_env=False, follow_redirects=False, timeout=httpx.Timeout(timeout, connect=connect_timeout)
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> MoonshineTranscriber:
        return cls(
            os.getenv("MOONSHINE_API_URL", "").strip(),
            api_key=os.getenv("MOONSHINE_API_KEY", ""),
            language=os.getenv("MOONSHINE_LANGUAGE", "en"),
            timeout=float(os.getenv("MOONSHINE_TIMEOUT", "180")),
            max_concurrent=int(os.getenv("MOONSHINE_MAX_CONCURRENT", "2")),
            **kwargs,
        )

    async def __aenter__(self) -> MoonshineTranscriber:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def health(self) -> Any:
        try:
            response = await self.client.get(f"{self.base_url}/health", headers=self._headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise MoonshineError(f"health check failed ({type(exc).__name__})") from None

    async def _acquire(self, on_queued: Callable[[int], Awaitable[None]] | None) -> None:
        if self._slots.locked():
            ahead = self.active + self.waiting
            self.waiting += 1
            try:
                if on_queued is not None:
                    try:
                        await on_queued(ahead)
                    except Exception:
                        logger.warning("Could not tell the user about the ASR queue")
                await self._slots.acquire()
            finally:
                self.waiting -= 1
        else:
            await self._slots.acquire()
        self.active += 1

    async def atranscribe(
        self,
        data: bytes,
        *,
        mime_type: str,
        filename: str,
        timeout: float | None = None,
        language: str | None = None,
        on_queued: Callable[[int], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        if not data or len(data) > self.max_bytes:
            raise ValueError("Audio must be nonempty and within the configured size limit")
        if language is not None and language != self.language:
            raise ValueError(f"This Moonshine server transcribes {self.language!r} only")
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Transcription timeout must be finite and positive")
        deadline = min(timeout or self.timeout, self.timeout)
        context = current_context()
        event = UsageEvent(
            provider=self.provider,
            model=self.model_name,
            operation="transcription",
            request_id=context.get("request_id") or uuid4().hex,
            cost_status="UNKNOWN_USAGE",
            meta={"byte_count": len(data), "mime_type": mime_type},
        )
        async with asyncio.timeout(deadline):
            await self._acquire(on_queued)
            # Latency is measured from the start of recognition, not from the queue wait.
            started = time.perf_counter()
            try:
                try:
                    response = await self.client.post(
                        f"{self.base_url}/v1/audio/transcriptions",
                        files={"file": (filename, data, mime_type)},
                        headers=self._headers,
                    )
                except httpx.HTTPError as exc:
                    event.http_status = getattr(getattr(exc, "response", None), "status_code", None)
                    raise MoonshineError(f"request failed ({type(exc).__name__})") from None
                event.http_status = response.status_code
                if not response.is_success:
                    raise MoonshineError(f"HTTP {response.status_code}")
                try:
                    payload = response.json()
                    text = payload["text"]
                except (ValueError, KeyError, TypeError):
                    raise MoonshineError("unexpected response format") from None
                if not isinstance(text, str):
                    raise MoonshineError("unexpected response format")
                text = text.strip()
                duration = payload.get("duration")
                seconds = float(duration) if isinstance(duration, (int, float)) and math.isfinite(duration) else None
                segments = payload.get("segments")
                event.meta.update(
                    {
                        "text_chars": len(text),
                        "audio_duration_seconds": seconds,
                    }
                )
                return LLMResponse(
                    text,
                    text,
                    "raw_text_fallback",
                    model=self.model_name,
                    provider=self.provider,
                    request_id=event.request_id,
                    meta={
                        "audio_duration_seconds": seconds,
                        "segment_count": len(segments) if isinstance(segments, list) else 0,
                    },
                )
            except BaseException as exc:
                event.status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
                event.error_type = type(exc).__name__
                raise
            finally:
                self.active -= 1
                self._slots.release()
                event.latency_ms = (time.perf_counter() - started) * 1000
                # Transcript and audio are never written, even with LOG_TEXT=true.
                await safe_log(self.tracker, event)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()
