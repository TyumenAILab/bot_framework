"""Provider-independent layer around any ``SpeechSynthesizer``.

``CachedSynthesizer`` adds what every provider needs and none should reimplement:

* one overall timeout (including the provider's own auth/retries);
* identical concurrent requests are merged into one paid call;
* successful audio is cached **in RAM only**, with a TTL and size/entry caps, so a
  replay of the same text is neither slow nor billed twice. Failures are never cached;
* a usage event per real provider call (``kind=speech``, ``operation=tts``): status, latency,
  character and byte counts. No text, audio or keys. Money cost is **not** estimated
  (``cost_status=UNKNOWN_USAGE``): the token price book does not fit speech billing.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import time
import weakref
from collections import OrderedDict

from botkit.usage import UsageEvent, UsageTracker, current_context, safe_log

from .base import MAX_AUDIO_BYTES, SpeechSynthesizer, SynthesizedAudio


class CachedSynthesizer:
    def __init__(
        self,
        inner: SpeechSynthesizer,
        *,
        timeout: float = 6.0,
        cache_ttl: float = 300.0,
        cache_max_entries: int = 64,
        cache_max_bytes: int = 20 * 1024 * 1024,
        tracker: UsageTracker | None = None,
        provider: str | None = None,
    ):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("TTS timeout must be finite and positive")
        if not math.isfinite(cache_ttl) or cache_ttl < 0 or cache_max_entries < 0 or cache_max_bytes < 0:
            raise ValueError("TTS cache limits must be nonnegative")
        self.inner, self.timeout, self.tracker = inner, timeout, tracker
        self.provider = provider or type(inner).__name__
        self.cache_ttl, self.cache_max_entries, self.cache_max_bytes = cache_ttl, cache_max_entries, cache_max_bytes
        self.cache_bytes = 0
        self._cache: OrderedDict[str, tuple[float, SynthesizedAudio]] = OrderedDict()
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        self._expiry_handle: asyncio.TimerHandle | None = None

    @property
    def cache(self) -> OrderedDict[str, tuple[float, SynthesizedAudio]]:
        return self._cache

    def _drop(self, key: str) -> None:
        _, audio = self._cache.pop(key)
        self.cache_bytes -= len(audio.data)

    def _expire(self) -> None:
        now = time.monotonic()
        for key, (until, _) in list(self._cache.items()):
            if until <= now:
                self._drop(key)

    def _expire_on_timer(self) -> None:
        self._expiry_handle = None
        self._expire()
        if self._cache:
            delay = max(0.0, min(until for until, _ in self._cache.values()) - time.monotonic())
            self._expiry_handle = asyncio.get_running_loop().call_later(delay, self._expire_on_timer)

    def _store(self, key: str, audio: SynthesizedAudio) -> None:
        if self.cache_ttl <= 0 or len(audio.data) > self.cache_max_bytes or self.cache_max_entries == 0:
            return
        while self._cache and (
            len(self._cache) >= self.cache_max_entries or self.cache_bytes + len(audio.data) > self.cache_max_bytes
        ):
            self._drop(next(iter(self._cache)))
        self._cache[key] = (time.monotonic() + self.cache_ttl, audio)
        self.cache_bytes += len(audio.data)
        if self._expiry_handle is None:
            self._expiry_handle = asyncio.get_running_loop().call_later(self.cache_ttl, self._expire_on_timer)

    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        self._expire()
        key = hashlib.sha256(f"{voice or ''}\x00{text}".encode()).hexdigest()
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            self._expire()
            if key in self._cache:
                return self._cache[key][1]
            started = time.monotonic()
            status, error, audio = "success", None, None
            try:
                async with asyncio.timeout(self.timeout):
                    audio = await self.inner.synthesize(text, voice=voice)
                if not 0 < len(audio.data) <= MAX_AUDIO_BYTES or not audio.mime_type.startswith("audio/"):
                    raise ValueError("Invalid TTS audio")
                self._store(key, audio)
                return audio
            except BaseException as exc:
                status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
                error = type(exc).__name__
                raise
            finally:
                context = current_context()
                await safe_log(
                    self.tracker,
                    UsageEvent(
                        kind="speech",
                        operation="tts",
                        skill_context=context.get("skill_context") or "TTS",
                        provider=self.provider,
                        model=voice or getattr(self.inner, "voice", ""),
                        bot_id=context.get("bot_id", ""),
                        platform=context.get("platform", ""),
                        user_id=context.get("user_id", ""),
                        chat_id=context.get("chat_id", ""),
                        request_id=context.get("request_id", ""),
                        status=status,
                        error_type=error,
                        cost_status="UNKNOWN_USAGE",
                        latency_ms=(time.monotonic() - started) * 1000,
                        meta={
                            "characters": len(text),
                            "utf8_bytes": len(text.encode()),
                            "audio_bytes": len(audio.data) if audio else 0,
                            "audio_seconds": audio.seconds if audio else None,
                            "billing": "not_estimated",
                        },
                    ),
                )

    async def aclose(self) -> None:
        if self._expiry_handle:
            self._expiry_handle.cancel()
            self._expiry_handle = None
        self._cache.clear()
        self.cache_bytes = 0
        await self.inner.aclose()
