"""Client for a self-hosted Kokoro TTS server (English text → OGG/Opus or WAV).

Contract (published at /openapi.json of the server):
  GET  /health            → {"status", "voices", "max_pending", "sample_rate", ...}
  POST /v1/audio/speech   JSON {"model": "kokoro", "input": <1..5000 chars>, "voice", "speed": 0.5..2.0,
                          "response_format": "wav" | "ogg"} with "Authorization: Bearer <key>"

API 1.1 produced WAV only (PCM16 mono, 24 kHz); API 1.2 can also return OGG, about ten times
smaller, which matters on a lossy network. We ask for OGG and fall back to WAV if the server
refuses. Whatever arrives is normalised to OGG/Opus at the configured rate (needs the ``audio``
extra); ``output="wav"`` returns the raw WAV. The server is on a private network, so proxy
settings from the environment are ignored. Text, audio and the API key never reach logs or errors.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import struct
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from .audio import OPUS_RATE, OPUS_RATES, audio_seconds, to_ogg_opus
from .base import MAX_AUDIO_BYTES, SynthesizedAudio, TTSError

logger = logging.getLogger(__name__)

MAX_TEXT = 5000


def wav_seconds(data: bytes) -> float | None:
    """Playing time of a canonical PCM WAV (44-byte header); None if the header cannot be read."""
    try:
        channels, rate = struct.unpack("<HI", data[22:28])
        bits = struct.unpack("<H", data[34:36])[0]
        size = struct.unpack("<I", data[40:44])[0]
    except struct.error:
        return None
    size = min(size, max(0, len(data) - 44))  # a streaming header may carry a bogus size
    bytes_per_second = rate * channels * bits // 8
    return round(size / bytes_per_second, 3) if bytes_per_second else None


class KokoroError(TTSError):
    """Safe-to-log failure reason (no text, audio, URL with credentials or key)."""


class KokoroTTS:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        voice: str = "af_heart",
        speed: float = 1.0,
        timeout: float = 15.0,
        max_concurrent: int = 4,
        output: str = "ogg",
        attempt_timeout: float = 40.0,
        attempts: int = 5,
        idle_timeout: float = 6.0,
        opus_rate: int = OPUS_RATE,
        request_format: str = "ogg",
        client: httpx.AsyncClient | None = None,
    ):
        parts = urlsplit(base_url or "")
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise ValueError("KOKORO_TTS_URL must be an http(s) URL such as http://host:8777")
        if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
            raise ValueError("KOKORO_TTS_SPEED must be between 0.5 and 2.0")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("TTS timeout must be positive")
        if output not in {"ogg", "wav"}:
            raise ValueError("KOKORO_TTS_OUTPUT must be ogg or wav")
        # ogg = OGG/Opus voice message (what VK and Telegram play as a voice bubble); wav = raw server output.
        if int(opus_rate) not in OPUS_RATES:
            raise ValueError(f"KOKORO_TTS_OPUS_RATE must be one of {OPUS_RATES}")
        if request_format not in {"ogg", "wav"}:
            raise ValueError("KOKORO_TTS_REQUEST_FORMAT must be ogg or wav")
        self.opus_rate, self.output = int(opus_rate), output
        if output == "wav":
            request_format = "wav"  # the raw WAV is only available when WAV is what the server sends
        self.request_format = request_format
        self.attempt_timeout, self.attempts = float(attempt_timeout), max(1, int(attempts))
        self.base_url, self.voice, self.speed = base_url.rstrip("/"), voice, speed
        self.timeout = timeout
        self._headers = {"Authorization": "Bearer " + api_key} if api_key else {}
        # The server queues at most `max_pending` requests (10 on the reference server); stay below that.
        self._slots = asyncio.Semaphore(max(1, int(max_concurrent)))
        self._owns_client = client is None
        # read = the longest silence tolerated between two chunks. A slow transfer that keeps delivering
        # data may run up to attempt_timeout; one that stops dead is abandoned after idle_timeout seconds.
        self.idle_timeout = float(idle_timeout)
        self.client = client or httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(timeout, connect=10, read=self.idle_timeout),
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> KokoroTTS:
        return cls(
            os.getenv("KOKORO_TTS_URL", "").strip(),
            api_key=os.getenv("KOKORO_TTS_API_KEY", ""),
            voice=os.getenv("KOKORO_TTS_VOICE", "af_heart"),
            speed=float(os.getenv("KOKORO_TTS_SPEED", "1")),
            max_concurrent=int(os.getenv("KOKORO_TTS_MAX_CONCURRENT", "4")),
            output=os.getenv("KOKORO_TTS_OUTPUT", "ogg").lower(),
            attempt_timeout=float(os.getenv("KOKORO_TTS_ATTEMPT_TIMEOUT", "40")),
            attempts=int(os.getenv("KOKORO_TTS_ATTEMPTS", "5")),
            idle_timeout=float(os.getenv("KOKORO_TTS_IDLE_TIMEOUT", "6")),
            opus_rate=int(os.getenv("KOKORO_TTS_OPUS_RATE", str(OPUS_RATE))),
            request_format=os.getenv("KOKORO_TTS_REQUEST_FORMAT", "ogg").lower(),
            **kwargs,
        )

    async def health(self) -> dict[str, Any]:
        try:
            response = await self.client.get(f"{self.base_url}/health", headers=self._headers, timeout=10)
            response.raise_for_status()
            info = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise KokoroError(f"health check failed ({type(exc).__name__})") from None
        voices = info.get("voices")
        if isinstance(voices, list) and self.voice not in voices:
            raise KokoroError(f"voice {self.voice!r} is not offered by the server (available: {voices})")
        return info

    async def _attempt(self, payload: dict[str, Any], state: dict[str, Any]) -> bytes:
        """One request. `state` records how far it got, so a timeout can still be explained."""
        state["started"] = time.monotonic()
        state["headers_at"] = None
        try:
            async with self.client.stream(
                "POST", f"{self.base_url}/v1/audio/speech", json=payload, headers=self._headers
            ) as response:
                state["headers_at"] = time.monotonic() - state["started"]
                if response.status_code != 200:
                    raise KokoroError(f"HTTP {response.status_code}", retry=response.status_code >= 500)
                server = {
                    name: response.headers.get("x-tts-" + name, "?")
                    for name in ("generation-seconds", "queue-seconds", "audio-seconds")
                }
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_AUDIO_BYTES:
                        raise KokoroError("response too large", retry=False)
        except httpx.HTTPError as exc:
            raise KokoroError(f"request failed ({type(exc).__name__}, {self._where(state)})") from None
        logger.info(
            "Kokoro TTS: total %.1fs (headers after %.1fs); server generation %ss, queue %ss, audio %ss; %d bytes",
            time.monotonic() - state["started"],
            state["headers_at"],
            server["generation-seconds"],
            server["queue-seconds"],
            server["audio-seconds"],
            len(data),
        )
        return bytes(data)

    @staticmethod
    def _where(state: dict[str, Any]) -> str:
        # Where the time went tells network trouble from a slow server: no headers at all means the request
        # never got an answer; headers but no body means the transfer stalled.
        spent = time.monotonic() - state["started"]
        if state["headers_at"] is None:
            return f"no response headers after {spent:.1f}s"
        return f"stalled after headers ({state['headers_at']:.1f}s), {spent:.1f}s in total"

    async def _fetch_with_retries(self, payload: dict[str, Any]) -> bytes:
        """Short attempts instead of one long wait.

        On a lossy network a request either completes in seconds or stalls until the connection dies,
        so a stalled attempt is abandoned (its connection is dropped) and repeated within the budget.
        """
        deadline = time.monotonic() + max(1.0, self.timeout - 2)
        failures: list[str] = []
        for attempt in range(1, self.attempts + 1):
            remaining = deadline - time.monotonic()
            if remaining < 1:
                break
            state: dict[str, Any] = {"started": time.monotonic(), "headers_at": None}
            try:
                return await asyncio.wait_for(
                    self._attempt(payload, state), timeout=min(self.attempt_timeout, remaining)
                )
            except TimeoutError:
                failures.append(f"timed out, {self._where(state)}")
            except KokoroError as exc:
                if not exc.retry:
                    raise
                failures.append(str(exc))
            logger.info("Kokoro TTS attempt %d/%d failed (%s)", attempt, self.attempts, failures[-1])
        raise KokoroError(f"no audio after {len(failures)} attempts: " + "; ".join(failures))

    def _payload(self, text: str, voice: str | None) -> dict[str, Any]:
        return {
            "model": "kokoro",
            "input": text,
            "voice": voice or self.voice,
            "speed": self.speed,
            "response_format": self.request_format,
        }

    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        text = text.strip()
        if not 1 <= len(text) <= MAX_TEXT:
            raise ValueError("Kokoro text must be 1..5000 characters")
        async with self._slots:
            try:
                data = await self._fetch_with_retries(self._payload(text, voice))
            except KokoroError as exc:
                if self.request_format != "ogg" or "HTTP 422" not in str(exc):
                    raise
                # An older server (API 1.1) validates response_format as "wav" only.
                logger.warning("Kokoro server does not accept response_format=ogg; switching to wav")
                self.request_format = "wav"
                data = await self._fetch_with_retries(self._payload(text, voice))
        if data[:4] == b"RIFF" and data[8:12] == b"WAVE" and len(data) >= 44:
            seconds = wav_seconds(data)
            if self.output == "wav":
                return SynthesizedAudio(bytes(data), "audio/wav", seconds)
        elif data[:4] == b"OggS":
            seconds = audio_seconds(data)
        else:
            raise KokoroError("the server returned neither WAV nor OGG audio", retry=False)
        try:
            ogg = await asyncio.to_thread(to_ogg_opus, bytes(data), target_rate=self.opus_rate)
        except ValueError as exc:
            raise KokoroError(f"audio conversion failed: {exc}", retry=False) from None
        return SynthesizedAudio(ogg, "audio/ogg", seconds)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()
