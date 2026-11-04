"""Client for Pneuma's queued ASR API (not the OpenAI audio API)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from botkit.usage import UsageEvent, UsageTracker, safe_log

from .base import AudioTranscriptionProvider, LLMResponse
from .config import _env_bool
from .gateway import ProviderError


@dataclass(frozen=True)
class PneumaConfig:
    # Include any deployment prefix here, including /api for the web proxy.
    base_url: str
    api_mode: str = "direct"
    api_key: str = field(default="", repr=False)
    timeout: float = 600
    poll_interval: float = 2
    request_timeout: float = 180
    max_bytes: int = 20 * 1024 * 1024
    max_response_bytes: int = 2 * 1024 * 1024
    verify_ssl: bool = True
    ca_bundle: str | None = None
    trust_env: bool = True
    cancel_on_timeout: bool = True

    def __post_init__(self) -> None:
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("PNEUMA_BASE_URL must be an HTTP(S) URL without embedded credentials")
        if url.query or url.fragment:
            raise ValueError("PNEUMA_BASE_URL must not contain query parameters or fragments")
        if self.api_mode not in {"direct", "proxy"}:
            raise ValueError("PNEUMA_API_MODE must be direct or proxy")
        for name in ("timeout", "poll_interval", "request_timeout", "max_bytes", "max_response_bytes"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Pneuma {name} must be finite and positive")
        if any(c in self.api_key for c in "\r\n"):
            raise ValueError("PNEUMA_API_KEY must not contain line breaks")

    @classmethod
    def from_env(cls) -> PneumaConfig:
        return cls(
            base_url=os.getenv("PNEUMA_BASE_URL", ""),
            api_mode=os.getenv("PNEUMA_API_MODE", "direct").strip().lower(),
            api_key=os.getenv("PNEUMA_API_KEY", ""),
            timeout=float(os.getenv("PNEUMA_TIMEOUT", "600")),
            poll_interval=float(os.getenv("PNEUMA_POLL_INTERVAL", "2")),
            request_timeout=float(os.getenv("PNEUMA_REQUEST_TIMEOUT", "180")),
            max_bytes=int(os.getenv("MAX_ATTACHMENT_BYTES", str(20 * 1024 * 1024))),
            verify_ssl=_env_bool("PNEUMA_VERIFY_SSL", True),
            ca_bundle=os.getenv("PNEUMA_CA_BUNDLE") or None,
            trust_env=_env_bool("PNEUMA_TRUST_ENV", True),
            cancel_on_timeout=_env_bool("PNEUMA_CANCEL_ON_TIMEOUT", True),
        )

    def tls_context(self) -> bool | ssl.SSLContext:
        return ssl.create_default_context(cafile=self.ca_bundle) if self.ca_bundle else self.verify_ssl


class PneumaTranscriber:
    """Upload once, poll a job, return the unmodified diarized transcript.

    A supplied HTTP client belongs to the caller. Redirects and automatic upload
    retries are disabled. The API has no per-request model/language selection.
    """

    model_name = "t-one"

    def __init__(
        self,
        config: PneumaConfig,
        *,
        tracker: UsageTracker | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.config, self.tracker = config, tracker
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            verify=config.tls_context(), trust_env=config.trust_env, follow_redirects=False
        )
        # Preselect a stable session: simultaneous first uploads must not race
        # over different Set-Cookie values and lose access to each other's tasks.
        self._session = "web-" + uuid4().hex

    async def __aenter__(self) -> PneumaTranscriber:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = "Bearer " + self.config.api_key
        if self.config.api_mode == "proxy":
            headers["Cookie"] = "pneuma_session=" + self._session
        try:
            async with self.client.stream(
                method,
                self.config.base_url.rstrip("/") + "/" + path,
                headers=headers,
                timeout=self.config.request_timeout,
                follow_redirects=False,
                **kwargs,
            ) as response:
                if not response.is_success:
                    raise ProviderError(
                        "Pneuma HTTP request failed; check endpoint, access and server logs",
                        status_code=response.status_code,
                    )
                chunks, size = [], 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > self.config.max_response_bytes:
                        raise ProviderError("Pneuma response exceeds size limit")
                    chunks.append(chunk)
                try:
                    body = json.loads(b"".join(chunks))
                except (ValueError, UnicodeError, RecursionError):
                    raise ProviderError(
                        "Pneuma returned invalid JSON; check API URL and authorization"
                    ) from None
                if not isinstance(body, dict):
                    raise ProviderError("Pneuma returned an invalid response object")
                return body
        except httpx.TimeoutException:
            raise TimeoutError("Pneuma HTTP request timed out") from None
        except httpx.HTTPError:
            # Never include URLs, credentials or the server's diagnostic body.
            raise ProviderError("Pneuma connection failed") from None

    @staticmethod
    def _status(body: dict[str, Any], job_id: str) -> str:
        if "job_id" in body and body["job_id"] != job_id:
            raise ProviderError("Pneuma returned a different job identifier")
        status = body.get("status")
        if status in ("failed", "cancelled"):
            raise ProviderError(f"Pneuma transcription {status}; check server logs")
        if status not in ("queued", "running", "cancelling", "completed"):
            raise ProviderError("Pneuma returned an unknown job status")
        return status

    @staticmethod
    def _duration(body: dict[str, Any], event: UsageEvent) -> None:
        for key, multiplier in (("audio_duration_sec", 1), ("audio_duration_min", 60)):
            value = body.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if math.isfinite(value) and value >= 0:
                    event.meta["audio_duration_seconds"] = value * multiplier
                    return

    async def _cancel(self, job_id: str, event: UsageEvent) -> None:
        if not job_id or not self.config.cancel_on_timeout or self.config.api_mode != "direct":
            return
        event.meta["cancel_requested"] = True
        try:
            # This small cleanup deadline is additional to the transcription deadline.
            async with asyncio.timeout(5):
                await self._request("POST", f"transcriptions/{job_id}/cancel")
            event.meta["cancel_request_accepted"] = True
        except Exception:
            event.meta["cancel_request_accepted"] = False

    async def atranscribe(
        self,
        data: bytes,
        *,
        mime_type: str,
        filename: str,
        timeout: float | None = None,
        language: str | None = None,
    ) -> LLMResponse:
        if not data or len(data) > self.config.max_bytes:
            raise ValueError("Audio must be nonempty and within the configured attachment size limit")
        if language is not None:
            raise ValueError("Pneuma does not support per-request language selection")
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Transcription timeout must be finite and positive")
        # Do not disclose local paths or user-provided filenames to the server.
        suffix = filename.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[-1].lower()
        if suffix not in {"wav", "flac", "mp3", "m4a", "ogg", "opus", "aac", "wma", "mp4", "webm"}:
            raise ValueError("Unsupported audio filename extension")
        if not mime_type.startswith("audio/") or any(c in mime_type for c in "\r\n"):
            raise ValueError("An audio MIME type is required")
        event = UsageEvent(
            provider="pneuma",
            model=self.model_name,
            operation="transcription",
            request_id=uuid4().hex,
            cost_status="UNKNOWN_USAGE",
            meta={
                "byte_count": len(data),
                "mime_type": mime_type,
                "source_sha256": hashlib.sha256(data).hexdigest(),
                "api_mode": self.config.api_mode,
            },
        )
        started, job_id = time.perf_counter(), ""
        try:
            async with asyncio.timeout(min(timeout or self.config.timeout, self.config.timeout)):
                body = await self._request(
                    "POST", "transcriptions", files={"audio": ("audio." + suffix, data, mime_type)}
                )
                identifier = "task_id" if self.config.api_mode == "proxy" else "job_id"
                candidate = body.get(identifier)
                if not isinstance(candidate, str) or not re.fullmatch(r"[0-9a-f]{32}", candidate):
                    raise ProviderError("Pneuma returned an invalid job identifier")
                job_id = candidate
                event.provider_request_id = job_id
                # Never follow status_url/result_url from a response to another host.
                while True:
                    self._duration(body, event)
                    if self._status(body, job_id) == "completed":
                        result = await self._request("GET", f"transcriptions/{job_id}/result")
                        if self._status(result, job_id) != "completed" or result.get("ready") is not True:
                            raise ProviderError("Pneuma completed job has no ready result")
                        transcript = result.get("transcript")
                        if not isinstance(transcript, str):
                            raise ProviderError("Pneuma result has no transcript")
                        event.parse_status = "raw_text_fallback"
                        event.meta["text_chars"] = len(transcript)
                        return LLMResponse(
                            transcript,
                            transcript,
                            "raw_text_fallback",
                            model=self.model_name,
                            provider="pneuma",
                            request_id=job_id,
                            meta={"job_id": job_id, "poll_count": event.poll_count, **event.meta},
                        )
                    await asyncio.sleep(self.config.poll_interval)
                    event.poll_count += 1
                    body = await self._request("GET", f"transcriptions/{job_id}")
        except asyncio.CancelledError:
            event.status, event.error_type = "cancelled", "CancelledError"
            await self._cancel(job_id, event)
            raise
        except Exception as exc:
            event.status, event.error_type = "error", type(exc).__name__
            event.http_status = getattr(exc, "status_code", None)
            if isinstance(exc, TimeoutError):
                await self._cancel(job_id, event)
            raise
        finally:
            event.latency_ms = (time.perf_counter() - started) * 1000
            # Transcript/audio/server paths are never included, even with LOG_TEXT=true.
            await safe_log(self.tracker, event)


def load_transcriber(*, tracker: UsageTracker | None = None, store=None) -> AudioTranscriptionProvider | None:
    """Opt-in ASR; legacy upload and no LLM processing remain the defaults.

    A supplied store implements UploadStore and is used only for chunked uploads.
    Postprocessing has its own explicit local-model configuration, never the
    primary chatbot's automatic fallback chain.
    """
    provider = os.getenv("ASR_PROVIDER", "").strip().lower()
    if provider in {"", "none", "disabled"}:
        return None
    if provider != "pneuma":
        raise ValueError("ASR_PROVIDER must be pneuma or disabled")
    from .pneuma_upload import ChunkedPneumaTranscriber, UploadConfig
    from .transcript import TranscriptionPipeline, load_transcript_processor
    from .upload_state import SQLiteUploadStore

    protocol = os.getenv("PNEUMA_UPLOAD_PROTOCOL", "legacy").strip().lower()
    if protocol not in {"legacy", "chunked"}:
        raise ValueError("PNEUMA_UPLOAD_PROTOCOL must be legacy or chunked")
    if protocol == "legacy" and store is not None:
        raise ValueError("Resumable upload stores require PNEUMA_UPLOAD_PROTOCOL=chunked")
    config = UploadConfig.from_env() if protocol == "chunked" else PneumaConfig.from_env()
    enabled = _env_bool("ASR_POSTPROCESS_ENABLED", False)
    processor = load_transcript_processor(tracker=tracker) if enabled else None
    if protocol == "chunked":
        if store is None and (path := os.getenv("PNEUMA_UPLOAD_STATE_DB", "").strip()):
            store = SQLiteUploadStore(path)
        transcriber = ChunkedPneumaTranscriber(config, tracker=tracker, store=store)
    else:
        transcriber = PneumaTranscriber(config, tracker=tracker)
    if processor is not None:
        return TranscriptionPipeline(transcriber, processor, owns_components=True)
    return transcriber
