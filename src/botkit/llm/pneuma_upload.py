"""Async adaptation of audio_processing_client.py / resumable_chunked_upload_example.ipynb.

The supplied examples are the wire-contract reference, not executed/imported code.
Bytes stay in RAM; only upload/job IDs and offsets are persisted for resumption.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import ssl
import time
import weakref
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import httpx

from botkit.llm.base import LLMResponse
from botkit.usage import UsageEvent, current_context, safe_log

from .gateway import ProviderError
from .upload_state import MemoryUploadStore, UploadStore

SUCCESS = {"completed", "complete", "done", "success", "succeeded", "finished"}
FAILED = {"failed", "error", "cancelled", "canceled"}
PENDING = {"queued", "pending", "running", "processing", "submitted", "waiting", "cancelling", "unknown"}


class UploadError(ProviderError):
    def __init__(self, message="Pneuma request failed", *, status=None):
        super().__init__(message, status_code=status)
        self.status = status


class UploadConnectionError(UploadError):
    pass


def env_bool(name, default=True):
    value = os.getenv(name, str(default)).lower()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be true or false")
    return value in {"true", "1", "yes"}


@dataclass(frozen=True)
class UploadConfig:
    base_url: str
    user_id: str = ""  # If empty, use the application's pseudonymous user ID.
    api_key: str = field(default="", repr=False)
    device: str = "cuda"
    chunk_bytes: int = 16 * 1024 * 1024
    timeout: float = 600
    connect_timeout: float = 15
    request_timeout: float = 30
    poll_interval: float = 5
    retry_delays: tuple = (2, 5, 10)
    verify_ssl: bool = True
    ca_bundle: str | None = None
    trust_env: bool = True
    max_bytes: int = 20 * 1024 * 1024
    max_response_bytes: int = 2 * 1024 * 1024

    def __post_init__(self):
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("PNEUMA_BASE_URL must be an HTTP(S) base URL without credentials/query")
        for name in (
            "timeout",
            "connect_timeout",
            "request_timeout",
            "poll_interval",
            "chunk_bytes",
            "max_bytes",
            "max_response_bytes",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"Invalid Pneuma {name}")
        for name in ("chunk_bytes", "max_bytes", "max_response_bytes"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"Pneuma {name} must be an integer")
        if any(not math.isfinite(v) or v < 0 for v in self.retry_delays):
            raise ValueError("Invalid Pneuma retry delays")
        if not self.device or len(self.user_id) > 200 or any(c in self.api_key for c in "\r\n"):
            raise ValueError("Invalid Pneuma configuration")

    @classmethod
    def from_env(cls, *, max_bytes: int | None = None):
        return cls(
            base_url=os.getenv("PNEUMA_BASE_URL") or os.getenv("TRANSCRIPTION_API_BASE", ""),
            user_id=os.getenv("PNEUMA_USER_ID") or os.getenv("TRANSCRIPTION_USER_ID", ""),
            api_key=os.getenv("PNEUMA_API_KEY", ""),
            device=os.getenv("PNEUMA_DEVICE", "cuda"),
            timeout=float(os.getenv("PNEUMA_TIMEOUT", "600")),
            connect_timeout=float(os.getenv("PNEUMA_CONNECT_TIMEOUT", "15")),
            request_timeout=float(os.getenv("PNEUMA_REQUEST_TIMEOUT", "30")),
            poll_interval=float(os.getenv("PNEUMA_POLL_INTERVAL", "5")),
            verify_ssl=env_bool("PNEUMA_VERIFY_SSL"),
            ca_bundle=os.getenv("PNEUMA_CA_BUNDLE") or None,
            trust_env=env_bool("PNEUMA_TRUST_ENV"),
            max_bytes=int(os.getenv("MAX_ATTACHMENT_BYTES", str(20 * 1024 * 1024)))
            if max_bytes is None
            else max_bytes,
        )


class ChunkedPneumaTranscriber:
    model_name = "pneuma-server-asr"

    def __init__(self, config: UploadConfig, *, tracker=None, client=None, store: UploadStore | None = None):
        self.config, self.tracker = config, tracker
        self.store = store if store is not None else MemoryUploadStore()
        self.owns_client = client is None
        self.client = client or httpx.AsyncClient(
            verify=ssl.create_default_context(cafile=config.ca_bundle)
            if config.ca_bundle
            else config.verify_ssl,
            trust_env=config.trust_env,
            follow_redirects=False,
        )
        self.locks = weakref.WeakValueDictionary()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def aclose(self):
        if self.owns_client:
            await self.client.aclose()

    async def _request(self, method, path, *, headers=None, **kwargs):
        auth = {"Accept": "application/json", **(headers or {})}
        if self.config.api_key:
            auth["Authorization"] = "Bearer " + self.config.api_key
        try:
            async with self.client.stream(
                method,
                self.config.base_url.rstrip("/") + "/" + path,
                headers=auth,
                follow_redirects=False,
                timeout=httpx.Timeout(self.config.request_timeout, connect=self.config.connect_timeout),
                **kwargs,
            ) as response:
                if not response.is_success:
                    raise UploadError("Pneuma HTTP request failed", status=response.status_code)
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > self.config.max_response_bytes:
                        raise UploadError("Pneuma response too large")
                if not body and method == "PATCH":
                    return {}
                try:
                    value = json.loads(body)
                except (ValueError, UnicodeError, RecursionError):
                    raise UploadError("Pneuma returned invalid JSON") from None
                if not isinstance(value, dict):
                    raise UploadError("Pneuma returned an invalid response object")
                return value
        except httpx.HTTPError:
            raise UploadConnectionError("Pneuma connection failed") from None

    async def _post(self, path, payload):
        try:
            return await self._request("POST", path, json=payload)
        except UploadError as exc:
            if exc.status not in {415, 422}:
                raise
        return await self._request("POST", path, data=payload)

    @staticmethod
    def _id(value):
        # Path segments only; never follow status_url/result_url from responses.
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise UploadError("Pneuma returned an invalid identifier")
        value = str(value)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value):
            raise UploadError("Pneuma returned an invalid identifier")
        return value

    @classmethod
    def _job(cls, value):
        candidates = [
            value[name] for name in ("job_id", "transcription_job_id") if value.get(name) is not None
        ]
        nested = value.get("job")
        if nested is not None:
            if not isinstance(nested, dict):
                raise UploadError("Pneuma returned an invalid job object")
            candidates.extend(nested[name] for name in ("job_id", "id") if nested.get(name) is not None)
        identifiers = [cls._id(candidate) for candidate in candidates]
        if len(set(identifiers)) > 1:
            raise UploadError("Pneuma returned conflicting job identifiers")
        return identifiers[0] if identifiers else None

    @staticmethod
    def _offset(value, total):
        candidate = value.get("received_bytes")
        if isinstance(candidate, str) and candidate.isascii() and candidate.isdecimal():
            if len(candidate) > 20:
                raise UploadError("Pneuma returned an invalid received_bytes offset")
            candidate = int(candidate)
        if type(candidate) is not int or not 0 <= candidate <= total:
            raise UploadError("Pneuma returned an invalid received_bytes offset")
        return candidate

    @staticmethod
    def _retryable(error):
        return isinstance(error, UploadConnectionError) or error.status in {408, 409, 429, 500, 502, 503, 504}

    async def _info(self, upload_id, event):
        for attempt in range(len(self.config.retry_delays) + 1):
            try:
                return await self._request("GET", f"uploads/{upload_id}")
            except UploadError as exc:
                if not self._retryable(exc) or attempt == len(self.config.retry_delays):
                    raise
                event.attempts += 1
                await asyncio.sleep(self.config.retry_delays[attempt])

    async def _upload(self, data, payload, state, key, event):
        total = len(data)
        if state.get("upload_id"):
            upload_id = self._id(state["upload_id"])
            try:
                info = await self._info(upload_id, event)
                offset = self._offset(info, total)
                if job := self._job(info):
                    state.update(job_id=job)
                    await self.store.put(key, state)
                    return job
            except UploadError as exc:
                if exc.status not in {404, 410}:
                    raise
                state.clear()
        if not state.get("upload_id"):
            info = await self._post("uploads", payload)
            upload_id = self._id(info.get("upload_id"))
            offset = self._offset({"received_bytes": info.get("received_bytes", 0)}, total)
            state.update(upload_id=upload_id, received_bytes=offset)
            await self.store.put(key, state)

        while offset < total:
            chunk = data[offset : offset + self.config.chunk_bytes]
            previous = offset
            for attempt in range(len(self.config.retry_delays) + 1):
                try:
                    info = await self._request(
                        "PATCH",
                        f"uploads/{upload_id}",
                        content=chunk,
                        headers={"Upload-Offset": str(previous), "Content-Type": "application/octet-stream"},
                    )
                    if "received_bytes" not in info:
                        info = await self._info(upload_id, event)
                    offset = self._offset(info, total)
                except UploadError as exc:
                    if not self._retryable(exc):
                        raise
                    # Lost PATCH reply is not proof of failure. Reconcile before resending.
                    info = await self._info(upload_id, event)
                    offset = self._offset(info, total)
                    event.meta["reconciliations"] += 1
                if offset < previous or offset > previous + len(chunk):
                    raise UploadError("Pneuma offset does not match the submitted chunk")
                if offset > previous:
                    state["received_bytes"] = offset
                    await self.store.put(key, state)
                    break
                if attempt == len(self.config.retry_delays):
                    raise UploadError("Pneuma did not acknowledge the chunk")
                event.attempts += 1
                await asyncio.sleep(self.config.retry_delays[attempt])
            event.meta["chunks"] += 1
        info = await self._info(upload_id, event)
        if self._offset(info, total) != total:
            raise UploadError("Pneuma upload is incomplete")
        if job := self._job(info):
            state["job_id"] = job
            await self.store.put(key, state)
            return job

        for attempt in range(len(self.config.retry_delays) + 1):
            try:
                result = await self._post(f"uploads/{upload_id}/complete", payload)
                job = self._job(result)
                if not job:
                    raise UploadConnectionError("Pneuma did not return a job ID")
            except UploadError as exc:
                if not self._retryable(exc):
                    raise
                # /complete may already have created the job. Never blindly recreate it.
                info = await self._info(upload_id, event)
                job = self._job(info)
                event.meta["reconciliations"] += 1
                if not job:
                    if attempt == len(self.config.retry_delays):
                        raise UploadError("Pneuma completion is not confirmed") from None
                    event.attempts += 1
                    await asyncio.sleep(self.config.retry_delays[attempt])
                    continue
            state["job_id"] = job
            await self.store.put(key, state)
            return job

    @staticmethod
    def _transcript(value, depth=0):
        if depth > 64:
            raise UploadError("Pneuma transcript nesting is invalid")
        if isinstance(value, str):
            return value.strip()
        # Structured segments are accepted only when they contain explicit text.
        if isinstance(value, dict):
            for key in ("text", "transcript", "segments"):
                if key in value:
                    return ChunkedPneumaTranscriber._transcript(value[key], depth + 1)
        if isinstance(value, list) and value:
            return "\n".join(ChunkedPneumaTranscriber._transcript(part, depth + 1) for part in value)
        raise UploadError("Pneuma has no textual transcript")

    async def _result(self, job_id, event):
        while True:
            try:
                value = await self._request("GET", f"transcriptions/{job_id}")
                if "job_id" in value and self._id(value["job_id"]) != job_id:
                    raise UploadError("Pneuma returned a different job ID")
                status = str(value.get("status", "unknown")).lower()
                if status in FAILED:
                    raise UploadError("Pneuma processing failed")
                if status in SUCCESS:
                    result = await self._request("GET", f"transcriptions/{job_id}/result")
                    if "job_id" in result and self._id(result["job_id"]) != job_id:
                        raise UploadError("Pneuma returned a different result ID")
                    result_status = result.get("status", "completed")
                    if not isinstance(result_status, str):
                        raise UploadError("Pneuma returned an invalid result status")
                    result_status = result_status.lower()
                    if result_status in FAILED:
                        raise UploadError("Pneuma processing failed")
                    if result_status not in SUCCESS | PENDING:
                        raise UploadError("Pneuma returned an unknown result status")
                    if "ready" in result and type(result["ready"]) is not bool:
                        raise UploadError("Pneuma returned an invalid result readiness")
                    if result.get("ready") is False or result_status in PENDING:
                        event.poll_count += 1
                        await asyncio.sleep(self.config.poll_interval)
                        continue
                    text = self._transcript(result.get("transcript"))
                    if not 1 <= len(text) <= 20000:
                        raise UploadError("Pneuma transcript length is invalid")
                    return text
                if status not in PENDING:
                    raise UploadError("Pneuma returned an unknown job status")
            except UploadError as exc:
                if not self._retryable(exc):
                    raise
            event.poll_count += 1
            await asyncio.sleep(self.config.poll_interval)

    async def atranscribe(self, data, *, mime_type, filename, timeout=None, language=None):
        if not data or len(data) > self.config.max_bytes:
            raise ValueError("Audio must be nonempty and within its size limit")
        suffix = filename.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[-1].lower()
        if suffix not in {"mp3", "ogg", "opus", "wav", "m4a", "webm", "flac", "aac", "mp4", "wma"}:
            raise ValueError("Unsupported audio filename")
        if not mime_type.startswith("audio/") or any(c in mime_type for c in "\r\n") or language is not None:
            raise ValueError("Unsupported ASR arguments")
        if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("Invalid timeout")
        context = current_context()
        user = self.config.user_id or context.get("user_id", "")
        if not isinstance(user, str) or not user.strip() or len(user) > 200:
            raise ValueError("A pseudonymous user ID is required for Pneuma")
        key = hashlib.sha256(
            json.dumps(
                [
                    self.config.base_url,
                    user,
                    self.config.device,
                    suffix,
                    context.get("request_id", ""),
                    hashlib.sha256(data).hexdigest(),
                ]
            ).encode()
        ).hexdigest()
        event = UsageEvent(
            provider="pneuma",
            model=self.model_name,
            operation="transcription",
            request_id=context.get("request_id", ""),
            cost_status="UNKNOWN_USAGE",
            meta={"byte_count": len(data), "upload_protocol": "chunked", "chunks": 0, "reconciliations": 0},
        )
        started = time.monotonic()
        lock = self.locks.setdefault(key, asyncio.Lock())
        try:
            async with asyncio.timeout(min(timeout or self.config.timeout, self.config.timeout)), lock:
                await self.store.prune()
                state = await self.store.get(key)
                job = state.get("job_id") or await self._upload(
                    data,
                    {
                        "filename": "voice." + suffix,
                        "size": len(data),
                        "user_id": user,
                        "device": self.config.device,
                    },
                    state,
                    key,
                    event,
                )
                job = self._id(job)
                event.provider_request_id = job
                text = await self._result(job, event)
                event.meta["text_chars"] = len(text)
                return LLMResponse(
                    text,
                    text,
                    "raw_text_fallback",
                    provider="pneuma",
                    model=self.model_name,
                    request_id=job,
                    meta={"job_id": job, "poll_count": event.poll_count, **event.meta},
                )
        except BaseException as exc:
            event.status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            event.error_type = type(exc).__name__
            event.http_status = getattr(exc, "status", None)
            raise
        finally:
            event.latency_ms = (time.monotonic() - started) * 1000
            await safe_log(self.tracker, event)
