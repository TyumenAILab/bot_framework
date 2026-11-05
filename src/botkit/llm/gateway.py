from __future__ import annotations

import asyncio
import base64
import json
import math
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any
from uuid import uuid4

import httpx

from botkit.usage import UsageEvent, UsageTracker, current_context, safe_log

from .base import LLMResponse, StreamChunk
from .config import LLMConfig
from .pacing import request_slot
from .parsing import parse_response


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, reason: str = "unknown"):
        super().__init__(message)
        self.status_code = status_code
        # Never expose provider text, URL, body or credentials as diagnostics.
        self.reason = reason if reason in {
            "unknown", "network_connect", "network_timeout", "network_other", "http_status",
            "invalid_json", "invalid_response"
        } else "unknown"


def _usage(data: dict[str, Any]) -> dict[str, Any]:
    # Optional accounting metadata must never turn valid generated text into a
    # TypeError, or turn booleans/string counts into apparently measured usage.
    usage = data.get("usage")
    usage = usage if isinstance(usage, dict) else {}

    def valid(value: Any) -> bool:
        return type(value) is int and value >= 0

    def count(value: Any) -> int:
        return value if valid(value) else 0

    prompt_details = usage.get("prompt_tokens_details")
    output_details = usage.get("completion_tokens_details")
    prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
    output_details = output_details if isinstance(output_details, dict) else {}
    known = all(valid(usage.get(k)) for k in ("prompt_tokens", "completion_tokens"))
    return {
        "input_tokens": count(usage.get("prompt_tokens")),
        "output_tokens": count(usage.get("completion_tokens")),
        "cached_tokens": count(prompt_details.get("cached_tokens")),
        "reasoning_tokens": count(output_details.get("reasoning_tokens")),
        "usage_known": known,
    }


def _tools(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ProviderError("Provider returned invalid tool calls", reason="invalid_response")
    return value


def _finish_reason(choice: dict[str, Any]) -> str | None:
    value = choice.get("finish_reason")
    if value is not None and not isinstance(value, str):
        raise ProviderError("Provider returned an invalid finish reason", reason="invalid_response")
    return value


def _decode(data: dict[str, Any], config: LLMConfig) -> LLMResponse:
    if data.get("error"):
        raise ProviderError("Provider returned an API error", reason="invalid_response")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError("Provider response has no choices", reason="invalid_response")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ProviderError("Provider response has an invalid choice/message", reason="invalid_response")
    message = choice["message"]
    content = message.get("content")
    if isinstance(content, list):
        if any(not isinstance(item, dict) for item in content):
            raise ProviderError("Provider response has invalid content blocks", reason="invalid_response")
        text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
        if any(not isinstance(part, str) for part in text_parts):
            raise ProviderError("Provider response has invalid text blocks", reason="invalid_response")
        content = "".join(text_parts)
    tools = _tools(message.get("tool_calls"))
    if content is None and (tools or message.get("refusal")):
        content = message.get("refusal") or ""
    if not isinstance(content, str):
        raise ProviderError("Provider response has no text or tool calls", reason="invalid_response")
    response, status = parse_response(content)
    return LLMResponse(
        response=response,
        raw=content,
        parse_status=status,
        model=data.get("model") if isinstance(data.get("model"), str) else config.model,
        provider=config.provider,
        request_id=data.get("id") if isinstance(data.get("id"), str) else "",
        finish_reason=_finish_reason(choice),
        tool_calls=tools,
        **_usage(data),
        meta={"usage": data.get("usage"), "system_fingerprint": data.get("system_fingerprint")},
    )


class LLMGateway:
    """OpenAI wire protocol on httpx, shared by 302.ai and vLLM.

    SDK-independent messages/tools/response_format/extra_body are forwarded.
    There is one deadline for submission, backoff and polling combined.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        tracker: UsageTracker | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self.model_name = config.model
        self.tracker = tracker
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(
            timeout=config.timeout,
            follow_redirects=False,
            verify=config.tls_context(),
            trust_env=config.trust_env,
        )

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"} if self.config.api_key else {}

    async def __aenter__(self) -> LLMGateway:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _timeout(self, value: float | None) -> float:
        value = self.config.timeout if value is None else value
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("timeout must be finite and positive")
        return value

    def _payload(self, messages: Sequence[dict[str, Any]], options: dict[str, Any]) -> dict[str, Any]:
        # The original organizer uses max_tokens=None to omit the cap entirely.
        options = {key: value for key, value in options.items() if value is not None}
        extra = options.pop("extra_body", {})
        forbidden = {"model", "messages", "stream"}.intersection(options.keys() | extra.keys())
        if forbidden:
            raise ValueError(f"Use the gateway configuration/method for: {sorted(forbidden)}")
        if options.get("n", extra.get("n", 1)) != 1:
            raise ValueError("The gateway returns one response; n must be 1")
        return {"model": self.model_name, "messages": list(messages), **options, **extra}

    def _event(self, operation: str, messages: Any) -> UsageEvent:
        return UsageEvent(
            provider=self.config.provider,
            model=self.model_name,
            operation=operation,
            request_id=current_context().get("request_id") or uuid4().hex,
            input_text=json.dumps(messages, ensure_ascii=False),
        )

    async def _request(self, method: str, path: str, event: UsageEvent, **kwargs: Any) -> dict[str, Any]:
        for attempt in range(self.config.max_retries + 1):
            try:
                async with request_slot(self.config.base_url, self.config.request_interval):
                    response = await self.client.request(
                        method,
                        self.config.base_url.rstrip("/") + "/" + path.lstrip("/"),
                        headers=self._headers,
                        timeout=self.config.timeout,
                        follow_redirects=False,
                        **kwargs,
                    )
            except httpx.HTTPError as exc:
                # Retrying an ambiguous POST could bill the user twice. Let the
                # caller decide; never include exception URLs/headers in logs.
                event.meta["network_error"] = type(exc).__name__
                reason = (
                    "network_timeout" if isinstance(exc, httpx.TimeoutException)
                    else "network_connect" if isinstance(exc, httpx.ConnectError)
                    else "network_other"
                )
                raise ProviderError(f"Provider network failure ({type(exc).__name__})", reason=reason) from None
            event.http_status = response.status_code
            if response.status_code in {429, 503} and attempt < self.config.max_retries:
                event.attempts += 1
                try:
                    delay = max(0, float(response.headers.get("retry-after", 2**attempt)))
                except ValueError:
                    delay = 2**attempt
                await asyncio.sleep(delay)
                continue
            if not response.is_success:
                raise ProviderError(f"Provider HTTP {response.status_code}", status_code=response.status_code, reason="http_status")
            try:
                data = response.json()
            except (ValueError, RecursionError):
                raise ProviderError("Provider returned invalid JSON", reason="invalid_json") from None
            if not isinstance(data, dict):
                raise ProviderError("Provider response must be a JSON object", reason="invalid_response")
            return data
        raise AssertionError("Unreachable")

    async def _completion(self, payload: dict[str, Any], event: UsageEvent) -> dict[str, Any]:
        params = {"async": "true"} if self.config.mode == "async" else None
        data = await self._request("POST", "chat/completions", event, json=payload, params=params)
        if self.config.mode != "async" or "choices" in data:
            return data
        task_id = data.get("task_id")
        if not task_id:
            raise ProviderError("302.ai did not return task_id")
        event.meta["task_id"] = task_id
        while True:
            await asyncio.sleep(self.config.poll_interval)
            event.poll_count += 1
            result = await self._request("GET", "async_result", event, params={"task_id": task_id})
            status, error, body = result.get("status_code"), result.get("err"), result.get("data")
            pending = str(error or "").lower() in {"pending", "result pending"}
            if pending or status in {0, 102, 202} or (status == 200 and not body and not error):
                continue
            if status != 200 or error:
                raise ProviderError("302.ai asynchronous task failed", status_code=status)
            if isinstance(body, str):
                try:
                    decoded = json.loads(body)
                except ValueError:
                    decoded = None
                if isinstance(decoded, dict) and "choices" in decoded:
                    return decoded
                return {"choices": [{"message": {"content": body}, "finish_reason": "stop"}]}
            if isinstance(body, dict) and "choices" in body:
                return body
            if isinstance(body, dict) and "response" in body:
                return {"choices": [{"message": {"content": json.dumps(body, ensure_ascii=False)}}]}
            raise ProviderError("302.ai task returned an unsupported result")

    @staticmethod
    def _populate(event: UsageEvent, result: LLMResponse) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "usage_known",
            "parse_status",
            "finish_reason",
        ):
            setattr(event, name, getattr(result, name))
        event.output_text = result.raw
        event.provider_request_id = result.request_id
        event.meta.update(
            {
                "reported_model": result.model,
                "usage": result.meta.get("usage"),
                "tool_call_count": len(result.tool_calls),
            }
        )

    async def ainvoke(self, prompt: str, *, timeout: float | None = None, **options: Any) -> LLMResponse:
        return await self.achat([{"role": "user", "content": prompt}], timeout=timeout, **options)

    async def achat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        timeout: float | None = None,
        validator: Callable[[LLMResponse], None] | None = None,
        **options: Any,
    ) -> LLMResponse:
        timeout = self._timeout(timeout)
        payload = self._payload(messages, options)
        payload["stream"] = False
        event = self._event("chat", messages)
        event.meta["request_options"] = {k: v for k, v in payload.items() if k != "messages"}
        event.meta["mode"] = self.config.mode
        start = time.perf_counter()
        try:
            async with asyncio.timeout(timeout):
                data = await self._completion(payload, event)
                result = _decode(data, self.config)
            self._populate(event, result)
            if validator:
                validator(result)
            result.meta.update({"attempts": event.attempts, "poll_count": event.poll_count})
            return result
        except asyncio.CancelledError:
            event.status, event.error_type = "cancelled", "CancelledError"
            raise
        except Exception as exc:
            event.status, event.error_type = "error", type(exc).__name__
            raise
        finally:
            event.latency_ms = (time.perf_counter() - start) * 1000
            await safe_log(self.tracker, event)

    async def ainvoke_with_image_b64(
        self, prompt: str, image_b64: str, *, mime_type: str = "image/png", **options: Any
    ) -> LLMResponse:
        if not mime_type.startswith("image/"):
            raise ValueError("Image MIME type required")
        base64.b64decode(image_b64, validate=True)
        return await self.achat(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}},
                    ],
                }
            ],
            **options,
        )

    async def ainvoke_with_image(
        self, prompt: str, image: bytes, *, mime_type: str = "image/png", **options: Any
    ) -> LLMResponse:
        return await self.ainvoke_with_image_b64(
            prompt, base64.b64encode(image).decode("ascii"), mime_type=mime_type, **options
        )

    async def astream(
        self, messages: Sequence[dict[str, Any]], *, timeout: float | None = None, **options: Any
    ) -> AsyncIterator[StreamChunk]:
        """SSE streaming. Use contextlib.aclosing when consuming only part of a stream.

        No replay/retry after output has started. Cancellation/early close is logged.
        Server must support stream_options to report exact streamed usage.
        """
        if self.config.mode == "async":
            raise ValueError("Use A302_MODE=chat for streaming")
        if not self.config.supports_streaming:
            raise ValueError("Streaming is disabled for this provider")
        timeout = self._timeout(timeout)
        payload = self._payload(messages, options)
        payload.update(stream=True)
        payload.setdefault("stream_options", {"include_usage": True})
        event = self._event("stream", messages)
        start = time.perf_counter()
        parts: list[str] = []
        complete = False
        try:
            async with asyncio.timeout(timeout):
                async with (
                    request_slot(self.config.base_url, self.config.request_interval),
                    self.client.stream(
                        "POST",
                        self.config.base_url.rstrip("/") + "/chat/completions",
                        headers=self._headers,
                        json=payload,
                        timeout=self.config.timeout,
                        follow_redirects=False,
                    ) as response,
                ):
                    event.http_status = response.status_code
                    if not response.is_success:
                        raise ProviderError(
                            f"Provider HTTP {response.status_code}", status_code=response.status_code
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        value = line[5:].strip()
                        if value == "[DONE]":
                            complete = True
                            break
                        try:
                            data = json.loads(value)
                        except (ValueError, RecursionError):
                            raise ProviderError("Provider stream returned invalid JSON") from None
                        if not isinstance(data, dict):
                            raise ProviderError("Provider stream returned an invalid response object")
                        if data.get("error"):
                            raise ProviderError("Provider stream returned an API error")
                        if isinstance(data.get("id"), str):
                            event.provider_request_id = data["id"] or event.provider_request_id
                        usage = data.get("usage")
                        if usage:
                            for name, count in _usage(data).items():
                                setattr(event, name, count)
                        choices = data.get("choices", [])
                        if not isinstance(choices, list) or any(
                            not isinstance(item, dict) for item in choices
                        ):
                            raise ProviderError("Provider stream returned invalid choices")
                        choice = choices[0] if choices else {}
                        delta = choice.get("delta", {})
                        if not isinstance(delta, dict):
                            raise ProviderError("Provider stream returned an invalid delta")
                        text = delta.get("content")
                        text = "" if text is None else text
                        if not isinstance(text, str):
                            raise ProviderError("Provider stream returned non-text content")
                        tools = _tools(delta.get("tool_calls"))
                        finish_reason = _finish_reason(choice)
                        if text and event.first_token_ms is None:
                            event.first_token_ms = (time.perf_counter() - start) * 1000
                        parts.append(text)
                        event.finish_reason = finish_reason or event.finish_reason
                        yield StreamChunk(
                            text, tools, finish_reason, usage if isinstance(usage, dict) else None
                        )
                    if not complete:
                        raise ProviderError("Provider stream ended before [DONE]")
        except (asyncio.CancelledError, GeneratorExit):
            event.status, event.error_type = "cancelled", "StreamCancelled"
            raise
        except Exception as exc:
            event.status, event.error_type = "error", type(exc).__name__
            raise
        finally:
            event.output_text = "".join(parts)
            _, event.parse_status = parse_response(event.output_text)
            event.latency_ms = (time.perf_counter() - start) * 1000
            await safe_log(self.tracker, event)

    async def aembed(
        self, texts: str | list[str], *, timeout: float | None = None, **options: Any
    ) -> list[list[float]]:
        timeout = self._timeout(timeout)
        if "model" in options or "input" in options:
            raise ValueError("Configure the embedding model on a separate gateway")
        event = self._event("embeddings", texts)
        start = time.perf_counter()
        try:
            async with asyncio.timeout(timeout):
                data = await self._request(
                    "POST", "embeddings", event, json={"model": self.model_name, "input": texts, **options}
                )
                usage = data.get("usage") or {}
                event.input_tokens = usage.get("prompt_tokens") or 0
                event.usage_known = "prompt_tokens" in usage
                values = sorted(data["data"], key=lambda item: item["index"])
                return [item["embedding"] for item in values]
        except asyncio.CancelledError:
            event.status, event.error_type = "cancelled", "CancelledError"
            raise
        except Exception as exc:
            event.status, event.error_type = "error", type(exc).__name__
            raise
        finally:
            event.latency_ms = (time.perf_counter() - start) * 1000
            await safe_log(self.tracker, event)

    async def list_models(self) -> list[dict[str, Any]]:
        async with asyncio.timeout(self.config.timeout):
            data = await self._request("GET", "models", self._event("models", []))
            return data.get("data", [])

    async def atranscribe(
        self,
        data: bytes,
        *,
        mime_type: str,
        filename: str = "audio.wav",
        timeout: float | None = None,
        language: str | None = None,
    ) -> LLMResponse:
        """Optional OpenAI-compatible ASR endpoint; requires an ASR deployment.

        No assumption that the configured chat/vision model supports audio.
        Audio bytes are excluded from telemetry, even when text logging is on.
        """
        if self.config.mode != "chat":
            raise ValueError("Transcription does not support 302.ai async polling")
        timeout = self._timeout(timeout)
        event = self._event("transcription", {"mime_type": mime_type, "byte_count": len(data)})
        start = time.perf_counter()
        form = {"model": self.model_name, "response_format": "json"}
        if language is not None:
            form["language"] = language
        try:
            async with asyncio.timeout(timeout):
                body = await self._request(
                    "POST",
                    "audio/transcriptions",
                    event,
                    data=form,
                    files={"file": (filename, data, mime_type)},
                )
            if body.get("error") or not isinstance(body.get("text"), str):
                raise ProviderError("Transcription response has no text")
            result = LLMResponse(
                body["text"],
                body["text"],
                "raw_text_fallback",
                model=self.model_name,
                provider=self.config.provider,
                meta={"usage": body.get("usage")},
                **_usage(body),
            )
            self._populate(event, result)
            return result
        except asyncio.CancelledError:
            event.status, event.error_type = "cancelled", "CancelledError"
            raise
        except Exception as exc:
            event.status, event.error_type = "error", type(exc).__name__
            raise
        finally:
            event.latency_ms = (time.perf_counter() - start) * 1000
            await safe_log(self.tracker, event)
