from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import aclosing
from typing import Any, TypeVar
from uuid import uuid4

from botkit.usage import UsageEvent, UsageTracker, current_context, safe_log, usage_context

from .base import LLMProvider, LLMResponse, StreamChunk
from .gateway import ProviderError

logger = logging.getLogger(__name__)
T = TypeVar("T")
ResponseValidator = Callable[[LLMResponse], None]


class InvalidLLMResponse(ValueError):
    """A protocol-successful result that requires trying another model."""


class DoNotFallbackError(Exception):
    """A domain result such as 'not a table'; do not spend a second model call."""


class FallbackExhaustedError(ProviderError):
    def __init__(self, failures: list[dict[str, Any]]):
        super().__init__("All configured LLM providers failed")
        self.failures = failures


def validate_json_response(response: LLMResponse) -> None:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.raw.strip(), flags=re.IGNORECASE)
    try:
        json.loads(text)
    except (ValueError, TypeError, RecursionError):
        raise InvalidLLMResponse("Model did not return valid JSON") from None


class FallbackLLM:
    """One logical call, separate recorded attempt for every model actually used."""

    def __init__(
        self,
        primary: LLMProvider,
        fallbacks: Sequence[LLMProvider],
        *,
        tracker: UsageTracker | None = None,
        timeout: float = 300,
        primary_timeout: float = 90,
        validator: ResponseValidator | None = None,
    ):
        if (
            not math.isfinite(timeout)
            or not math.isfinite(primary_timeout)
            or timeout <= 0
            or primary_timeout <= 0
        ):
            raise ValueError("Timeouts must be finite and positive")
        self.providers = [primary, *fallbacks]
        self.model_name = primary.model_name
        self.tracker, self.timeout, self.primary_timeout = tracker, timeout, primary_timeout
        self.validator = validator

    def _scope(self, chain_id: str, index: int, failures: list[dict[str, Any]]):
        context = current_context()
        return usage_context(
            request_id=context.get("request_id") or chain_id,
            meta={
                **context.get("meta", {}),
                "fallback_chain_id": chain_id,
                "fallback_index": index,
                "fallback_used": index > 0,
                "fallback_from": failures[-1]["model"] if failures else None,
                "fallback_reason": failures[-1]["error_type"] if failures else None,
            },
        )

    async def _invoke(
        self,
        method: str,
        *args: Any,
        timeout: float | None = None,
        validator: ResponseValidator | None = None,
        **options: Any,
    ) -> LLMResponse:
        budget = self.timeout if timeout is None else timeout
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("timeout must be finite and positive")
        deadline = time.monotonic() + budget
        chain_id, failures = uuid4().hex, []
        check = validator or self.validator
        for index, provider in enumerate(self.providers):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("LLM fallback deadline exceeded")
            # Keep time for every remaining fallback, even if the entire hub is down.
            per_attempt = remaining / (len(self.providers) - index)
            if index == 0:
                per_attempt = min(self.primary_timeout, per_attempt)
            with self._scope(chain_id, index, failures):
                try:
                    async with asyncio.timeout(per_attempt):
                        result = await getattr(provider, method)(*args, timeout=per_attempt, **options)
                    if result.finish_reason == "length":
                        raise InvalidLLMResponse("Model output was truncated")
                    if not result.raw.strip() and not result.tool_calls:
                        raise InvalidLLMResponse("Model returned empty output")
                    if check:
                        check(result)
                    result.meta.update(
                        fallback_used=index > 0,
                        fallback_index=index,
                        fallback_chain_id=chain_id,
                        fallback_failures=failures,
                    )
                    return result
                except (ProviderError, TimeoutError, InvalidLLMResponse) as exc:
                    failure = {
                        "model": provider.model_name,
                        "error_type": type(exc).__name__,
                        "status_code": getattr(exc, "status_code", None),
                    }
                    failures.append(failure)
                    if isinstance(exc, InvalidLLMResponse):
                        await safe_log(
                            self.tracker,
                            UsageEvent(
                                kind="validation",
                                operation=method,
                                model=provider.model_name,
                                status="error",
                                error_type=type(exc).__name__,
                                cost_status="NOT_APPLICABLE",
                            ),
                        )
                    logger.warning(
                        "LLM attempt failed: model=%s type=%s next=%s",
                        provider.model_name,
                        type(exc).__name__,
                        index + 1 < len(self.providers),
                    )
        raise FallbackExhaustedError(failures)

    async def ainvoke(self, prompt: str, **options: Any) -> LLMResponse:
        return await self._invoke("ainvoke", prompt, **options)

    async def achat(self, messages: Sequence[dict[str, Any]], **options: Any) -> LLMResponse:
        return await self._invoke("achat", messages, **options)

    async def ainvoke_with_image_b64(
        self, prompt: str, image_b64: str, *, mime_type: str = "image/png", **options: Any
    ) -> LLMResponse:
        return await self._invoke("ainvoke_with_image_b64", prompt, image_b64, mime_type=mime_type, **options)

    async def ainvoke_with_image(
        self, prompt: str, image: bytes, *, mime_type: str = "image/png", **options: Any
    ) -> LLMResponse:
        import base64

        return await self.ainvoke_with_image_b64(
            prompt, base64.b64encode(image).decode("ascii"), mime_type=mime_type, **options
        )

    async def astream(
        self, messages: Sequence[dict[str, Any]], *, timeout: float | None = None, **options: Any
    ) -> AsyncIterator[StreamChunk]:
        budget = self.timeout if timeout is None else timeout
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("timeout must be finite and positive")
        deadline = time.monotonic() + budget
        chain_id, failures = uuid4().hex, []
        for index, provider in enumerate(self.providers):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("LLM fallback deadline exceeded")
            per_attempt = remaining / (len(self.providers) - index)
            if index == 0:
                per_attempt = min(self.primary_timeout, per_attempt)
            emitted = False
            with self._scope(chain_id, index, failures):
                try:
                    async with (
                        asyncio.timeout(per_attempt),
                        aclosing(provider.astream(messages, timeout=per_attempt, **options)) as stream,
                    ):
                        async for chunk in stream:
                            if chunk.text or chunk.tool_calls or chunk.finish_reason or chunk.usage:
                                emitted = True
                                yield chunk
                    return
                except (ProviderError, TimeoutError) as exc:
                    if emitted:
                        raise  # replay would duplicate a partial answer
                    failures.append({"model": provider.model_name, "error_type": type(exc).__name__})
        raise FallbackExhaustedError(failures)

    async def aembed(self, texts: str | list[str], **options: Any) -> list[list[float]]:
        # Mixing embedding models silently corrupts vector-space compatibility.
        return await self.providers[0].aembed(texts, **options)

    async def list_models(self) -> list[dict[str, Any]]:
        return await self.providers[0].list_models()

    async def aclose(self) -> None:
        results = await asyncio.gather(*(p.aclose() for p in self.providers), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def __aenter__(self) -> FallbackLLM:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


async def call_with_fallback(
    llm: Any,
    call_fn: Callable[[Any], Awaitable[T]],
    fallback_llm: Any = None,
    label: str = "llm_call",
    *,
    no_fallback_for: tuple[type[Exception], ...] = (),
) -> T:
    """Migration helper for domain validators, preserving the organizer's exception types.

    Use no_fallback_for=(NotATableError,) for its existing image parser.
    Cancellation (BaseException) never starts a fallback.
    """
    try:
        return await call_fn(llm)
    except Exception as exc:
        if fallback_llm is None or isinstance(exc, (DoNotFallbackError, *no_fallback_for)):
            raise
        logger.warning("%s primary failed (%s); trying fallback", label, type(exc).__name__)
        return await call_fn(fallback_llm)
