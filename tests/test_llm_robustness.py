"""Malformed upstream replies must be bounded provider failures, not code errors."""

import asyncio
from contextlib import aclosing
from dataclasses import replace

import httpx
import pytest

from botkit.llm import FallbackLLM, LLMConfig, LLMGateway, ProviderError, StreamChunk

CONFIG = LLMConfig("hub", "local-test", "https://hub.test/v1", "synthetic-key", max_retries=0)


async def test_network_failure_exposes_only_safe_category():
    def network(request):
        raise httpx.ConnectError("SECRET_URL_AND_TOKEN", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(network)) as client:
        gateway = LLMGateway(CONFIG, client=client)
        with pytest.raises(ProviderError) as caught:
            await gateway.ainvoke("test")
    assert caught.value.reason == "network_connect"
    assert caught.value.status_code is None
    assert "SECRET_URL_AND_TOKEN" not in str(caught.value)


def completion(**extra):
    return {"choices": [{"message": {"content": "Synthetic reply."}}], **extra}


@pytest.mark.parametrize(
    "body",
    [
        {"choices": [None]},
        {"choices": ["bad"]},
        {"choices": [{"message": []}]},
        {"choices": [{"message": "bad"}]},
        {"choices": [{"message": {"content": ["bad"]}}]},
        {"choices": [{"message": {"content": [{"type": "text", "text": 123}]}}]},
        {"choices": [{"message": {"content": "text", "tool_calls": {"bad": True}}}]},
        {"choices": [{"message": {"content": "text", "tool_calls": [None]}}]},
        {"choices": [{"message": {"refusal": {"private": "diagnostic"}}}]},
        {"choices": [{"message": {"content": "text"}, "finish_reason": []}]},
    ],
)
async def test_malformed_choices_raise_provider_error(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as client:
        with pytest.raises(ProviderError):
            await LLMGateway(CONFIG, client=client).ainvoke("Synthetic prompt.")


@pytest.mark.parametrize(
    "usage",
    [
        1,
        "bad",
        [],
        {"prompt_tokens": "5", "completion_tokens": 3},
        {"prompt_tokens": True, "completion_tokens": 3},
        {"prompt_tokens": -1, "completion_tokens": 3},
    ],
)
async def test_malformed_optional_usage_keeps_valid_reply_without_false_accounting(usage):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=completion(usage=usage)))
    ) as client:
        result = await LLMGateway(CONFIG, client=client).ainvoke("Synthetic prompt.")
    assert result.response == "Synthetic reply."
    assert result.usage_known is False
    assert type(result.input_tokens) is int and result.input_tokens == 0


@pytest.mark.parametrize("details", [1, "bad", [], {"cached_tokens": "5"}, {"cached_tokens": True}])
async def test_invalid_optional_token_details_do_not_break_valid_response(details):
    usage = {
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "prompt_tokens_details": details,
        "completion_tokens_details": details,
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=completion(usage=usage)))
    ) as client:
        result = await LLMGateway(CONFIG, client=client).ainvoke("Synthetic prompt.")
    assert result.response == "Synthetic reply."
    assert result.cached_tokens == result.reasoning_tokens == 0


async def test_malformed_primary_response_activates_fallback():
    calls = []

    def server(request):
        calls.append(request)
        return httpx.Response(200, json={"choices": [None]} if len(calls) == 1 else completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        result = await FallbackLLM(
            LLMGateway(CONFIG, client=client), [LLMGateway(replace(CONFIG, model="backup"), client=client)]
        ).ainvoke("Synthetic prompt.")
    assert result.meta["fallback_used"] is True and len(calls) == 2


@pytest.mark.parametrize("streaming", [False, True])
async def test_injected_redirect_client_cannot_change_api_destination(streaming):
    calls = []

    def server(request):
        calls.append(request)
        return httpx.Response(307, headers={"Location": "https://other.test/collect"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(server), follow_redirects=True) as client:
        gateway = LLMGateway(CONFIG, client=client)
        with pytest.raises(ProviderError) as error:
            if streaming:
                async with aclosing(
                    gateway.astream([{"role": "user", "content": "Synthetic prompt."}])
                ) as stream:
                    _ = [chunk async for chunk in stream]
            else:
                await gateway.ainvoke("Synthetic prompt.")
    assert error.value.status_code == 307 and len(calls) == 1


@pytest.mark.parametrize(
    "body", ["not-json", "[]", '{"choices":[null]}', '{"choices":[{"delta":{"content":12}}]}']
)
async def test_malformed_sse_is_a_provider_error(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, text="data: " + body + "\n\n"))
    ) as client:
        gateway = LLMGateway(CONFIG, client=client)
        with pytest.raises(ProviderError):
            async with aclosing(
                gateway.astream([{"role": "user", "content": "Synthetic prompt."}])
            ) as stream:
                _ = [chunk async for chunk in stream]


@pytest.mark.parametrize("field", ["timeout", "poll_interval", "request_interval"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_config_deadlines_are_rejected(field, value):
    with pytest.raises(ValueError):
        replace(CONFIG, **{field: value})


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
async def test_invalid_per_call_timeout_is_rejected_without_network(timeout):
    calls = []

    def server(request):
        calls.append(request)
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        with pytest.raises(ValueError):
            await LLMGateway(CONFIG, client=client).ainvoke("Synthetic prompt.", timeout=timeout)
    assert calls == []


async def test_caller_cancellation_is_not_retyped_or_retried():
    entered = asyncio.Event()

    async def server(request):
        entered.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        task = asyncio.create_task(LLMGateway(CONFIG, client=client).ainvoke("Synthetic prompt."))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_stream_fallback_bounds_custom_provider_that_ignores_timeout_argument():
    class Hanging:
        model_name = "synthetic-primary"

        async def astream(self, *args, **kwargs):
            await asyncio.Event().wait()
            yield StreamChunk("unreachable")

    class Backup:
        model_name = "synthetic-backup"

        async def astream(self, *args, **kwargs):
            yield StreamChunk("backup")

    chain = FallbackLLM(Hanging(), [Backup()], timeout=0.5, primary_timeout=0.01)
    async with asyncio.timeout(1), aclosing(chain.astream([])) as stream:
        assert [chunk.text async for chunk in stream] == ["backup"]
