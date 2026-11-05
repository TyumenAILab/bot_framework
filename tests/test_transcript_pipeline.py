"""Offline ASR -> LLM -> A5 tests. No real keys, speech models or network."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from test_pneuma_upload import API, config, transcribe

from botkit.extraction.audio import AudioExtractor
from botkit.llm import (
    ChunkedPneumaTranscriber,
    InvalidLLMResponse,
    LLMConfig,
    LLMGateway,
    LLMResponse,
    LLMTranscriptProcessor,
    PneumaTranscriber,
    TranscriptionPipeline,
    TranscriptResult,
    load_transcriber,
    load_transcript_processor,
    validate_transcript,
)
from botkit.usage import SQLiteUsageTracker, usage_context


class Model:
    model_name = "test-local-qwen"

    def __init__(self, text):
        self.text = text
        self.calls = []
        self.aclose = AsyncMock()

    async def achat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if isinstance(self.text, Exception):
            raise self.text
        # Deliberately ignore validator to check the processor's own validation.
        return LLMResponse(self.text, self.text, "raw_text_fallback")


@pytest.mark.parametrize(
    "candidate",
    [
        "Рост 5%.",
        "Рост -50%.",
        "Рост -5%, вот готовый ответ.",
        "",
    ],
)
def test_lexical_guard(candidate):
    if candidate:
        result = validate_transcript(candidate, "рост -5%")
        assert not result.processed and result.text == "рост -5%"
    else:
        with pytest.raises(InvalidLLMResponse):
            validate_transcript(candidate, "рост -5%")


async def test_default_processor_and_diarization_headers():
    model = Model('{"transcript":"Я не согласен. Рост: -5%."}')
    processor = LLMTranscriptProcessor(model)
    result = await processor.aprocess("[00:00:00.000 - 00:00:01.000] Speaker 1: я не согласен рост -5%")
    assert result == TranscriptResult("Я не согласен. Рост: -5%.", True)
    messages, options = model.calls[0]
    assert json.loads(messages[1]["content"]) == {"text": "я не согласен рост -5%"}
    assert options["temperature"] == 0 and options["timeout"] == 30
    await processor.aclose()
    model.aclose.assert_not_awaited()
    assert (await processor.aprocess(" ")).reason == "empty"
    assert len(model.calls) == 1


@pytest.mark.parametrize("raw", ["not JSON", "[]", '{"transcript":null}', '{"transcript":"речь","extra":1}'])
async def test_invalid_json_fail_closed_or_explicit_original(raw):
    model = Model(raw)
    with pytest.raises(InvalidLLMResponse):
        await LLMTranscriptProcessor(model).aprocess("речь")
    result = await LLMTranscriptProcessor(model, on_error="original").aprocess("речь")
    assert result == TranscriptResult("речь", False, "llm_error")


async def test_custom_joint_classifier_request_cannot_override_text():
    model = Model('{"intent":"allowed","transcript":"Речь."}')
    processor = LLMTranscriptProcessor(model)

    def validator(response):
        return json.loads(response.raw)

    result = await processor.arequest(
        "речь",
        system_prompt="Application-specific intent rules",
        validator=validator,
        context={"text": "must not override", "audio": True},
        max_tokens=256,
    )
    assert json.loads(result.raw)["intent"] == "allowed"
    assert json.loads(model.calls[0][0][1]["content"])["text"] == "речь"
    assert model.calls[0][1]["max_tokens"] == 256


async def test_limits_deadline_and_cancellation():
    model = Model('{"transcript":"Речь."}')
    with pytest.raises(ValueError, match="limit"):
        await LLMTranscriptProcessor(model, max_chars=2).aprocess("речь")
    assert not model.calls

    class Slow(Model):
        async def achat(self, *args, **kwargs):
            await asyncio.sleep(30)

    processor = LLMTranscriptProcessor(Slow(""), timeout=0.01)
    with pytest.raises(TimeoutError):
        await processor.aprocess("речь")
    processor = LLMTranscriptProcessor(Slow(""), on_error="original")
    task = asyncio.create_task(processor.aprocess("речь"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_full_http_pipeline_a5_processed_text_and_private_usage(tmp_path):
    api = API()
    events = SQLiteUsageTracker(tmp_path / "usage.db", log_text=False)
    seen = []

    def server(request):
        if request.url.host == "local.test":
            payload = json.loads(request.content)
            seen.append(payload)
            assert not payload["stream"] and payload["temperature"] == 0
            assert json.loads(payload["messages"][1]["content"])["text"] == api.result["transcript"]
            return httpx.Response(
                200,
                json={
                    "model": "local-qwen",
                    "choices": [{"message": {"content": '{"transcript":"Мой ответ не изменился."}'}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 6},
                },
            )
        return api(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        asr = ChunkedPneumaTranscriber(config(), client=client, tracker=events)
        llm = LLMGateway(
            LLMConfig("vllm", "local-qwen", "https://local.test/v1"), client=client, tracker=events
        )
        pipeline = TranscriptionPipeline(asr, LLMTranscriptProcessor(llm, tracker=events))
        with usage_context(user_id="stu_private", request_id="synthetic-case"):
            result = await transcribe(pipeline)
            assert result.raw == "мой ответ не изменился"
            assert result.response == "Мой ответ не изменился."
            assert result.meta["postprocessed"] is True and result.meta["job_id"] == "job-1"
            extracted = await AudioExtractor(pipeline).extract(b"0123456789", "audio/ogg")
        assert extracted.text == result.response
        assert "audio_transcript_postprocessed" in extracted.warnings
        assert "audio_transcript_unverified" in extracted.warnings
        assert extracted.metadata["postprocessor_model"] == "local-qwen"
        assert len([call for call in api.calls if call[:2] == ("POST", "/uploads")]) == 1
        assert len(seen) == 2  # ASR job reused, LLM processing is explicit on each call.
    rows = await events.events()
    assert {row["operation"] for row in rows} == {"transcription", "chat", "transcript_postprocess"}
    assert "мой ответ" not in json.dumps(rows, ensure_ascii=False)
    assert "secret" not in json.dumps(rows, ensure_ascii=False)
    assert all(row["user_id"] == "stu_private" for row in rows)


async def test_asr_failure_never_calls_llm_and_total_deadline():
    asr = AsyncMock()
    asr.atranscribe.side_effect = RuntimeError("ASR failed")
    model = Model('{"transcript":"Речь."}')
    pipeline = TranscriptionPipeline(asr, LLMTranscriptProcessor(model))
    with pytest.raises(RuntimeError, match="ASR failed"):
        await transcribe(pipeline)
    assert not model.calls

    async def slow_asr(*args, **kwargs):
        await asyncio.sleep(0.015)
        return LLMResponse("речь", "речь", "raw_text_fallback")

    class SlowProcessor:
        async def aprocess(self, text):
            await asyncio.sleep(30)

    asr.atranscribe.side_effect = slow_asr
    pipeline = TranscriptionPipeline(asr, SlowProcessor())
    with pytest.raises(TimeoutError):
        await pipeline.atranscribe(b"a", mime_type="audio/ogg", filename="a.ogg", timeout=0.03)


async def test_cleanup_ownership_and_close_even_if_processor_fails():
    asr, processor = AsyncMock(), AsyncMock()
    async with TranscriptionPipeline(asr, processor):
        pass
    asr.aclose.assert_not_awaited()
    processor.aclose.side_effect = ValueError("close")
    with pytest.raises(ValueError, match="close"):
        async with TranscriptionPipeline(asr, processor, owns_components=True):
            pass
    asr.aclose.assert_awaited_once()


@pytest.fixture
def asr_env(monkeypatch):
    for name, value in {
        "ASR_PROVIDER": "pneuma",
        "PNEUMA_BASE_URL": "https://pneuma.test/api",
        "ASR_POSTPROCESS_ENABLED": "false",
        "PNEUMA_UPLOAD_PROTOCOL": "legacy",
        "LOCAL_HUB_API_BASE": "https://local.test/v1",
        "LOCAL_HUB_MODEL_NAME": "local-qwen",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("PNEUMA_UPLOAD_STATE_DB", raising=False)


async def test_factory_defaults_chunked_and_optional_pipeline(asr_env, monkeypatch, tmp_path):
    async with load_transcriber() as legacy:
        assert isinstance(legacy, PneumaTranscriber)
    monkeypatch.setenv("PNEUMA_UPLOAD_PROTOCOL", "chunked")
    monkeypatch.setenv("PNEUMA_UPLOAD_STATE_DB", str(tmp_path / "metadata.db"))
    async with load_transcriber() as chunked:
        assert isinstance(chunked, ChunkedPneumaTranscriber)
        assert chunked.store.path == tmp_path / "metadata.db"
    monkeypatch.setenv("ASR_POSTPROCESS_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "302ai")
    monkeypatch.setenv("LLM_DIRECT_FALLBACK", "true")
    monkeypatch.setenv("LOCAL_HUB_FALLBACK_MODEL_NAME", "cloud-model")
    async with load_transcriber() as pipeline:
        assert isinstance(pipeline, TranscriptionPipeline)
        llm = pipeline.processor.llm
        assert isinstance(llm, LLMGateway)  # not FallbackLLM, even with global fallback enabled
        assert llm.model_name == "local-qwen" and not llm.config.supports_streaming
        assert llm.config.max_retries == 0
    assert llm.client.is_closed and pipeline.transcriber.client.is_closed


def test_disabled_never_reads_llm_or_upload_config(asr_env, monkeypatch):
    monkeypatch.setenv("ASR_PROVIDER", "disabled")
    monkeypatch.setenv("ASR_POSTPROCESS_ENABLED", "true")
    monkeypatch.setenv("PNEUMA_BASE_URL", "bad URL")
    monkeypatch.setenv("ASR_POSTPROCESS_LLM_PROVIDER", "302ai")
    assert load_transcriber() is None


@pytest.mark.parametrize(
    "name,value",
    [
        ("PNEUMA_UPLOAD_PROTOCOL", "automatic"),
        ("ASR_POSTPROCESS_LLM_PROVIDER", "302ai"),
        ("ASR_POSTPROCESS_TIMEOUT", "nan"),
        ("ASR_POSTPROCESS_ON_ERROR", "silently_allow"),
    ],
)
def test_invalid_factory_settings_fail_explicitly(asr_env, monkeypatch, name, value):
    monkeypatch.setenv("ASR_POSTPROCESS_ENABLED", "true")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        load_transcriber()


async def test_openai_compatible_provider_reuses_framework_configuration(asr_env, monkeypatch):
    monkeypatch.setenv("ASR_POSTPROCESS_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("VLLM_BASE_URL", "https://internal.test/ai/v1")
    monkeypatch.setenv("VLLM_MODEL", "local-model")
    monkeypatch.setenv("ASR_POSTPROCESS_MODEL", "override-model")
    processor = load_transcript_processor()
    try:
        assert processor.llm.config.base_url == "https://internal.test/ai/v1"
        assert processor.llm.model_name == "override-model"
    finally:
        await processor.aclose()


async def test_a5_warns_if_model_rewrites_words():
    asr = AsyncMock()
    asr.atranscribe.return_value = LLMResponse("не согласен", "не согласен", "raw_text_fallback")
    pipeline = TranscriptionPipeline(asr, LLMTranscriptProcessor(Model('{"transcript":"Согласен."}')))
    content = await AudioExtractor(pipeline).extract(b"audio", "audio/mpeg")
    assert content.text == "не согласен"
    assert content.metadata["postprocess_reason"] == "content_changed"
    assert "audio_transcript_postprocess_skipped" in content.warnings
