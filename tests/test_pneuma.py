"""Pneuma wire-contract tests; no real server, speech model or network involved."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest

from botkit.extraction import ContentExtractionService, ExtractionConfig, ExtractionLimitError
from botkit.llm import PneumaConfig, PneumaTranscriber, ProviderError, load_transcriber
from botkit.usage import SQLiteUsageTracker, usage_context

JOB = "a" * 32
TEXT = "[00:00:00.000 - 00:00:02.000] Speaker 1: Проверка распознавания."
CONFIG = PneumaConfig("https://asr.test/prefix", poll_interval=0.001, trust_env=False)


def ready():
    return {"status": "completed", "ready": True, "transcript": TEXT}


async def transcribe(asr, **kwargs):
    return await asr.atranscribe(b"audio bytes", filename="voice.mp3", mime_type="audio/mpeg", **kwargs)


async def test_direct_contract_multipart_poll_result_and_private_telemetry(tmp_path):
    requests = []
    tracker = SQLiteUsageTracker(tmp_path / "usage.db", log_text=True)

    def server(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-secret"
        assert request.url.host == "asr.test"
        if request.method == "POST":
            assert request.url.path == "/prefix/transcriptions"
            assert b'name="audio"; filename="audio.mp3"' in request.content
            assert b"audio bytes" in request.content and b"voice.mp3" not in request.content
            assert b'name="file"' not in request.content and b'name="model"' not in request.content
            return httpx.Response(
                200,
                json={
                    "job_id": JOB,
                    "status": "queued",
                    "audio_duration_sec": 2,
                    "status_url": "https://untrusted.test/steal",
                    "result_url": "/different",
                },
            )
        if request.url.path.endswith("/result"):
            return httpx.Response(200, json={**ready(), "transcript_path": "/private/server/path"})
        return httpx.Response(
            200, json={"job_id": JOB, "status": "running" if len(requests) == 2 else "completed"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        asr = PneumaTranscriber(replace(CONFIG, api_key="test-secret"), client=client, tracker=tracker)
        with usage_context(user_id="u", chat_id="c", bot_id="b", platform="telegram", skill_context="asr"):
            result = await transcribe(asr)
        await asr.aclose()
        assert not client.is_closed  # caller-owned client
    assert result.raw == result.response == TEXT
    assert result.provider == "pneuma" and not result.usage_known
    assert result.meta["poll_count"] == 2 and result.meta["audio_duration_seconds"] == 2
    events = await tracker.events()
    assert len(events) == 1
    event = events[0]
    assert (event["user_id"], event["chat_id"], event["skill_context"]) == ("u", "c", "asr")
    assert event["operation"] == "transcription" and event["poll_count"] == 2
    assert event["provider_request_id"] == JOB and event["cost"] is None
    assert event["cost_status"] == "UNKNOWN_USAGE" and event["input_tokens"] == 0
    saved = json.dumps(events, ensure_ascii=False)
    for private in ("test-secret", TEXT, "/private/server/path", "audio bytes"):
        assert private not in saved and private not in repr(result.meta)


async def test_proxy_concurrent_uploads_have_stable_session_and_task_ids():
    owners, seen = {}, []

    async def server(request):
        assert request.url.path.startswith("/api/transcriptions")
        cookie = request.headers.get("cookie", "")
        assert cookie.startswith("pneuma_session=web-")
        seen.append(cookie)
        if request.method == "POST":
            task = f"{len(owners) + 1:032x}"
            owners[task] = cookie
            await asyncio.sleep(0)  # overlap first uploads
            return httpx.Response(200, json={"task_id": task, "status": "queued", "audio_duration_min": 0.5})
        task = request.url.path.split("/")[3]
        assert owners[task] == cookie
        return httpx.Response(
            200, json=ready() if request.url.path.endswith("/result") else {"status": "completed"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        asr = PneumaTranscriber(
            replace(CONFIG, base_url="https://asr.test/api", api_mode="proxy"), client=client
        )
        results = await asyncio.gather(transcribe(asr), transcribe(asr))
    assert len(set(seen)) == 1 and len({r.request_id for r in results}) == 2
    assert all(r.meta["audio_duration_seconds"] == 30 for r in results)


@pytest.mark.parametrize("status", [302, 401, 403, 404, 422, 500, 503])
async def test_http_failures_do_not_retry_upload_follow_redirect_or_expose_body(status):
    calls = []

    def server(request):
        calls.append(request)
        return httpx.Response(
            status, text="secret-key /server/private", headers={"Location": "https://evil.test"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server), follow_redirects=True) as client:
        asr = PneumaTranscriber(CONFIG, client=client)
        with pytest.raises(ProviderError) as error:
            await transcribe(asr)
    assert error.value.status_code == status and len(calls) == 1
    assert "secret" not in str(error.value) and "/server" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>Login</html>"),
        httpx.Response(200, json=[]),
        httpx.Response(200, json={"job_id": "../evil", "status": "queued"}),
        httpx.Response(200, json={"task_id": JOB, "status": "queued"}),
        httpx.Response(200, json={"job_id": JOB, "status": "unexpected"}),
        httpx.Response(200, json={"job_id": JOB, "status": ["running"]}),
    ],
)
async def test_invalid_submission_fails_safely(response):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: response)) as client:
        with pytest.raises(ProviderError):
            await transcribe(PneumaTranscriber(CONFIG, client=client))


@pytest.mark.parametrize(
    "body",
    [
        {"status": "failed", "error": "private server error"},
        {"status": "cancelled"},
        {"status": "completed", "job_id": "b" * 32},
        {"status": "alien"},
    ],
)
async def test_terminal_or_wrong_job_status(body):
    def server(request):
        return httpx.Response(
            200, json={"job_id": JOB, "status": "queued"} if request.method == "POST" else body
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        with pytest.raises(ProviderError) as error:
            await transcribe(PneumaTranscriber(CONFIG, client=client))
    assert "private" not in str(error.value)


@pytest.mark.parametrize(
    "result",
    [
        {"status": "completed", "ready": False, "transcript": TEXT},
        {"status": "completed", "ready": True, "transcript": None},
        {"status": "failed", "ready": False},
    ],
)
async def test_invalid_result_is_not_success(result):
    def server(request):
        return httpx.Response(
            200,
            json=result if request.url.path.endswith("/result") else {"job_id": JOB, "status": "completed"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        with pytest.raises(ProviderError):
            await transcribe(PneumaTranscriber(CONFIG, client=client))


@pytest.mark.parametrize(
    "cancel,mode,expected_cancel", [(True, "direct", True), (False, "direct", False), (True, "proxy", False)]
)
async def test_deadline_and_only_direct_best_effort_cancel(cancel, mode, expected_cancel):
    paths = []
    tracker = AsyncMock()

    def server(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/cancel"):
            return httpx.Response(500)  # cancellation failure must not replace timeout
        key = "job_id" if mode == "direct" else "task_id"
        return httpx.Response(200, json={key: JOB, "status": "queued"})

    config = replace(CONFIG, timeout=0.03, poll_interval=1, cancel_on_timeout=cancel, api_mode=mode)
    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        with pytest.raises(TimeoutError):
            await transcribe(PneumaTranscriber(config, client=client, tracker=tracker))
    assert any(p.endswith("/cancel") for p in paths) == expected_cancel
    event = tracker.log.await_args.args[0]
    assert event.status == "error" and event.error_type == "TimeoutError"
    assert not any("cleanup" in p for p in paths)


async def test_caller_cancellation_propagates_and_cancels_owned_job():
    uploaded, paths = asyncio.Event(), []

    def server(request):
        paths.append(request.url.path)
        uploaded.set()
        return httpx.Response(200, json={"job_id": JOB, "status": "queued"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        asr = PneumaTranscriber(replace(CONFIG, poll_interval=60), client=client)
        task = asyncio.create_task(transcribe(asr))
        await uploaded.wait()
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert paths == ["/prefix/transcriptions", f"/prefix/transcriptions/{JOB}/cancel"]


async def test_timeout_during_upload_is_not_retried_or_cancelled_without_job_id():
    calls = []

    def server(request):
        calls.append(request)
        raise httpx.ReadTimeout("secret-key", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        with pytest.raises(TimeoutError) as error:
            await transcribe(PneumaTranscriber(CONFIG, client=client))
    assert len(calls) == 1 and "secret" not in str(error.value)


async def test_response_and_input_limits_before_further_requests():
    calls = []

    def server(request):
        calls.append(request)
        return httpx.Response(200, content=b"x" * 32)

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        asr = PneumaTranscriber(replace(CONFIG, max_bytes=11, max_response_bytes=10), client=client)
        for data in (b"", b"x" * 12):
            with pytest.raises(ValueError):
                await asr.atranscribe(data, mime_type="audio/mpeg", filename="a.mp3")
        with pytest.raises(ValueError, match="language"):
            await transcribe(asr, language="ru")
        assert not calls
        with pytest.raises(ProviderError, match="size limit"):
            await transcribe(asr)
        assert len(calls) == 1


@pytest.mark.parametrize(
    "mime,filename,data,expected",
    [
        ("application/octet-stream", "voice.MP3", b"mp3", "audio/mpeg"),
        ("application/octet-stream", "voice.opus", b"voice", "audio/ogg"),
        ("audio/ogg; codecs=opus", None, b"voice", "audio/ogg"),
        ("audio/mp3", None, b"mp3", "audio/mpeg"),
        ("application/ogg", None, b"voice", "audio/ogg"),
        ("application/octet-stream", None, b"OggSvoice", "audio/ogg"),
        ("application/octet-stream", None, b"ID3music", "audio/mpeg"),
        ("application/octet-stream", None, b"RIFF0000WAVEdata", "audio/wav"),
        ("application/octet-stream", "file.m4a", b"data", "audio/mp4"),
        ("application/octet-stream", None, b"fLaCaudio", "audio/flac"),
    ],
)
async def test_a5_voice_and_file_mime_routing_without_vision(mime, filename, data, expected):
    def server(request):
        if request.url.path.endswith("/result"):
            return httpx.Response(200, json=ready())
        assert f"Content-Type: {expected}".encode() in request.content
        assert data in request.content
        return httpx.Response(200, json={"job_id": JOB, "status": "completed"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        asr = PneumaTranscriber(CONFIG, client=client)
        service = ContentExtractionService(transcriber=asr)
        result = await service.extract_bytes(data, mime, filename=filename)
    assert result.kind == "transcript" and result.text == TEXT
    assert result.metadata["job_id"] == JOB and result.metadata["mime_type"] == expected
    assert result.source_confidence == "low" and "audio_transcript_unverified" in result.warnings


async def test_empty_transcript_warning_and_text_limit():
    transcript = ""

    def server(request):
        return httpx.Response(
            200,
            json={**ready(), "transcript": transcript}
            if request.url.path.endswith("/result")
            else {"job_id": JOB, "status": "completed"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        service = ContentExtractionService(
            transcriber=PneumaTranscriber(CONFIG, client=client), config=ExtractionConfig(max_text_chars=2)
        )
        result = await service.extract_bytes(b"audio", "audio/mpeg")
        assert "empty_extraction" in result.warnings
        transcript = "too long"
        with pytest.raises(ExtractionLimitError):
            await service.extract_bytes(b"audio", "audio/mpeg")


async def test_env_factory_disabled_enabled_and_close(monkeypatch):
    monkeypatch.delenv("ASR_PROVIDER", raising=False)
    assert load_transcriber() is None
    monkeypatch.setenv("ASR_PROVIDER", "pneuma")
    monkeypatch.setenv("PNEUMA_BASE_URL", "https://test.example/mount/api")
    monkeypatch.setenv("PNEUMA_API_MODE", "proxy")
    monkeypatch.setenv("PNEUMA_API_KEY", "private-key")
    monkeypatch.setenv("PNEUMA_POLL_INTERVAL", "3")
    monkeypatch.setenv("EXTRACTION_AUDIO_TIMEOUT", "400")
    async with load_transcriber() as asr:
        assert asr.config.base_url.endswith("/mount/api") and asr.config.api_mode == "proxy"
        assert asr.config.poll_interval == 3 and "private-key" not in repr(asr.config)
        assert ExtractionConfig.from_env().audio_timeout == 400
    assert asr.client.is_closed
    monkeypatch.setenv("ASR_PROVIDER", "typo")
    with pytest.raises(ValueError, match="ASR_PROVIDER"):
        load_transcriber()


@pytest.mark.parametrize(
    "options",
    [
        {"base_url": ""},
        {"base_url": "https://key:secret@test"},
        {"base_url": "https://test?key=secret"},
        {"api_mode": "typo"},
        {"timeout": 0},
        {"poll_interval": float("nan")},
        {"request_timeout": -1},
        {"max_bytes": 0},
        {"api_key": "secret\r\nextra"},
    ],
)
def test_configuration_validation(options):
    with pytest.raises(ValueError):
        replace(CONFIG, **options)
