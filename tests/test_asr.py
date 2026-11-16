import asyncio
import sys
import types

import httpx
import pytest

from botkit.asr import (
    ASRError,
    MoonshineError,
    MoonshineTranscriber,
    audio_filename,
    available_asr,
    load_asr,
    register_asr,
    transcribe_attachment,
)
from botkit.llm import LLMResponse, load_transcriber
from botkit.transport.base import Attachment
from botkit.usage import usage_context


def transcriber(handler, **kwargs):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return MoonshineTranscriber("http://asr.internal:8181/", client=client, **kwargs)


async def test_sends_multipart_file_and_returns_the_common_contract():
    seen = {}

    def handler(request):
        seen["url"], seen["body"] = str(request.url), request.read()
        return httpx.Response(200, json={"text": " Ever tried. Ever failed. ", "duration": 9.96, "segments": [{}]})

    asr = transcriber(handler)
    result = await asr.atranscribe(b"OGGDATA", mime_type="audio/ogg", filename="voice.ogg")
    assert seen["url"] == "http://asr.internal:8181/v1/audio/transcriptions"
    assert b'name="file"' in seen["body"] and b"OGGDATA" in seen["body"] and b"audio/ogg" in seen["body"]
    assert isinstance(result, LLMResponse)
    assert result.response == result.raw == "Ever tried. Ever failed."
    assert result.provider == "moonshine" and result.meta["audio_duration_seconds"] == 9.96
    assert result.meta["segment_count"] == 1


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(503, text="private details"), "HTTP 503"),
        (httpx.Response(200, text="not json"), "unexpected response format"),
        (httpx.Response(200, json={"words": []}), "unexpected response format"),
        (httpx.Response(200, json={"text": 5}), "unexpected response format"),
    ],
)
async def test_failures_raise_a_safe_error(response, reason):
    with pytest.raises(MoonshineError, match=reason) as info:
        await transcriber(lambda request: response).atranscribe(b"audio", mime_type="audio/ogg", filename="a.ogg")
    assert "private details" not in str(info.value) and isinstance(info.value, ASRError)


async def test_network_error_is_wrapped_without_the_url():
    def boom(request):
        raise httpx.ConnectError("connect to http://secret-host failed")

    with pytest.raises(MoonshineError, match="ConnectError") as info:
        await transcriber(boom).atranscribe(b"audio", mime_type="audio/ogg", filename="a.ogg")
    assert "secret-host" not in str(info.value)


async def test_health_and_api_key_header():
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"status": "ok", "model": "SMALL"})

    asr = transcriber(handler, api_key="k")
    assert (await asr.health())["model"] == "SMALL" and seen["auth"] == "Bearer k"
    with pytest.raises(MoonshineError, match="health check failed"):
        await transcriber(lambda r: httpx.Response(500)).health()


@pytest.mark.parametrize(
    "kwargs",
    [{"data": b""}, {"language": "ru"}, {"timeout": float("nan")}, {"timeout": 0}],
)
async def test_invalid_requests_are_rejected_before_any_network_call(kwargs):
    calls = []
    asr = transcriber(lambda r: calls.append(r) or httpx.Response(200, json={"text": "x"}))
    arguments = {"data": b"a", "mime_type": "audio/ogg", "filename": "a.ogg", **kwargs}
    with pytest.raises(ValueError):
        await asr.atranscribe(arguments.pop("data"), **arguments)
    assert not calls


@pytest.mark.parametrize("url", ["", "ftp://host", "http://user:pw@host", "not a url"])
def test_url_must_be_plain_http(url):
    with pytest.raises(ValueError, match="MOONSHINE_API_URL"):
        MoonshineTranscriber(url)


async def test_recordings_beyond_the_limit_wait_and_are_told_about_the_queue():
    release = asyncio.Event()
    running = peak = 0

    async def handler(request):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await release.wait()
        running -= 1
        return httpx.Response(200, json={"text": "ok", "duration": 1.0})

    asr = transcriber(handler, max_concurrent=2)
    told = []

    async def queued(ahead):
        told.append(ahead)

    tasks = [
        asyncio.create_task(asr.atranscribe(b"a", mime_type="audio/ogg", filename="a.ogg", on_queued=queued))
        for _ in range(4)
    ]
    await asyncio.sleep(0.05)
    assert peak == 2 and asr.active == 2 and asr.waiting == 2
    assert sorted(told) == [2, 3]
    release.set()
    assert [r.response for r in await asyncio.gather(*tasks)] == ["ok"] * 4
    assert asr.active == 0 and asr.waiting == 0


async def test_slot_is_released_after_failure_and_a_broken_notice_does_not_break_transcription():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(500) if len(calls) == 1 else httpx.Response(200, json={"text": "fine"})

    asr = transcriber(handler, max_concurrent=1)
    with pytest.raises(MoonshineError):
        await asr.atranscribe(b"a", mime_type="audio/ogg", filename="a.ogg")

    async def broken_notice(ahead):
        raise RuntimeError("cannot send")

    result = await asr.atranscribe(b"b", mime_type="audio/ogg", filename="a.ogg", on_queued=broken_notice)
    assert result.response == "fine" and asr.active == 0


async def test_usage_event_has_no_text_or_audio_and_unknown_cost():
    events = []

    class Tracker:
        async def log(self, event):
            events.append(event)

    asr = transcriber(lambda r: httpx.Response(200, json={"text": "private words", "duration": 2}), tracker=Tracker())
    with usage_context(request_id="req-1"):
        await asr.atranscribe(b"secret-audio", mime_type="audio/ogg", filename="a.ogg")
    (event,) = events
    assert event.operation == "transcription" and event.provider == "moonshine" and event.request_id == "req-1"
    assert event.cost is None and event.cost_status == "UNKNOWN_USAGE" and event.http_status == 200
    assert "private" not in repr(event.meta) and event.meta["text_chars"] == 13


# ---- registry / loader -------------------------------------------------------------------------


def test_loader_builds_moonshine_from_env_and_disabled_is_none(monkeypatch):
    monkeypatch.delenv("ASR_PROVIDER", raising=False)
    assert load_asr() is None and load_transcriber() is None
    monkeypatch.setenv("ASR_PROVIDER", "moonshine")
    monkeypatch.setenv("MOONSHINE_API_URL", "http://asr:8181")
    monkeypatch.setenv("MOONSHINE_MAX_CONCURRENT", "3")
    asr = load_transcriber()
    assert isinstance(asr, MoonshineTranscriber) and asr.max_concurrent == 3
    monkeypatch.setenv("MOONSHINE_API_URL", "")
    with pytest.raises(ValueError, match="MOONSHINE_API_URL"):
        load_asr()
    monkeypatch.setenv("MOONSHINE_API_URL", "http://asr:8181")
    monkeypatch.setenv("MOONSHINE_MAX_CONCURRENT", "0")
    with pytest.raises(ValueError, match="MOONSHINE_MAX_CONCURRENT"):
        load_asr()


def test_unknown_provider_names_the_options(monkeypatch):
    with pytest.raises(ValueError, match="ASR_PROVIDER.*moonshine.*pneuma"):
        load_asr("typo")
    with pytest.raises(ValueError, match="module:factory"):
        load_asr(":nothing")


async def test_registered_and_module_factory_providers_are_used_through_the_same_interface(monkeypatch):
    class Fake:
        def __init__(self, tracker=None, **_):
            self.tracker = tracker

        async def atranscribe(self, data, *, mime_type, filename, timeout=None, language=None):
            return LLMResponse("fake text", "fake text", "raw_text_fallback", provider="fake")

        async def aclose(self):
            pass

    import botkit.asr.factory as factory

    monkeypatch.setattr(factory, "_REGISTRY", dict(factory._REGISTRY))  # keep the global registry clean
    register_asr("fake-asr", Fake)
    assert "fake-asr" in available_asr()
    with pytest.raises(ValueError, match="already registered"):
        register_asr("fake-asr", Fake)
    assert isinstance(load_asr("fake-asr", tracker="t"), Fake)

    module = types.ModuleType("my_asr_plugin")
    module.build = lambda tracker=None: Fake(tracker)
    monkeypatch.setitem(sys.modules, "my_asr_plugin", module)
    custom = load_asr("my_asr_plugin:build")
    assert (await custom.atranscribe(b"a", mime_type="audio/ogg", filename="a.ogg")).provider == "fake"


async def test_postprocessing_wraps_any_provider(monkeypatch):
    from botkit.llm import TranscriptionPipeline

    monkeypatch.setenv("ASR_POSTPROCESS_ENABLED", "true")
    monkeypatch.setenv("ASR_POSTPROCESS_LLM_PROVIDER", "hub")
    monkeypatch.setenv("LOCAL_HUB_API_BASE", "https://hub.example/ai/v1")
    monkeypatch.setenv("LOCAL_HUB_API_KEY", "k")
    monkeypatch.setenv("LOCAL_HUB_MODEL_NAME", "m")
    monkeypatch.setenv("MOONSHINE_API_URL", "http://asr:8181")
    pipeline = load_asr("moonshine")
    assert isinstance(pipeline, TranscriptionPipeline) and isinstance(pipeline.transcriber, MoonshineTranscriber)
    await pipeline.aclose()


# ---- helpers ------------------------------------------------------------------------------------


def test_audio_filename_follows_mime():
    assert audio_filename("audio/ogg") == "voice.ogg"
    assert audio_filename("audio/mpeg; codecs=x") == "voice.mp3"
    assert audio_filename("audio/x-wav") == "voice.wav"
    assert audio_filename("audio/unknown") == "voice.ogg"
    assert audio_filename("audio/ogg", "../../etc/x") == "x.ogg"


async def test_transcribe_attachment_downloads_and_strips_speaker_prefixes():
    seen = {}

    class Recognizer:
        async def atranscribe(self, data, *, mime_type, filename, timeout=None, language=None):
            seen.update(data=data, mime=mime_type, name=filename, timeout=timeout)
            return LLMResponse("[00:00:00.000 - 00:00:01.000] Speaker 1: Стоп.", "raw", "raw_text_fallback")

    async def read():
        return b"OggS-bytes"

    attachment = Attachment(kind="audio", mime_type="audio/ogg", reader=read)
    result = await transcribe_attachment(Recognizer(), attachment, timeout=30)
    assert result.response == "Стоп." and seen == {
        "data": b"OggS-bytes", "mime": "audio/ogg", "name": "voice.ogg", "timeout": 30,
    }
    plain = await transcribe_attachment(Recognizer(), attachment, strip_speaker_prefixes=False)
    assert plain.response.startswith("[00:00:00.000")


async def test_transcribe_attachment_rejects_non_audio_and_oversize():
    async def read():
        return b"x" * 10

    with pytest.raises(ASRError):
        await transcribe_attachment(object(), Attachment(kind="photo", mime_type="image/png", reader=read))
    with pytest.raises(ValueError, match="size limit"):
        await transcribe_attachment(
            object(), Attachment(kind="audio", mime_type="audio/ogg", reader=read, max_bytes=5)
        )
