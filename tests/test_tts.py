import asyncio
import base64
import json
import logging
import struct
import sys
import types
from urllib.parse import parse_qs

import httpx
import pytest

import botkit.tts.factory as factory
from botkit.tts import (
    CachedSynthesizer,
    GoogleTTS,
    KokoroError,
    KokoroTTS,
    SynthesizedAudio,
    TTSError,
    YandexTTS,
    available_tts,
    load_tts,
    register_tts,
    send_audio,
    speak,
)
from botkit.tts.audio import to_ogg_opus
from botkit.tts.kokoro import wav_seconds
from botkit.usage import usage_context

# ---- Yandex / Google ----------------------------------------------------------------------------


@pytest.mark.parametrize("audio_format,mime", [("mp3", "audio/mpeg"), ("oggopus", "audio/ogg")])
@pytest.mark.parametrize("iam", [False, True])
async def test_yandex_wire_contract(audio_format, mime, iam):
    def server(request):
        assert str(request.url) == YandexTTS.endpoint
        assert request.headers["Authorization"] == ("Bearer token" if iam else "Api-Key secret")
        data = parse_qs(request.content.decode())
        assert data["text"] == ["В чём ограничение метода?"]
        assert data["format"] == [audio_format] and data["voice"] == ["alena"] and data["lang"] == ["ru-RU"]
        assert data.get("folderId") == (["folder"] if iam else None)
        return httpx.Response(200, content=b"audio-bytes", headers={"Content-Type": mime})

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        provider = YandexTTS(
            client=client,
            audio_format=audio_format,
            iam_token="token" if iam else "",
            api_key="" if iam else "secret",
            folder_id="folder" if iam else "",
        )
        # a per-call voice overrides the configured one
        assert await provider.synthesize("В чём ограничение метода?", voice="alena") == SynthesizedAudio(
            b"audio-bytes", mime
        )
        await provider.aclose()
        assert not client.is_closed


@pytest.mark.parametrize("audio_format,encoding,mime", [("mp3", "MP3", "audio/mpeg"), ("oggopus", "OGG_OPUS", "audio/ogg")])
async def test_google_wire_contract(audio_format, encoding, mime):
    def server(request):
        assert str(request.url) == GoogleTTS.endpoint and "key" not in request.url.params
        assert request.headers["Authorization"] == "Bearer oauth"
        assert request.headers["x-goog-user-project"] == "project"
        assert json.loads(request.content) == {
            "input": {"text": "Ваш аргумент?"},
            "voice": {"languageCode": "ru-RU", "name": "configured-voice"},
            "audioConfig": {"audioEncoding": encoding, "speakingRate": 1},
        }
        return httpx.Response(200, json={"audioContent": base64.b64encode(b"audio").decode()})

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        provider = GoogleTTS(
            client=client, access_token="oauth", quota_project="project", voice="configured-voice",
            audio_format=audio_format,
        )
        assert await provider.synthesize("Ваш аргумент?") == SynthesizedAudio(b"audio", mime)


async def test_google_adc_refresh_and_reuse():
    pytest.importorskip("google.auth", reason="needs the botkit[google] extra")

    class Credentials:
        valid = False
        token = ""
        count = 0
        quota_project_id = "adc-project"

        def refresh(self, request):
            self.count += 1
            self.token, self.valid = "refreshed-token", True

    credentials = Credentials()
    async with httpx.AsyncClient() as client:
        provider = GoogleTTS(client=client, credentials=credentials)
        assert await asyncio.gather(*(provider.authorization() for _ in range(8))) == ["refreshed-token"] * 8
        assert credentials.count == 1


async def test_google_adc_failure_is_redacted():
    class Credentials:
        valid = False

        def refresh(self, request):
            raise ValueError("private key and local path")

    async with httpx.AsyncClient() as client:
        with pytest.raises(TTSError, match="authentication failed") as error:
            await GoogleTTS(client=client, credentials=Credentials()).authorization()
        assert "private key" not in str(error.value)


@pytest.mark.parametrize("provider", ["yandex", "google"])
@pytest.mark.parametrize("status", [302, 401, 403, 429, 500])
async def test_errors_never_redirect_retry_or_echo_secrets(provider, status):
    calls = []

    def server(request):
        calls.append(request)
        return httpx.Response(
            status, content=b"secret-key sensitive input",
            headers={"Location": "https://evil.test", "Content-Type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(server), follow_redirects=True) as client:
        speech = (
            YandexTTS(api_key="secret", client=client) if provider == "yandex"
            else GoogleTTS(access_token="oauth", client=client)
        )
        with pytest.raises(TTSError) as error:
            await speech.synthesize("Question?")
        assert "secret-key" not in str(error.value) and len(calls) == 1


@pytest.mark.parametrize("payload", [{}, {"audioContent": ""}, {"audioContent": "not-base64!"}, {"audioContent": 123}, []])
async def test_google_malformed_response(payload):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
        with pytest.raises(TTSError, match="invalid audioContent"):
            await GoogleTTS(client=client, access_token="test").synthesize("Question?")


@pytest.mark.parametrize("provider", ["yandex", "google"])
async def test_input_limit_and_connection_redaction(provider):
    calls = []

    def server(request):
        calls.append(request)
        raise httpx.ConnectError("secret host and bearer", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(server)) as client:
        speech = (
            YandexTTS(api_key="secret", client=client) if provider == "yandex"
            else GoogleTTS(access_token="oauth", client=client)
        )
        with pytest.raises(ValueError):
            await speech.synthesize("я" * 16000)
        assert not calls
        with pytest.raises(TTSError, match="^TTS connection failed$"):
            await speech.synthesize("Question?")


def test_bad_cloud_configuration_rejected():
    with pytest.raises(ValueError):
        YandexTTS(api_key="key", iam_token="token")
    with pytest.raises(ValueError):
        YandexTTS(api_key="key", folder_id="folder")
    with pytest.raises(ValueError):
        GoogleTTS(audio_format="unsupported")


# ---- Kokoro ---------------------------------------------------------------------------------------


def wav(seconds=0.1, rate=24000):
    pcm = b"\x00\x00" * int(seconds * rate)
    header = (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    )
    return header + b"data" + struct.pack("<I", len(pcm)) + pcm


def kokoro(handler, **kwargs):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return KokoroTTS("http://tts.internal:8777/", api_key="secret-key", client=client, **kwargs)


def declared_rate(ogg):
    head = ogg.index(b"OpusHead")
    return struct.unpack("<I", ogg[head + 12 : head + 16])[0]


async def test_kokoro_request_follows_the_server_contract():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers.get("authorization")
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, content=wav(), headers={"content-type": "audio/wav"})

    audio = await kokoro(handler, voice="bf_emma", speed=1.2, output="wav").synthesize("  What is your main claim?  ")
    assert seen["url"] == "http://tts.internal:8777/v1/audio/speech" and seen["auth"] == "Bearer secret-key"
    assert seen["json"] == {
        "model": "kokoro", "input": "What is your main claim?", "voice": "bf_emma", "speed": 1.2,
        "response_format": "wav",
    }
    assert audio.mime_type == "audio/wav" and audio.data[:4] == b"RIFF"
    await kokoro(handler, output="wav").synthesize("Hi", voice="am_michael")
    assert seen["json"]["voice"] == "am_michael"


@pytest.mark.parametrize("text", ["", "   ", "x" * 5001])
async def test_kokoro_text_limits(text):
    with pytest.raises(ValueError, match="1..5000"):
        await kokoro(lambda r: httpx.Response(200, content=wav())).synthesize(text)


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (httpx.Response(401, text="private"), "HTTP 401"),
        (httpx.Response(200, content=b"not audio at all, just text bytes" * 5), "neither WAV nor OGG"),
        (httpx.Response(200, content=b""), "neither WAV nor OGG"),
    ],
)
async def test_kokoro_errors_are_safe(response, reason):
    with pytest.raises(KokoroError, match=reason) as info:
        await kokoro(lambda r: response).synthesize("Secret student topic?")
    assert "secret-key" not in str(info.value) and "Secret student" not in str(info.value)
    assert isinstance(info.value, TTSError)


async def test_kokoro_health_checks_the_voice():
    ok = kokoro(lambda r: httpx.Response(200, json={"status": "ready", "voices": ["af_heart"]}))
    assert (await ok.health())["status"] == "ready"
    wrong = kokoro(lambda r: httpx.Response(200, json={"voices": ["bm_george"]}))
    with pytest.raises(KokoroError, match="not offered"):
        await wrong.health()


async def test_kokoro_concurrency_is_limited():
    release, running, peak = asyncio.Event(), 0, 0

    async def handler(request):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await release.wait()
        running -= 1
        return httpx.Response(200, content=wav())

    provider = kokoro(handler, max_concurrent=2)
    tasks = [asyncio.create_task(provider.synthesize("Question?")) for _ in range(5)]
    await asyncio.sleep(0.05)
    assert peak == 2
    release.set()
    assert len(await asyncio.gather(*tasks)) == 5


async def test_kokoro_default_output_is_ogg_opus_at_16k_and_reports_seconds():
    audio = await kokoro(lambda r: httpx.Response(200, content=wav(1.5))).synthesize("Question?")
    assert audio.mime_type == "audio/ogg" and audio.data[:4] == b"OggS" and b"OpusHead" in audio.data[:100]
    assert declared_rate(audio.data) == 16000 and audio.seconds == 1.5 and audio.is_voice_message
    assert declared_rate(to_ogg_opus(wav(0.5, rate=22050))) == 16000
    assert declared_rate(to_ogg_opus(wav(0.5), target_rate=48000)) == 48000
    with pytest.raises(ValueError, match="Unsupported sample rate"):
        to_ogg_opus(wav(0.2, rate=4000))
    with pytest.raises(ValueError, match="KOKORO_TTS_OPUS_RATE"):
        KokoroTTS("http://tts.internal:8777", opus_rate=44100)


@pytest.mark.parametrize("bad", [b"RIFF" + b"\x00" * 4 + b"WAVE" + b"junk" * 20, wav(0.0)])
async def test_kokoro_unconvertible_audio_is_a_safe_error(bad):
    with pytest.raises(KokoroError, match="conversion failed"):
        await kokoro(lambda r: httpx.Response(200, content=bad)).synthesize("Question?")


async def test_kokoro_asks_for_ogg_and_falls_back_to_wav_after_422():
    formats = []

    def handler(request):
        fmt = json.loads(request.content)["response_format"]
        formats.append(fmt)
        return httpx.Response(422) if fmt == "ogg" else httpx.Response(200, content=wav(0.5))

    provider = kokoro(handler)
    assert (await provider.synthesize("Question?")).data[:4] == b"OggS"
    await provider.synthesize("Another?")
    assert formats == ["ogg", "wav", "wav"]  # remembered: no more failed attempts


async def test_kokoro_retries_stalls_and_5xx_but_not_4xx():
    calls = []

    async def stall_once(request):
        calls.append(1)
        if len(calls) == 1:
            await asyncio.sleep(5)
        return httpx.Response(200, content=wav())

    audio = await kokoro(stall_once, attempt_timeout=0.1, timeout=30).synthesize("Question?")
    assert audio.data[:4] == b"OggS" and len(calls) == 2

    async def always_stall(request):
        await asyncio.sleep(5)

    with pytest.raises(KokoroError, match=r"no audio after 3 attempts: timed out, no response headers") as info:
        await kokoro(always_stall, attempt_timeout=0.05, attempts=3, timeout=30).synthesize("Secret question?")
    assert "Secret" not in str(info.value) and "secret-key" not in str(info.value)

    codes = {"5xx": 0, "4xx": 0}

    def flaky(request):
        codes["5xx"] += 1
        return httpx.Response(503) if codes["5xx"] < 3 else httpx.Response(200, content=wav())

    assert (await kokoro(flaky).synthesize("Question?")).data[:4] == b"OggS" and codes["5xx"] == 3

    def forbidden(request):
        codes["4xx"] += 1
        return httpx.Response(401)

    with pytest.raises(KokoroError, match="HTTP 401"):
        await kokoro(forbidden).synthesize("Question?")
    assert codes["4xx"] == 1


async def test_kokoro_log_line_has_server_timing_but_no_text(caplog):
    handler = lambda r: httpx.Response(  # noqa: E731
        200, content=wav(0.2), headers={"x-tts-generation-seconds": "0.12", "x-tts-queue-seconds": "0.1"}
    )
    with caplog.at_level(logging.INFO, logger="botkit.tts.kokoro"):
        await kokoro(handler).synthesize("Secret student question?")
    line = next(r.getMessage() for r in caplog.records if "Kokoro TTS:" in r.getMessage())
    assert "server generation 0.12s" in line and "Secret" not in line


def test_kokoro_validation_and_wav_seconds():
    for kwargs, pattern in [({"speed": 3}, "SPEED"), ({"request_format": "mp3"}, "REQUEST_FORMAT"), ({"output": "mp3"}, "OUTPUT")]:
        with pytest.raises(ValueError, match=pattern):
            KokoroTTS("http://tts.internal:8777", **kwargs)
    with pytest.raises(ValueError, match="KOKORO_TTS_URL"):
        KokoroTTS("")
    assert wav_seconds(wav(1.0)) == 1.0 and wav_seconds(b"not a wav") is None


# ---- cache / timeout / usage wrapper ---------------------------------------------------------------


class Counting:
    voice = "v1"

    def __init__(self):
        self.count = 0

    async def synthesize(self, text, *, voice=None):
        self.count += 1
        await asyncio.sleep(0.01)
        return SynthesizedAudio(text.encode(), "audio/ogg")

    async def aclose(self):
        self.closed = True


async def test_cache_serializes_identical_requests_and_separates_voices_and_texts():
    inner = Counting()
    tts = CachedSynthesizer(inner)
    answers = await asyncio.gather(*(tts.synthesize("question") for _ in range(10)))
    assert inner.count == 1 and all(a.data == b"question" for a in answers)
    await tts.synthesize("question", voice="other")
    await tts.synthesize("different")
    assert inner.count == 3
    tts.cache_ttl = 0.0  # expiry is checked against the stored deadline
    for key in list(tts.cache):
        tts.cache[key] = (0.0, tts.cache[key][1])
    await tts.synthesize("question")
    assert inner.count == 4
    tts.cache_max_bytes = 12
    await tts.synthesize("third question")  # larger than the budget: served but not kept
    assert tts.cache_bytes <= 12
    await tts.aclose()
    assert not tts.cache and not tts.cache_bytes and inner.closed


async def test_cache_evicts_the_oldest_entry():
    tts = CachedSynthesizer(Counting(), cache_max_entries=2)
    for text in ("a", "b", "c"):
        await tts.synthesize(text)
    assert len(tts.cache) == 2 and tts.cache_bytes == 2
    await tts.aclose()


async def test_timeout_and_cancel_and_failure_are_not_cached():
    class Slow:
        async def synthesize(self, text, *, voice=None):
            await asyncio.sleep(20)

        async def aclose(self):
            pass

    tts = CachedSynthesizer(Slow(), timeout=0.01)
    with pytest.raises(TimeoutError):
        await tts.synthesize("question")
    assert not tts.cache
    task = asyncio.create_task(CachedSynthesizer(Slow()).synthesize("question"))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with pytest.raises(ValueError):
        CachedSynthesizer(Slow(), timeout=float("nan"))


async def test_usage_has_counts_latency_status_and_context_but_no_text():
    events = []

    class Tracker:
        async def log(self, event):
            events.append(event)

    tts = CachedSynthesizer(Counting(), tracker=Tracker(), provider="fake")
    with usage_context(bot_id="b", user_id="u", chat_id="c", platform="vk"):
        await tts.synthesize("Private question?")
        await tts.synthesize("Private question?")  # cache hit: no provider call, no event
    (event,) = events
    assert event.meta["characters"] == 17 and event.meta["audio_bytes"] == 17
    assert not event.input_text and not event.output_text and event.cost is None
    assert (event.kind, event.operation, event.status, event.provider) == ("speech", "tts", "success", "fake")
    assert (event.bot_id, event.user_id, event.chat_id, event.platform) == ("b", "u", "c", "vk")
    assert event.cost_status == "UNKNOWN_USAGE" and "Private" not in repr(event.meta)

    class Broken:
        async def synthesize(self, text, *, voice=None):
            raise TTSError("boom")

        async def aclose(self):
            pass

    events.clear()
    with pytest.raises(TTSError):
        await CachedSynthesizer(Broken(), tracker=Tracker()).synthesize("x")
    assert events[0].status == "error" and events[0].error_type == "TTSError"


# ---- loader / registry ---------------------------------------------------------------------------


def test_loader_disabled_and_errors(monkeypatch):
    monkeypatch.delenv("TTS_PROVIDER", raising=False)
    assert load_tts() is None and load_tts("disabled") is None
    with pytest.raises(ValueError, match="TTS_PROVIDER.*kokoro.*yandex"):
        load_tts("typo")
    with pytest.raises(ValueError, match="module:factory"):
        load_tts(":x")


async def test_loader_builds_kokoro_from_env_with_cache_and_timeout(monkeypatch):
    monkeypatch.setenv("TTS_PROVIDER", "kokoro")
    monkeypatch.setenv("KOKORO_TTS_URL", "http://tts.internal:8777")
    monkeypatch.setenv("TTS_TIMEOUT", "12")
    monkeypatch.setenv("TTS_CACHE_TTL", "60")
    tts = load_tts()
    assert isinstance(tts, CachedSynthesizer) and isinstance(tts.inner, KokoroTTS)
    assert tts.timeout == 12 and tts.cache_ttl == 60 and tts.inner.timeout == 12
    await tts.aclose()
    raw = load_tts(cache=False)
    assert isinstance(raw, KokoroTTS)
    await raw.aclose()
    monkeypatch.setenv("KOKORO_TTS_URL", "")
    with pytest.raises(ValueError, match="KOKORO_TTS_URL"):
        load_tts()
    monkeypatch.setenv("KOKORO_TTS_URL", "http://tts.internal:8777")
    monkeypatch.setenv("KOKORO_TTS_SPEED", "3")
    with pytest.raises(ValueError, match="SPEED"):
        load_tts()


def test_loader_builds_yandex_from_env(monkeypatch):
    monkeypatch.setenv("YANDEX_TTS_API_KEY", "secret")
    monkeypatch.setenv("YANDEX_TTS_VOICE", "alena")
    monkeypatch.setenv("TTS_FORMAT", "mp3")
    tts = load_tts("yandex", cache=False)
    assert isinstance(tts, YandexTTS) and tts.voice == "alena" and tts.audio_format == "mp3"


async def test_registered_and_module_factory_providers(monkeypatch):
    monkeypatch.setattr(factory, "_REGISTRY", dict(factory._REGISTRY))  # keep the global registry clean
    register_tts("fake-tts", lambda timeout: Counting())
    assert "fake-tts" in available_tts()
    with pytest.raises(ValueError, match="already registered"):
        register_tts("fake-tts", lambda timeout: Counting())
    assert isinstance(load_tts("fake-tts", cache=False), Counting)

    module = types.ModuleType("my_tts_plugin")
    module.build = Counting
    monkeypatch.setitem(sys.modules, "my_tts_plugin", module)
    custom = load_tts("my_tts_plugin:build")
    assert isinstance(custom, CachedSynthesizer) and custom.provider == "my_tts_plugin:build"
    assert (await custom.synthesize("hi")).data == b"hi"


def test_synthesized_audio_validates_its_content():
    with pytest.raises(ValueError):
        SynthesizedAudio(b"")
    with pytest.raises(ValueError):
        SynthesizedAudio(b"x", "text/plain")
    with pytest.raises(ValueError):
        SynthesizedAudio(b"x" * (10 * 1024 * 1024 + 1))
    assert SynthesizedAudio(b"x", "audio/mpeg").extension == "mp3"
    assert not SynthesizedAudio(b"x", "audio/mpeg").is_voice_message


# ---- delivery ------------------------------------------------------------------------------------


class FakeTyping:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        self.log.append("typing")
        return self

    async def __aexit__(self, *_):
        self.log.append("typing-stop")


class FakeAdapter:
    def __init__(self, fail=None):
        self.sent, self.log, self.fail = [], [], fail

    async def start_typing(self, chat_id):
        return FakeTyping(self.log)

    async def send_attachment(self, chat_id, data, *, filename, kind, caption="", **options):
        if self.fail:
            raise self.fail
        self.sent.append((chat_id, data, filename, kind, caption))
        return "42"


async def test_speak_sends_ogg_as_voice_and_other_formats_as_audio():
    adapter = FakeAdapter()
    audio = await speak(adapter, "7", "hello", Counting(), caption="cap", basename="q")
    assert audio.data == b"hello" and adapter.sent == [("7", b"hello", "q.ogg", "voice", "cap")]
    assert adapter.log == ["typing", "typing-stop"]

    class Mp3:
        async def synthesize(self, text, *, voice=None):
            return SynthesizedAudio(b"mp3", "audio/mpeg", 1.5)

    adapter = FakeAdapter()
    result = await speak(adapter, "7", "hello", Mp3(), typing=False)
    assert adapter.sent[0][2:4] == ("voice.mp3", "audio") and result.seconds == 1.5 and not adapter.log
    await send_audio(adapter, "7", SynthesizedAudio(b"w", "audio/wav"))
    assert adapter.sent[-1][3] == "audio"


async def test_speak_failures_return_none_and_log_no_content(caplog):
    class Failing:
        async def synthesize(self, text, *, voice=None):
            raise TTSError("TTS HTTP status 500")

    with caplog.at_level(logging.WARNING, logger="botkit.tts.delivery"):
        assert await speak(FakeAdapter(), "7", "Private words", Failing()) is None
        assert await speak(FakeAdapter(fail=RuntimeError("upload down")), "7", "Private words", Counting()) is None
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "TTS HTTP status 500" in text and "upload down" in text and "Private" not in text
