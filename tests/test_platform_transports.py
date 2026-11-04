import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from aiogram import Bot
from aiogram.types import Message
from vkbottle import API
from vkbottle import Bot as VKBot
from vkbottle.bot import Message as VKMessage

from botkit.extraction import ContentExtractionService
from botkit.llm import PneumaConfig, PneumaTranscriber
from botkit.transport import Attachment, Button, CallbackQuery, IncomingMessage, MemoryAdapter
from botkit.transport.base import split_text
from botkit.transport.telegram import TelegramAdapter
from botkit.transport.vk import VKAdapter


@pytest.mark.parametrize(
    "text",
    ["я" * 9000, "😀" * 4097, ("абв\n" * 1700), "a b " * 2500, "", "x"],
    ids=["cyrillic", "emoji", "newlines", "spaces", "empty", "short"],
)
def test_splitting_preserves_unicode_and_whitespace(text):
    chunks = split_text(text, 4000)
    assert "".join(chunks) == text
    assert all(0 < len(c.encode("utf-16-le")) // 2 <= 4000 for c in chunks)


async def test_commands_dispatch_once_and_typing_stops():
    adapter = MemoryAdapter()
    calls = []

    async def command(message):
        calls.append("command")

    async def other(message):
        calls.append("other")

    adapter.on_command("start", command)
    adapter.on_message(other)
    await adapter.dispatch(IncomingMessage("memory", "u", "c", "/start@MyBot value"))
    await adapter.dispatch(IncomingMessage("memory", "u", "c", "hello"))
    assert calls == ["command", "other"]
    with pytest.raises(ValueError):
        adapter.on_command("start", command)
    with pytest.raises(RuntimeError):
        async with await adapter.start_typing("c"):
            await asyncio.sleep(0.01)
            raise RuntimeError("handler failed")
    count = adapter.typing_count
    await asyncio.sleep(0.01)
    assert adapter.typing_count == count == 1
    await adapter.aclose()


async def test_attachments_reject_oversized_before_and_after_read():
    reader = AsyncMock(return_value=b"12345")
    attachment = Attachment("document", "text/plain", reader, size=6, max_bytes=4)
    with pytest.raises(ValueError):
        await attachment.read_bytes()
    reader.assert_not_awaited()
    attachment.size = None
    with pytest.raises(ValueError):
        await attachment.read_bytes()


def tg_message(**values):
    return Message.model_validate(
        {
            "message_id": 1,
            "date": datetime.now(UTC),
            "chat": {"id": 10, "type": "private"},
            "from": {"id": 20, "is_bot": False, "first_name": "Иван", "username": "ivan"},
            **values,
        }
    )


async def test_telegram_sdk_normalization_send_keyboard_and_download():
    bot = Bot("123456789:abcdefghijklmnopqrstuvwxyz123456789")
    bot.send_message = AsyncMock(return_value=tg_message(text="ok"))
    bot.get_file = AsyncMock(return_value=SimpleNamespace(file_path="doc/test", file_size=3))

    async def download(**kwargs):
        yield b"abc"

    bot.session.stream_content = download
    adapter = TelegramAdapter(bot=bot)
    original = tg_message(
        caption="Файл",
        document={
            "file_id": "f",
            "file_unique_id": "unique",
            "file_name": "test.txt",
            "mime_type": "text/plain",
            "file_size": 3,
        },
    )
    message = adapter.normalize(original)
    assert (message.user_id, message.chat_id, message.platform_username) == ("20", "10", "ivan")
    assert message.platform_display_name == "Иван" and message.text == "Файл"
    assert await message.attachments[0].read_bytes() == b"abc"
    await adapter.send("10", "😀" * 4000, buttons=[[Button("OK", data="ok")]])
    assert bot.send_message.await_count == 2
    calls = bot.send_message.await_args_list
    assert "reply_markup" not in calls[0].kwargs
    assert calls[1].kwargs["reply_markup"].inline_keyboard[0][0].callback_data == "ok"
    assert calls[1].kwargs["parse_mode"] is None
    await adapter.aclose()
    await bot.session.close()


async def test_telegram_streamed_download_limit():
    bot = Bot("123456789:abcdefghijklmnopqrstuvwxyz123456789")
    bot.get_file = AsyncMock(return_value=SimpleNamespace(file_path="x", file_size=None))

    async def download(**kwargs):
        yield b"123"
        yield b"456"

    bot.session.stream_content = download
    adapter = TelegramAdapter(bot=bot, max_attachment_bytes=4)
    with pytest.raises(ValueError):
        await adapter._download("file-id")
    await adapter.aclose()
    await bot.session.close()


async def test_telegram_raw_api_media_edit_callback_and_rate_limit(monkeypatch):
    from aiogram.exceptions import TelegramRetryAfter
    from aiogram.methods import SendMessage

    bot = Bot("123456789:abcdefghijklmnopqrstuvwxyz123456789")
    bot.send_message = AsyncMock(
        side_effect=[
            TelegramRetryAfter(SendMessage(chat_id=10, text="x"), "slow", retry_after=0),
            tg_message(text="x"),
        ]
    )
    bot.send_document = AsyncMock(return_value=tg_message(text="ok"))
    bot.edit_message_text = AsyncMock(return_value=True)
    bot.answer_callback_query = AsyncMock(return_value=True)
    bot.delete_message = AsyncMock(return_value=True)
    adapter = TelegramAdapter(bot=bot)
    assert await adapter.send("10", "hello") == ["1"]
    assert bot.send_message.await_count == 2
    assert await adapter.send_attachment("10", b"data", filename="test.txt") == "1"
    assert bot.send_document.await_args.kwargs["document"].data == b"data"
    await adapter.edit("10", "1", "edited")
    await adapter.delete("10", "1")
    await adapter.answer_callback(CallbackQuery("telegram", "q", "20", "10", "ok"), "Done")
    assert bot.answer_callback_query.await_args.kwargs["callback_query_id"] == "q"
    await adapter.aclose()
    await bot.session.close()


def vk_api():
    # Real VK API and validators, only HTTP boundary mocked. In particular,
    # this catches mistaken assumptions about the SDK's response envelope.
    client = SimpleNamespace(request_text=AsyncMock(), close=AsyncMock())
    client.request_text.return_value = json.dumps({"response": 77})
    return API("test-token", http_client=client), client


@pytest.mark.parametrize("platform", ["telegram", "vk"])
@pytest.mark.parametrize("kind", ["voice", "audio", "document"])
async def test_audio_attachment_download_to_pneuma(platform, kind):
    audio = b"OggSvoice" if kind == "voice" else b"ID3music"
    transcript = "Распознанная речь"
    uploaded = []

    def asr_server(request):
        if request.method == "POST":
            uploaded.append(request.content)
            return httpx.Response(200, json={"job_id": "c" * 32, "status": "completed"})
        return httpx.Response(200, json={"status": "completed", "ready": True, "transcript": transcript})

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(asr_server)) as asr_http,
        httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=audio))
        ) as download,
    ):
        asr = PneumaTranscriber(PneumaConfig("https://asr.test", trust_env=False), client=asr_http)
        service = ContentExtractionService(transcriber=asr)
        if platform == "telegram":
            bot = Bot("123456789:abcdefghijklmnopqrstuvwxyz123456789")
            bot.get_file = AsyncMock(
                return_value=SimpleNamespace(file_path="audio/test", file_size=len(audio))
            )

            async def download_tg(**kwargs):
                yield audio

            bot.session.stream_content = download_tg
            adapter = TelegramAdapter(bot=bot)
            media = {"file_id": "f", "file_unique_id": "u", "duration": 1, "file_size": len(audio)}
            if kind != "voice":
                media.update(
                    file_name="message.mp3",
                    mime_type="application/octet-stream" if kind == "document" else "audio/mpeg",
                )
            incoming = adapter.normalize(tg_message(**{kind: media}))
        else:
            api, api_http = vk_api()
            api_http.request_text.return_value = json.dumps({"response": [{"id": 20, "first_name": "Test"}]})
            adapter = VKAdapter(bot=VKBot(api=api), download_client=download)
            media = {"id": 1, "owner_id": 20, "duration": 1}
            media_kind = {"voice": "audio_message", "document": "doc"}.get(kind, kind)
            if kind == "voice":
                media.update(
                    link_ogg="https://files.userapi.com/voice.ogg",
                    link_mp3="https://files.userapi.com/voice.mp3",
                    waveform=[1],
                )
            elif kind == "document":
                media.update(
                    title="message.mp3",
                    size=len(audio),
                    ext="mp3",
                    date=1,
                    type=1,
                    url="https://files.userapi.com/a.mp3",
                )
            else:
                media.update(artist="Test", title="message", url="https://files.userapi.com/a.mp3")
            original = VKMessage.model_validate(
                {
                    "conversation_message_id": 1,
                    "date": 1,
                    "from_id": 20,
                    "id": 2,
                    "text": "",
                    "version": 1,
                    "out": False,
                    "peer_id": 10,
                    "attachments": [{"type": media_kind, media_kind: media}],
                }
            )
            incoming = await adapter.normalize(original)
        try:
            assert len(incoming.attachments) == 1
            result = await service.extract(incoming.attachments[0])
            assert result.text == transcript and result.kind == "transcript"
            assert len(uploaded) == 1 and audio in uploaded[0]
            assert b'name="audio"' in uploaded[0]
        finally:
            await adapter.aclose()
            if platform == "telegram":
                await bot.session.close()


async def test_vk_sdk_raw_envelope_send_and_random_ids():
    api, client = vk_api()
    adapter = VKAdapter(bot=VKBot(api=api))
    ids = await adapter.send("2000000001", "я" * 9000, buttons=[[Button("ok", data="ok")]])
    assert ids == ["77", "77", "77"]
    calls = client.request_text.await_args_list
    values = [call.kwargs["data"] for call in calls]
    assert len({p["random_id"] for p in values}) == 3
    assert "keyboard" not in values[0]
    keyboard = json.loads(values[-1]["keyboard"])
    assert keyboard["buttons"][0][0]["action"]["type"] == "callback"
    assert values[0]["peer_id"] == 2000000001
    await adapter.aclose()


async def test_vk_sdk_message_attachments_identity_and_callback():
    api, client = vk_api()
    client.request_text.return_value = json.dumps(
        {"response": [{"id": 20, "first_name": "Иван", "last_name": "Иванов", "screen_name": "ivan"}]}
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"abc"))
    ) as download:
        adapter = VKAdapter(bot=VKBot(api=api), download_client=download)
        message = VKMessage.model_validate(
            {
                "conversation_message_id": 1,
                "date": 1,
                "from_id": 20,
                "id": 2,
                "text": "doc",
                "version": 1,
                "out": False,
                "peer_id": 10,
                "attachments": [
                    {
                        "type": "doc",
                        "doc": {
                            "id": 1,
                            "owner_id": 20,
                            "title": "test.txt",
                            "size": 3,
                            "ext": "txt",
                            "date": 1,
                            "type": 1,
                            "url": "https://files.userapi.com/test.txt",
                        },
                    }
                ],
            }
        )
        normalized = await adapter.normalize(message)
        assert normalized.platform_display_name == "Иван Иванов"
        assert normalized.attachments[0].mime_type == "text/plain"
        assert await normalized.attachments[0].read_bytes() == b"abc"
        await adapter.normalize(message)
        assert client.request_text.await_count == 1  # bounded profile cache
        callback = AsyncMock()
        adapter.on_callback(callback)
        await adapter._receive_callback(
            {
                "object": {
                    "event_id": "e",
                    "user_id": 20,
                    "peer_id": 10,
                    "conversation_message_id": 1,
                    "payload": {"data": "ok"},
                }
            }
        )
        assert callback.await_args.args[0].data == "ok"
        await adapter.aclose()


async def test_vk_document_upload_uses_real_sdk():
    api, client = vk_api()
    calls = []

    async def http(url, **kwargs):
        calls.append((url, kwargs))
        if "docs.getMessagesUploadServer" in url:
            return json.dumps({"response": {"upload_url": "https://upload.test/doc"}})
        if "upload.test" in url:
            assert kwargs["data"]["file"].getvalue() == b"contents"
            assert kwargs["data"]["file"].name == "report.txt"
            return json.dumps({"file": "upload-id"})
        if "docs.save" in url:
            return json.dumps({"response": {"type": "doc", "doc": {"owner_id": 2, "id": 3}}})
        return json.dumps({"response": 99})

    client.request_text.side_effect = http
    adapter = VKAdapter(bot=VKBot(api=api))
    assert await adapter.send_attachment("10", b"contents", filename="report.txt") == "99"
    assert calls[-1][1]["data"]["attachment"] == "doc2_3"
    await adapter.aclose()


async def test_vk_http_download_bound_and_requires_https():
    api, _ = vk_api()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"12345"))
    ) as client:
        adapter = VKAdapter(bot=VKBot(api=api), download_client=client, max_attachment_bytes=3)
        with pytest.raises(ValueError):
            await adapter._download("https://files.userapi.com/a")
        with pytest.raises(ValueError):
            await adapter._download("http://files.userapi.com/a")
        await adapter.aclose()


async def test_shutdown_cancels_handlers():
    adapter = MemoryAdapter()
    entered = asyncio.Event()

    async def slow(message):
        entered.set()
        await asyncio.Event().wait()

    adapter.on_message(slow)
    task = asyncio.create_task(adapter.dispatch(IncomingMessage("memory", "u", "c", "hi")))
    await entered.wait()
    await adapter.aclose()
    assert task.cancelled()


async def test_vk_random_id_cannot_deduplicate_different_chunks():
    api, _ = vk_api()
    adapter = VKAdapter(bot=VKBot(api=api))
    try:
        with pytest.raises(ValueError, match="random_id"):
            await adapter.send("10", "x" * 5000, random_id=123)
    finally:
        await adapter.aclose()
