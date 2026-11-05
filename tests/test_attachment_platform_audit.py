import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.telegram import TelegramAPIServer
from aiogram.types import Message, Update
from vkbottle import API
from vkbottle import Bot as VKBot

from botkit.transport.telegram import TelegramAdapter, TelegramAttachmentDownloadError
from botkit.transport.vk import VKAdapter

TOKEN = "123456789:abcdefghijklmnopqrstuvwxyz123456789"


def tg_message(**fields):
    return Message.model_validate(
        {
            "message_id": 1,
            "date": 1,
            "chat": {"id": 10, "type": "private"},
            "from": {"id": 20, "is_bot": False, "first_name": "Test"},
            **fields,
        }
    )


def vk_adapter():
    http = SimpleNamespace(request_text=AsyncMock(return_value='{"response":1}'), close=AsyncMock())
    return VKAdapter(bot=VKBot(api=API("dummy", http_client=http))), http


async def test_all_four_documented_a4_acceptance_regressions_are_now_green(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "attachment_acceptance", Path(__file__).parents[1] / "scripts/acceptance.py"
    )
    harness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = harness
    spec.loader.exec_module(harness)
    report = harness.Report(tmp_path / "acceptance", "offline")
    await harness.regression_checks(report)
    assert len(report.data["cases"]) == 4
    assert all(case["status"] == "passed" for case in report.data["cases"]), report.data["cases"]


@pytest.mark.parametrize("kind", ["animation", "video_note"])
async def test_telegram_new_media_are_deferred_and_not_duplicated_as_documents(kind):
    bot = Bot(TOKEN)
    value = {"file_id": "video", "file_unique_id": "unique", "width": 10, "height": 10, "duration": 1}
    if kind == "video_note":
        value["length"] = 10
    bot.get_file = AsyncMock()
    adapter = TelegramAdapter(bot=bot)
    fields = {kind: value}
    if kind == "animation":
        fields["document"] = {"file_id": "video", "file_unique_id": "unique"}
    try:
        normalized = adapter.normalize(tg_message(**fields))
        assert len(normalized.attachments) == 1
        assert normalized.attachments[0].kind == "video"
        assert normalized.attachments[0].mime_type == "video/mp4"
        assert normalized.attachments[0].platform_id == "video"
        bot.get_file.assert_not_awaited()
    finally:
        await adapter.aclose()
        await bot.session.close()


async def test_telegram_framework_and_native_handlers_receive_once():
    bot, dispatcher = Bot(TOKEN), Dispatcher()
    adapter = TelegramAdapter(bot=bot, dispatcher=dispatcher)
    framework, native = AsyncMock(), AsyncMock()
    adapter.on_message(framework)

    async def native_handler(message):
        await native(message)

    dispatcher.message.register(native_handler)
    try:
        await dispatcher.feed_update(bot, Update(update_id=1, message=tg_message(text="synthetic")))
        framework.assert_awaited_once()
        native.assert_awaited_once()
    finally:
        await adapter.aclose()
        await bot.session.close()


@pytest.mark.parametrize("callback_payload", [None, [], 7, True, "private payload"])
async def test_vk_valid_json_non_object_callback_payload_does_not_crash(callback_payload):
    adapter, _ = vk_adapter()
    handler = AsyncMock()
    adapter.on_callback(handler)
    try:
        await adapter._receive_callback(
            {
                "object": {
                    "event_id": "e",
                    "user_id": 20,
                    "peer_id": 10,
                    "conversation_message_id": 7,
                    "payload": json.dumps(callback_payload),
                }
            }
        )
        if handler.await_count:
            assert handler.await_args.args[0].data == ""
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("operation", ["edit", "delete"])
async def test_vk_callback_ids_are_explicitly_local_in_edit_and_delete(operation):
    adapter, http = vk_adapter()

    async def handler(query):
        assert query.message_id == "cmid:7"
        if operation == "edit":
            await adapter.edit(query.chat_id, query.message_id, "synthetic")
        else:
            await adapter.delete(query.chat_id, query.message_id)

    adapter.on_callback(handler)
    try:
        await adapter._receive_callback(
            {
                "object": {
                    "event_id": "e",
                    "user_id": 20,
                    "peer_id": 2000000001,
                    "conversation_message_id": 7,
                    "payload": {"data": "ok"},
                }
            }
        )
        data = http.request_text.await_args.kwargs["data"]
        assert data["peer_id"] == 2000000001
        assert "message_id" not in data and "message_ids" not in data
        assert str(data["cmid" if operation == "edit" else "cmids"]) == "7"
    finally:
        await adapter.aclose()


@pytest.mark.parametrize("operation", ["edit", "delete"])
async def test_vk_numeric_message_ids_remain_global(operation):
    adapter, http = vk_adapter()
    try:
        if operation == "edit":
            await adapter.edit("10", "7", "synthetic")
        else:
            await adapter.delete("10", "7")
        data = http.request_text.await_args.kwargs["data"]
        assert str(data["message_id" if operation == "edit" else "message_ids"]) == "7"
        assert "cmid" not in data and "cmids" not in data
    finally:
        await adapter.aclose()


async def test_telegram_startup_failure_closes_owned_session():
    adapter = TelegramAdapter(TOKEN)
    adapter.native.get_me = AsyncMock(side_effect=RuntimeError("synthetic startup failure"))
    adapter.native.session.close = AsyncMock()
    with pytest.raises(RuntimeError):
        await adapter.run()
    adapter.native.session.close.assert_awaited_once()


async def test_telegram_real_sdk_overflow_closes_remote_stream_immediately():
    bot = Bot(TOKEN)
    bot.get_file = AsyncMock(return_value=SimpleNamespace(file_path="documents/test", file_size=None))
    streams, closed, yielded = [], [], []

    async def response_body():
        try:
            for part in [b"123", b"456", b"not-read"]:
                yielded.append(part)
                yield part
        finally:
            closed.append(True)

    def content(**kwargs):
        stream = response_body()
        streams.append(stream)  # keep alive: do not mistake delayed GC cleanup for explicit close
        return stream

    bot.session.stream_content = content
    adapter = TelegramAdapter(bot=bot, max_attachment_bytes=4)
    try:
        with pytest.raises(ValueError):
            await adapter._download("file-id")
        assert yielded == [b"123", b"456"]
        assert closed == [True]
    finally:
        for stream in streams:
            await stream.aclose()
        await adapter.aclose()
        await bot.session.close()


@pytest.mark.parametrize("mode", ["success", "cancel", "http_error"])
async def test_telegram_remote_stream_is_closed_on_every_exit_and_errors_are_safe(mode):
    bot = Bot(TOKEN)
    bot.get_file = AsyncMock(return_value=SimpleNamespace(file_path="documents/test", file_size=None))
    closed, calls = [], []

    async def content(**kwargs):
        calls.append(kwargs)
        try:
            yield b"data"
            if mode == "cancel":
                raise asyncio.CancelledError
            if mode == "http_error":
                error = RuntimeError("private token and document must not be exposed")
                error.status = 403
                raise error
        finally:
            closed.append(True)

    bot.session.stream_content = content
    adapter = TelegramAdapter(bot=bot, max_attachment_bytes=8)
    try:
        if mode == "success":
            assert await adapter._download("file-id") == b"data"
        elif mode == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await adapter._download("file-id")
        else:
            with pytest.raises(TelegramAttachmentDownloadError) as caught:
                await adapter._download("file-id")
            assert caught.value.stage == "read" and caught.value.status_code == 403
            assert "private" not in str(caught.value)
            assert caught.value.__context__ is None and caught.value.__cause__ is None
        assert closed == [True]
        assert calls[0]["chunk_size"] == 9 and calls[0]["raise_for_status"] is True
        assert calls[0]["url"] == bot.session.api.file_url(bot.token, "documents/test")
    finally:
        await adapter.aclose()
        await bot.session.close()


@pytest.mark.parametrize("size", [-1, True, "1", 1.5, float("nan"), 100])
async def test_telegram_corrupt_or_oversized_file_metadata_does_not_start_download(size):
    bot = Bot(TOKEN)
    bot.get_file = AsyncMock(return_value=SimpleNamespace(file_path="documents/test", file_size=size))
    bot.session.stream_content = lambda **kwargs: pytest.fail("Do not download invalid metadata")
    adapter = TelegramAdapter(bot=bot, max_attachment_bytes=8)
    try:
        with pytest.raises(TelegramAttachmentDownloadError):
            await adapter._download("file-id")
    finally:
        await adapter.aclose()
        await bot.session.close()


async def test_telegram_local_api_keeps_sdk_local_download_contract():
    bot = Bot(TOKEN)
    bot.session.api = TelegramAPIServer.from_base("http://local.invalid", is_local=True)
    bot.get_file = AsyncMock(return_value=SimpleNamespace(file_path="/local/documents/test", file_size=4))
    bot.session.stream_content = lambda **kwargs: pytest.fail("Local Bot API must not use remote stream")

    async def local_download(file_path, destination):
        assert file_path == "/local/documents/test"
        destination.write(b"data")

    bot.download_file = AsyncMock(side_effect=local_download)
    adapter = TelegramAdapter(bot=bot, max_attachment_bytes=8)
    try:
        assert await adapter._download("file-id") == b"data"
        bot.download_file.assert_awaited_once()
    finally:
        await adapter.aclose()
        await bot.session.close()


async def test_telegram_get_file_error_does_not_retain_bot_api_token_or_exception():
    bot = Bot(TOKEN)
    bot.get_file = AsyncMock(side_effect=RuntimeError("private token in SDK URL"))
    adapter = TelegramAdapter(bot=bot)
    try:
        with pytest.raises(TelegramAttachmentDownloadError) as caught:
            await adapter._download("file-id")
        assert caught.value.stage == "request"
        assert "private" not in str(caught.value)
        assert caught.value.__context__ is None
    finally:
        await adapter.aclose()
        await bot.session.close()


async def test_vk_closes_owned_api_even_if_download_client_close_fails():
    adapter = VKAdapter("dummy")
    adapter._downloads.aclose = AsyncMock(side_effect=RuntimeError("synthetic close failure"))
    adapter.api.http_client.close = AsyncMock()
    with pytest.raises(RuntimeError):
        await adapter.aclose()
    adapter.api.http_client.close.assert_awaited_once()


async def test_vk_message_registration_is_nonblocking_and_preserves_native_handlers():
    adapter, http = vk_adapter()
    http.request_text.return_value = '{"response":[{"id":20,"first_name":"Test"}]}'
    framework, native = AsyncMock(), AsyncMock()
    adapter.on_message(framework)

    @adapter.labeler.message()
    async def native_handler(message):
        await native(message)

    try:
        await adapter.native.process_event(
            {
                "type": "message_new",
                "object": {
                    "message": {
                        "id": 1,
                        "conversation_message_id": 1,
                        "date": 1,
                        "from_id": 20,
                        "peer_id": 10,
                        "text": "synthetic",
                        "attachments": [],
                        "fwd_messages": [],
                        "out": 0,
                        "version": 1,
                    },
                    "client_info": {},
                },
                "group_id": 1,
            },
            adapter.api,
        )
        framework.assert_awaited_once()
        native.assert_awaited_once()
    finally:
        await adapter.aclose()
