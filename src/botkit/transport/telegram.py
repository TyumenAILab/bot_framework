from __future__ import annotations

import asyncio
from contextlib import aclosing
from io import BytesIO
from typing import Any

from .base import (
    Attachment,
    Button,
    CallbackQuery,
    IncomingMessage,
    _validate_attachment_metadata,
    split_text,
)
from .common import BaseAdapter


class _LimitedBuffer(BytesIO):
    def __init__(self, limit: int):
        _validate_attachment_metadata(None, limit)
        super().__init__()
        self.limit = limit

    def write(self, value: bytes) -> int:
        if self.tell() + memoryview(value).nbytes > self.limit:
            raise ValueError("Attachment exceeds the configured size limit")
        return super().write(value)

    def writelines(self, lines: Any) -> None:
        # BytesIO.writelines uses its C-level writer, bypassing an overridden write.
        for line in lines:
            self.write(line)


class TelegramAttachmentDownloadError(ValueError):
    """Fixed diagnostic that does not expose Bot API URLs/tokens or SDK errors."""

    def __init__(self, stage: str, reason: str, status_code: int | None = None):
        self.stage, self.reason, self.status_code = stage, reason, status_code
        super().__init__(
            f"Telegram attachment download failed: stage={stage} reason={reason} status={status_code}"
        )


class TelegramAdapter(BaseAdapter):
    platform = "telegram"
    message_limit = 4000

    def __init__(self, token: str | None = None, *, bot: Any = None, dispatcher: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        from aiogram import Bot, Dispatcher

        self.native = bot or Bot(token=token)
        self.dispatcher = dispatcher or Dispatcher()
        self._owns_bot = bot is None
        self._username: str | None = None
        self.dispatcher.message.register(self._receive)
        self.dispatcher.callback_query.register(self._receive_callback)

    async def _request(self, method: str, **params: Any) -> Any:
        from aiogram.exceptions import TelegramRetryAfter

        for attempt in range(3):
            try:
                return await getattr(self.native, method)(**params)
            except TelegramRetryAfter as exc:
                if attempt == 2:
                    raise
                await asyncio.sleep(exc.retry_after)

    async def call_api(self, method: str, **params: Any) -> Any:
        """Full aiogram Bot API: snake_case names and SDK parameter objects."""
        return await self._operation(method, lambda: self._request(method, **params))

    @staticmethod
    def _keyboard(buttons: list[list[Button]] | None) -> Any:
        if not buttons:
            return None
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text=b.text, callback_data=b.data, url=b.url) for b in row]
                for row in buttons
            ]
        )

    async def _send_one(self, chat_id: str, text: str, *, buttons: Any = None, **options: Any) -> str:
        if "parse_mode" in options or "entities" in options:
            raise ValueError("send() splits plain text; use call_api for formatted messages")
        if buttons is not None:
            options["reply_markup"] = self._keyboard(buttons)
        message = await self._request("send_message", chat_id=chat_id, text=text, parse_mode=None, **options)
        return str(message.message_id)

    async def _type(self, chat_id: str) -> None:
        await self._request("send_chat_action", chat_id=chat_id, action="typing")

    async def send_attachment(
        self,
        chat_id: str,
        data: bytes,
        *,
        filename: str = "file.bin",
        kind: str = "document",
        caption: str | None = None,
        **options: Any,
    ) -> str:
        from aiogram.types import BufferedInputFile

        if kind not in {"document", "photo", "audio", "video", "voice"}:
            raise ValueError(f"Unsupported Telegram attachment kind: {kind}")
        result = await self.call_api(
            f"send_{kind}",
            chat_id=chat_id,
            **{kind: BufferedInputFile(data, filename=filename)},
            caption=caption,
            **options,
        )
        return str(result.message_id)

    async def edit(
        self,
        chat_id: str,
        message_id: str,
        text: str,
        *,
        buttons: list[list[Button]] | None = None,
        **options: Any,
    ) -> Any:
        if len(split_text(text, self.message_limit)) != 1:
            raise ValueError("An edited message must fit in one message")
        return await self.call_api(
            "edit_message_text",
            chat_id=chat_id,
            message_id=int(message_id),
            text=text,
            reply_markup=self._keyboard(buttons),
            **options,
        )

    async def delete(self, chat_id: str, message_id: str) -> Any:
        return await self.call_api("delete_message", chat_id=chat_id, message_id=int(message_id))

    async def answer_callback(self, callback: CallbackQuery, text: str = "", **options: Any) -> Any:
        return await self.call_api(
            "answer_callback_query", callback_query_id=callback.callback_id, text=text, **options
        )

    async def _download(self, file_id: str) -> bytes:
        stage = "request"
        status = None
        try:
            _validate_attachment_metadata(None, self.max_attachment_bytes)
            file = await self._request("get_file", file_id=file_id)
            stage = "metadata"
            _validate_attachment_metadata(file.file_size, self.max_attachment_bytes)
            if not isinstance(file.file_path, str) or not file.file_path:
                raise TelegramAttachmentDownloadError(stage, "missing_path")
            with _LimitedBuffer(self.max_attachment_bytes) as buffer:
                stage = "read"
                if self.native.session.api.is_local:
                    # aiogram explicitly closes its local-file reader in finally.
                    await self.native.download_file(file.file_path, destination=buffer)
                else:
                    # Bot.download_file does not close a remote async generator
                    # when destination.write fails. Own its public session stream
                    # so size errors/cancellation release the HTTP response now,
                    # rather than waiting for generator garbage collection.
                    stream = self.native.session.stream_content(
                        url=self.native.session.api.file_url(self.native.token, file.file_path),
                        timeout=30,
                        chunk_size=min(65536, self.max_attachment_bytes + 1),
                        raise_for_status=True,
                    )
                    async with aclosing(stream):
                        async for chunk in stream:
                            buffer.write(chunk)
                return buffer.getvalue()
        except TelegramAttachmentDownloadError:
            raise
        except Exception as exc:
            candidate_status = getattr(exc, "status", None)
            if type(candidate_status) is int and 100 <= candidate_status <= 599:
                status = candidate_status
            reason = "invalid_metadata" if stage == "metadata" else "transport_error"
            if isinstance(exc, ValueError) and str(exc) == "Attachment exceeds the configured size limit":
                reason = "size_limit"
            safe_error = TelegramAttachmentDownloadError(stage, reason, status)
        raise safe_error from None

    def normalize(self, message: Any) -> IncomingMessage:
        attachments = []
        for kind in ("animation", "video_note", "document", "photo", "audio", "voice", "video", "sticker"):
            value = getattr(message, kind, None)
            if not value:
                continue
            if kind == "document" and getattr(message, "animation", None):
                # Telegram repeats an animation as document for old clients.
                continue
            if kind == "photo":
                value = value[-1]
            default_mime = {
                "photo": "image/jpeg",
                "voice": "audio/ogg",
                "sticker": "image/webp",
                "animation": "video/mp4",
                "video_note": "video/mp4",
            }
            attachments.append(
                Attachment(
                    kind="audio"
                    if kind == "voice"
                    else "video"
                    if kind in {"animation", "video_note"}
                    else kind,
                    mime_type=getattr(value, "mime_type", None)
                    or default_mime.get(kind, "application/octet-stream"),
                    filename=getattr(value, "file_name", None),
                    size=getattr(value, "file_size", None),
                    platform_id=value.file_id,
                    max_bytes=self.max_attachment_bytes,
                    reader=lambda file_id=value.file_id: self._download(file_id),
                )
            )
        user = message.from_user
        return IncomingMessage(
            platform=self.platform,
            user_id=str(user.id) if user else str(message.chat.id),
            chat_id=str(message.chat.id),
            text=message.text or message.caption or "",
            attachments=attachments,
            platform_username=user.username if user else None,
            platform_display_name=user.full_name if user else None,
            message_id=str(message.message_id),
            raw=message,
        )

    async def _receive(self, message: Any) -> None:
        from aiogram.dispatcher.event.bases import SkipHandler

        text = message.text or ""
        first = text.split(maxsplit=1)[0] if text else ""
        if first.startswith("/") and "@" in first and self._username:
            if first.split("@", 1)[1].casefold() != self._username.casefold():
                raise SkipHandler
        await self.dispatch(self.normalize(message))
        # aiogram observers stop at the first successful handler. This adapter
        # must not consume an update before later, user-registered SDK handlers.
        raise SkipHandler

    async def _receive_callback(self, query: Any) -> None:
        from aiogram.dispatcher.event.bases import SkipHandler

        # Inline-mode callbacks without a chat remain accessible through native dispatcher.
        if query.message is None:
            raise SkipHandler
        await self.dispatch_callback(
            CallbackQuery(
                platform=self.platform,
                callback_id=query.id,
                user_id=str(query.from_user.id),
                chat_id=str(query.message.chat.id),
                data=query.data or "",
                message_id=str(query.message.message_id),
                raw=query,
            )
        )
        raise SkipHandler

    async def run(self) -> None:
        try:
            self._username = (await self._request("get_me")).username
            await self.dispatcher.start_polling(
                self.native, close_bot_session=False, tasks_concurrency_limit=64
            )
        finally:
            await self.aclose()

    async def aclose(self) -> None:
        try:
            await super().aclose()
        finally:
            if self._owns_bot:
                await self.native.session.close()
