from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any
from uuid import uuid4

from botkit.usage import UsageEvent, UsageTracker, safe_log, usage_context

from .base import (
    Button,
    CallbackHandler,
    CallbackQuery,
    IncomingMessage,
    MessageHandler,
    TypingHandle,
    split_text,
)

logger = logging.getLogger(__name__)


class BackgroundTypingHandle:
    def __init__(self, callback: Callable[[], Awaitable[Any]], interval: float = 4.0):
        self._callback = callback
        self._interval = interval
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                await self._callback()
            except Exception as exc:
                logger.warning("Typing indicator failed (%s)", type(exc).__name__)
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def __aenter__(self) -> BackgroundTypingHandle:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()


class BaseAdapter:
    platform = "base"
    message_limit = 4000

    def __init__(
        self,
        *,
        tracker: UsageTracker | None = None,
        bot_id: str = "bot",
        max_attachment_bytes: int = 20 * 1024 * 1024,
    ):
        if type(max_attachment_bytes) is not int or max_attachment_bytes <= 0:
            raise ValueError("max_attachment_bytes must be a positive integer")
        self.tracker = tracker
        self.bot_id = bot_id
        self.max_attachment_bytes = max_attachment_bytes
        self._handler: MessageHandler | None = None
        self._commands: dict[str, MessageHandler] = {}
        self._callback: CallbackHandler | None = None
        self._typing: set[TypingHandle] = set()
        self._active: set[asyncio.Task] = set()

    def on_message(self, handler: MessageHandler) -> MessageHandler:
        self._handler = handler
        return handler

    def on_command(self, command: str, handler: MessageHandler) -> MessageHandler:
        key = command.lstrip("/").casefold()
        if not key or " " in key:
            raise ValueError("Command must be one word")
        if key in self._commands:
            raise ValueError(f"Command already registered: {key}")
        self._commands[key] = handler
        return handler

    def on_callback(self, handler: CallbackHandler) -> CallbackHandler:
        self._callback = handler
        return handler

    async def dispatch(self, message: IncomingMessage) -> None:
        token = message.text.strip().split(maxsplit=1)
        command = token[0][1:].split("@", 1)[0].casefold() if token and token[0].startswith("/") else ""
        handler = self._commands.get(command, self._handler)
        if handler:
            await self._dispatch_event(message, handler, "incoming_message")

    async def dispatch_callback(self, callback: CallbackQuery) -> None:
        if self._callback:
            await self._dispatch_event(callback, self._callback, "callback")

    async def _dispatch_event(self, value: Any, handler: Any, operation: str) -> None:
        task = asyncio.current_task()
        if task:
            self._active.add(task)
        try:
            with usage_context(
                bot_id=self.bot_id,
                platform=self.platform,
                user_id=value.user_id,
                chat_id=value.chat_id,
                request_id=uuid4().hex,
                meta={
                    "platform_username": getattr(value, "platform_username", None),
                    "platform_display_name": getattr(value, "platform_display_name", None),
                },
            ):
                await self._operation(operation, lambda: handler(value))
        finally:
            self._active.discard(task)

    async def _operation(self, name: str, call: Callable[[], Awaitable[Any]], **meta: Any) -> Any:
        event = UsageEvent(
            kind="transport",
            operation=name,
            platform=self.platform,
            bot_id=self.bot_id,
            meta=meta,
            cost_status="NOT_APPLICABLE",
        )
        start = time.perf_counter()
        try:
            return await call()
        except asyncio.CancelledError:
            event.status, event.error_type = "cancelled", "CancelledError"
            raise
        except Exception as exc:
            event.status, event.error_type = "error", type(exc).__name__
            raise
        finally:
            event.latency_ms = (time.perf_counter() - start) * 1000
            await safe_log(self.tracker, event)

    async def send(
        self, chat_id: str, text: str, *, buttons: list[list[Button]] | None = None, **options: Any
    ) -> list[str]:
        chunks = split_text(text, self.message_limit)
        if len(chunks) > 1 and "random_id" in options:
            raise ValueError("Explicit random_id is only supported for one message; chunks need distinct IDs")
        ids = []
        for index, chunk in enumerate(chunks):

            async def send_chunk(chunk: str = chunk, index: int = index) -> Any:
                return await self._send_one(
                    chat_id, chunk, buttons=buttons if index == len(chunks) - 1 else None, **options
                )

            result = await self._operation("send", send_chunk, chat_id=chat_id, characters=len(chunk))
            ids.append(str(result))
        return ids

    async def _send_one(self, chat_id: str, text: str, **options: Any) -> str:
        raise NotImplementedError

    async def send_status(self, chat_id: str, text: str) -> None:
        await self.send(chat_id, text)

    async def start_typing(self, chat_id: str) -> TypingHandle:
        handle = BackgroundTypingHandle(lambda: self._operation("typing", lambda: self._type(chat_id)))
        self._typing.add(handle)
        handle._task.add_done_callback(lambda _: self._typing.discard(handle))
        return handle

    async def _type(self, chat_id: str) -> None:
        raise NotImplementedError

    async def aclose(self) -> None:
        for handle in list(self._typing):
            await handle.stop()
        active = [t for t in self._active if t is not asyncio.current_task()]
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
