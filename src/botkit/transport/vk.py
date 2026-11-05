from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import secrets
from collections import OrderedDict
from typing import Any
from urllib.parse import urlsplit

import httpx

from .base import Attachment, Button, CallbackQuery, IncomingMessage, split_text
from .common import BaseAdapter

logger = logging.getLogger(__name__)


class VKAttachmentDownloadError(ValueError):
    """Safe download diagnostic; never retains a signed URL or HTTP response."""

    def __init__(self, stage: str, reason: str, status_code: int | None = None, *, violation: str = "unknown"):
        self.stage = stage
        self.reason = reason
        self.status_code = status_code
        # Fixed categorical detail only. Never retain a URL, hostname or headers.
        self.violation = violation if violation in {
            "unknown", "malformed", "fragment", "userinfo", "scheme", "port", "host"
        } else "unknown"
        super().__init__(f"VK attachment download failed: stage={stage} reason={reason} status={status_code}")


# VK API attachment URLs and their CDN redirects form one explicit trust
# boundary. Suffixes must match a DNS label, never a textual substring. TLS and
# the system DNS resolver are trusted; this is not DNS pinning. A caller-supplied
# HTTP client (including its transport/event hooks) is likewise trusted code.
_DOWNLOAD_HOSTS = (
    "vk.com",
    "vk.ru",
    "userapi.com",
    "vkuser.net",
    "vkuserphoto.ru",
    "vkuseraudio.net",
    "vk-cdn.net",
    "vk-cdn.me",
)
_DOWNLOAD_REDIRECTS = {301, 302, 303, 307, 308}
_DOWNLOAD_MAX_REDIRECTS = 5
_DOWNLOAD_MAX_URL_LENGTH = 16384


def _download_url(value: str, *, base: httpx.URL | None = None, status_code: int | None = None) -> httpx.URL:
    stage = "validate" if base is None else "redirect"
    violation = "malformed"
    invalid = not isinstance(value, str) or not value or len(value) > _DOWNLOAD_MAX_URL_LENGTH
    if not invalid:
        # Reject parser ambiguities before joining/canonicalising a Location.
        invalid = any(ord(char) <= 32 or ord(char) == 127 for char in value) or "\\" in value
    if not invalid:
        try:
            # URL fragments are client-side only and never belong in an HTTP
            # request. Remove them *before* parsing or resolving; do not let a
            # fragment influence the scheme, authority or allowlist checks.
            request_value = value.split("#", 1)[0]
            invalid = not request_value and base is None
            has_userinfo = "@" in urlsplit(request_value).netloc  # includes empty userinfo
            url = httpx.URL(request_value) if base is None else base.join(request_value)
            host = url.host.lower()
            if invalid:
                violation = "malformed"
            elif has_userinfo or url.userinfo:
                invalid, violation = True, "userinfo"
            elif url.scheme != "https" or not url.is_absolute_url:
                invalid, violation = True, "scheme"
            elif url.port not in {None, 443}:
                invalid, violation = True, "port"
            elif not any(host == domain or host.endswith("." + domain) for domain in _DOWNLOAD_HOSTS):
                invalid, violation = True, "host"
        except (ValueError, httpx.InvalidURL):
            invalid = True
    if invalid:
        raise VKAttachmentDownloadError(stage, "unsafe_url", status_code, violation=violation) from None
    return url


def _message_id(value: str, *, local_key: str, global_key: str) -> dict[str, int]:
    encoded = str(value)
    local = encoded.startswith("cmid:")
    number = encoded[5:] if local else encoded
    if not number or len(number) > 20 or not number.isascii() or not number.isdecimal() or int(number) <= 0:
        raise ValueError("VK message ID must be a positive numeric ID or cmid:N")
    return {local_key if local else global_key: int(number)}


class VKAdapter(BaseAdapter):
    platform = "vk"
    message_limit = 4096

    def __init__(
        self,
        token: str | None = None,
        *,
        bot: Any = None,
        download_client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        from vkbottle import Bot

        self.native = bot or Bot(token=token)
        self.api = self.native.api
        self.labeler = self.native.labeler
        self._owns_bot = bot is None
        self._owns_downloads = download_client is None
        self._downloads = download_client or httpx.AsyncClient(
            timeout=60, follow_redirects=False, trust_env=False
        )
        self._users: OrderedDict[int, tuple[str | None, str | None]] = OrderedDict()
        self.labeler.message(blocking=False)(self._receive)
        self.labeler.raw_event("message_event", blocking=False)(self._receive_callback)

    async def call_api(self, method: str, **params: Any) -> Any:
        """Full VK API by dotted method name; .api also exposes typed SDK methods."""

        async def call() -> Any:
            from vkbottle import VKAPIError

            for attempt in range(3):
                try:
                    response = await self.api.request(method, params)
                    return response["response"]
                except VKAPIError as exc:
                    if exc.code not in {6, 9} or attempt == 2:
                        raise
                    await asyncio.sleep(2**attempt)

        return await self._operation(method, call)

    @staticmethod
    def _keyboard(buttons: list[list[Button]]) -> str:
        rows = []
        for row in buttons:
            items = []
            for button in row:
                if button.url:
                    action = {"type": "open_link", "label": button.text, "link": button.url}
                else:
                    action = {
                        "type": "callback",
                        "label": button.text,
                        "payload": json.dumps({"data": button.data}, ensure_ascii=False),
                    }
                items.append({"action": action})
            rows.append(items)
        return json.dumps({"inline": True, "buttons": rows}, ensure_ascii=False)

    async def _send_one(self, chat_id: str, text: str, *, buttons: Any = None, **options: Any) -> str:
        if buttons is not None:
            options["keyboard"] = self._keyboard(buttons)
        options.setdefault("random_id", secrets.randbelow(2**31 - 1) + 1)
        result = await self.call_api("messages.send", peer_id=int(chat_id), message=text, **options)
        return str(result)

    async def _type(self, chat_id: str) -> None:
        await self.call_api("messages.setActivity", peer_id=int(chat_id), type="typing")

    async def send_attachment(
        self,
        chat_id: str,
        data: bytes,
        *,
        filename: str = "file.bin",
        kind: str = "document",
        caption: str = "",
        **options: Any,
    ) -> str:
        from vkbottle import DocMessagesUploader, PhotoMessageUploader

        if kind == "photo":
            uploader = PhotoMessageUploader(self.api, attachment_name=filename)
        elif kind in {"document", "audio", "video"}:
            # Generic audio/video files are documents in the portable API. Use
            # native uploaders for playable VK music/video (different permissions).
            uploader = DocMessagesUploader(self.api)
        else:
            raise ValueError(f"Unsupported VK attachment kind: {kind}")
        upload_params = {"title": filename} if kind != "photo" else {}
        attachment = await self._operation(
            "upload", lambda: uploader.upload(data, peer_id=int(chat_id), **upload_params)
        )
        return str(
            await self.call_api(
                "messages.send",
                peer_id=int(chat_id),
                message=caption,
                attachment=attachment,
                random_id=secrets.randbelow(2**31 - 1) + 1,
                **options,
            )
        )

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
        if buttons is not None:
            options["keyboard"] = self._keyboard(buttons)
        return await self.call_api(
            "messages.edit",
            peer_id=int(chat_id),
            message=text,
            **_message_id(message_id, local_key="cmid", global_key="message_id"),
            **options,
        )

    async def delete(self, chat_id: str, message_id: str, **options: Any) -> Any:
        return await self.call_api(
            "messages.delete",
            peer_id=int(chat_id),
            delete_for_all=1,
            **_message_id(message_id, local_key="cmids", global_key="message_ids"),
            **options,
        )

    async def answer_callback(self, callback: CallbackQuery, text: str = "", **options: Any) -> Any:
        event_data = json.dumps({"type": "show_snackbar", "text": text}, ensure_ascii=False) if text else None
        return await self.call_api(
            "messages.sendMessageEventAnswer",
            event_id=callback.callback_id,
            user_id=int(callback.user_id),
            peer_id=int(callback.chat_id),
            event_data=event_data,
            **options,
        )

    async def _download(self, url: str) -> bytes:
        current = _download_url(url)
        visited = {str(current)}
        stage, status = "request", None
        try:
            for hop in range(_DOWNLOAD_MAX_REDIRECTS + 1):
                stage, status = "request", None
                # Do not use build_request/stream: those merge the injected
                # client's default Authorization, Cookie and Referer headers.
                # Start clean on *every* hop, even same-origin redirects.
                request = httpx.Request(
                    "GET", current, headers={"Accept": "*/*", "Accept-Encoding": "identity"}
                )
                response = await self._downloads.send(request, stream=True, follow_redirects=False, auth=None)
                try:
                    status = response.status_code
                    if status in _DOWNLOAD_REDIRECTS:
                        stage = "redirect"
                        locations = response.headers.get_list("location")
                        if len(locations) != 1 or not locations[0]:
                            raise VKAttachmentDownloadError(stage, "invalid_location", status)
                        if hop >= _DOWNLOAD_MAX_REDIRECTS:
                            raise VKAttachmentDownloadError(stage, "redirect_limit", status)
                        target = _download_url(locations[0], base=current, status_code=status)
                        if str(target) in visited:
                            raise VKAttachmentDownloadError(stage, "redirect_loop", status)
                        visited.add(str(target))
                        current = target
                        continue
                    if not 200 <= status < 300:
                        # raise_for_status would retain/request-print the entire
                        # signed URL, response and possibly redirect history.
                        raise VKAttachmentDownloadError(stage, "http_status", status)
                    stage = "read"
                    # HTTPX decodes compression before yielding aiter_bytes;
                    # a tiny compressed chunk can allocate far beyond our cap.
                    # We requested identity, so reject unexpected compression
                    # before reading any response body instead of decompressing.
                    encoding = response.headers.get("content-encoding", "identity").strip().casefold()
                    if encoding != "identity":
                        raise VKAttachmentDownloadError(stage, "unsupported_encoding", status)
                    stage = "size"
                    declared_length = response.headers.get("content-length")
                    if declared_length is not None:
                        if (
                            not 1 <= len(declared_length) <= 20
                            or not declared_length.isascii()
                            or not declared_length.isdecimal()
                        ):
                            raise VKAttachmentDownloadError(stage, "invalid_length", status)
                        if int(declared_length) > self.max_attachment_bytes:
                            raise VKAttachmentDownloadError(stage, "size_limit", status)
                    result = bytearray()
                    stage = "read"
                    async for chunk in response.aiter_bytes():
                        if len(result) + len(chunk) > self.max_attachment_bytes:
                            raise VKAttachmentDownloadError("size", "size_limit", status)
                        result.extend(chunk)
                    return bytes(result)
                finally:
                    await response.aclose()
        except VKAttachmentDownloadError:
            raise
        except Exception:
            # Raise outside the except suite: do not retain the raw exception
            # as __context__, since it can contain URL, headers or document data.
            safe_error = VKAttachmentDownloadError(stage, "transport_error", status)
        raise safe_error from None

    async def _identity(self, user_id: int) -> tuple[str | None, str | None]:
        if user_id <= 0:
            return None, None
        if user_id in self._users:
            self._users.move_to_end(user_id)
            return self._users[user_id]
        try:
            users = await self.call_api("users.get", user_ids=str(user_id), fields="screen_name")
            user = users[0]
            identity = (
                user.get("screen_name"),
                " ".join(filter(None, [user.get("first_name"), user.get("last_name")])),
            )
            self._users[user_id] = identity
            if len(self._users) > 512:
                self._users.popitem(last=False)
            return identity
        except Exception as exc:
            logger.warning("VK user profile unavailable (%s)", type(exc).__name__)
            return None, None

    async def normalize(self, message: Any) -> IncomingMessage:
        attachments = []
        for item in message.attachments or []:
            kind = item.type.value if hasattr(item.type, "value") else item.type
            value = getattr(item, kind, None)
            url, mime, filename, size = None, "application/octet-stream", None, None
            if kind == "doc" and value:
                kind, url, filename, size = "document", value.url, value.title, value.size
                mime = mimetypes.guess_type(filename or "")[0] or mime
            elif kind == "photo" and value and value.sizes:
                url = max(value.sizes, key=lambda p: p.width * p.height).url
                mime = "image/jpeg"
            elif kind == "audio_message" and value:
                kind, url, mime = (
                    "audio",
                    value.link_ogg or value.link_mp3,
                    "audio/ogg" if value.link_ogg else "audio/mpeg",
                )
            elif kind == "audio" and value:
                url, mime = value.url, "audio/mpeg"
            if url:
                attachments.append(
                    Attachment(
                        kind=kind,
                        mime_type=mime,
                        filename=filename,
                        size=size,
                        reader=lambda url=url: self._download(url),
                        max_bytes=self.max_attachment_bytes,
                        platform_id=str(getattr(value, "id", "")),
                    )
                )
        username, name = await self._identity(message.from_id)
        return IncomingMessage(
            platform=self.platform,
            user_id=str(message.from_id),
            chat_id=str(message.peer_id),
            text=message.text or "",
            attachments=attachments,
            platform_username=username,
            platform_display_name=name,
            message_id=str(message.id),
            raw=message,
        )

    async def _receive(self, message: Any) -> None:
        await self.dispatch(await self.normalize(message))

    async def _receive_callback(self, event: dict[str, Any]) -> None:
        value = event["object"]
        payload = value.get("payload") or {}
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = {"data": payload}
        if not isinstance(payload, dict):
            payload = {}
        data = payload.get("data", "")
        if not isinstance(data, str):
            data = ""
        local_id = value.get("conversation_message_id")
        message_id = None
        if local_id is not None:
            message_id = "cmid:" + str(
                _message_id(str(local_id), local_key="cmid", global_key="cmid")["cmid"]
            )
        await self.dispatch_callback(
            CallbackQuery(
                platform=self.platform,
                callback_id=value["event_id"],
                user_id=str(value["user_id"]),
                chat_id=str(value["peer_id"]),
                data=data,
                message_id=message_id,
                raw=event,
            )
        )

    async def run(self) -> None:
        # Own the task lifetime around SDK polling; bound in-flight updates and
        # retrieve every exception instead of leaving fire-and-forget tasks.
        pending: set[asyncio.Task] = set()
        semaphore = asyncio.Semaphore(64)

        async def process(update: dict[str, Any]) -> None:
            try:
                await self.native.process_event(update, self.api)
            except Exception as exc:
                logger.error("VK event handler failed (%s)", type(exc).__name__)
            finally:
                semaphore.release()

        try:
            async for event in self.native.polling.listen():
                for update in event.get("updates", []):
                    await semaphore.acquire()
                    task = asyncio.create_task(process(update))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
        finally:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            await self.aclose()

    async def aclose(self) -> None:
        try:
            await super().aclose()
        finally:
            try:
                if self._owns_downloads:
                    await self._downloads.aclose()
            finally:
                if self._owns_bot:
                    await self.api.http_client.close()
