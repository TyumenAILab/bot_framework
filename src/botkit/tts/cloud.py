"""Cloud TTS providers: Yandex SpeechKit and Google Cloud Text-to-Speech.

Credentials, text and audio are never included in error messages. TLS is verified,
redirects are refused and paid requests are never retried automatically.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import os
from typing import Any

import httpx

from .base import MAX_AUDIO_BYTES, SynthesizedAudio, TTSError

MIME = {"oggopus": "audio/ogg", "mp3": "audio/mpeg"}


class CloudTTS:
    def __init__(self, *, client: httpx.AsyncClient | None = None, timeout: float = 6):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("TTS timeout must be positive")
        self.timeout = timeout
        self.client = client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        self.owns_client = client is None

    async def request(self, url: str, *, limit: int = MAX_AUDIO_BYTES, **kwargs: Any) -> tuple[bytes, str]:
        try:
            async with self.client.stream(
                "POST", url, timeout=self.timeout, follow_redirects=False, **kwargs
            ) as response:
                if response.status_code != 200:
                    # Do not echo upstream bodies/headers which can contain credentials or input.
                    raise TTSError(f"TTS HTTP status {response.status_code}", retry=False)
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > limit:
                        raise TTSError("TTS response too large", retry=False)
                if not data:
                    raise TTSError("TTS response is empty", retry=False)
                return bytes(data), response.headers.get("content-type", "").split(";")[0].lower()
        except httpx.HTTPError:
            raise TTSError("TTS connection failed") from None

    async def aclose(self) -> None:
        if self.owns_client:
            await self.client.aclose()


class YandexTTS(CloudTTS):
    """SpeechKit REST v1: form data → MP3/OggOpus bytes (no streaming generation)."""

    endpoint = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize"

    def __init__(
        self,
        *,
        api_key: str = "",
        iam_token: str = "",
        folder_id: str = "",
        voice: str = "filipp",
        language: str = "ru-RU",
        audio_format: str = "oggopus",
        speed: float = 1.0,
        **kwargs: Any,
    ):
        if bool(api_key) == bool(iam_token):
            raise ValueError("Set exactly one YANDEX_TTS_API_KEY or YANDEX_TTS_IAM_TOKEN")
        if api_key and folder_id:
            raise ValueError("YANDEX_TTS_FOLDER_ID is only for user-account IAM authentication")
        if audio_format not in MIME or not math.isfinite(speed) or not 0.1 <= speed <= 3:
            raise ValueError("Invalid Yandex format or speed")
        super().__init__(**kwargs)
        self.authorization = ("Api-Key " + api_key) if api_key else ("Bearer " + iam_token)
        self.folder_id, self.voice, self.language = folder_id, voice, language
        self.audio_format, self.speed = audio_format, speed

    @classmethod
    def from_env(cls, **kwargs: Any) -> YandexTTS:
        return cls(
            api_key=os.getenv("YANDEX_TTS_API_KEY", ""),
            iam_token=os.getenv("YANDEX_TTS_IAM_TOKEN", ""),
            folder_id=os.getenv("YANDEX_TTS_FOLDER_ID", ""),
            voice=os.getenv("YANDEX_TTS_VOICE", "filipp"),
            language=os.getenv("YANDEX_TTS_LANGUAGE", "ru-RU"),
            audio_format=os.getenv("TTS_FORMAT", "oggopus"),
            speed=float(os.getenv("YANDEX_TTS_SPEED", "1")),
            **kwargs,
        )

    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        # Bound actual Unicode input, not URL-encoding overhead.
        if not text.strip() or len(text) > 5000 or len(text.encode("utf-8")) > 15000:
            raise ValueError("Yandex text limit exceeded or empty input")
        data = {
            "text": text,
            "voice": voice or self.voice,
            "lang": self.language,
            "format": self.audio_format,
            "speed": str(self.speed),
        }
        if self.folder_id:
            data["folderId"] = self.folder_id
        body, mime = await self.request(self.endpoint, headers={"Authorization": self.authorization}, data=data)
        if mime not in {MIME[self.audio_format], "application/ogg", "application/octet-stream"}:
            raise TTSError("Yandex did not return audio", retry=False)
        return SynthesizedAudio(body, MIME[self.audio_format])


class GoogleTTS(CloudTTS):
    """Google Cloud REST v1. OAuth bearer token or lazily refreshed ADC credentials."""

    endpoint = "https://texttospeech.googleapis.com/v1/text:synthesize"

    def __init__(
        self,
        *,
        access_token: str = "",
        credentials: Any = None,
        quota_project: str = "",
        voice: str = "",
        language: str = "ru-RU",
        audio_format: str = "oggopus",
        speed: float = 1.0,
        **kwargs: Any,
    ):
        if audio_format not in MIME or not math.isfinite(speed) or not 0.25 <= speed <= 2:
            raise ValueError("Invalid Google format or speed")
        super().__init__(**kwargs)
        self.access_token, self.credentials = access_token, credentials
        self.quota_project, self.voice, self.language = quota_project, voice, language
        self.audio_format, self.speed = audio_format, speed
        self.auth_lock = asyncio.Lock()

    @classmethod
    def from_env(cls, **kwargs: Any) -> GoogleTTS:
        return cls(
            access_token=os.getenv("GOOGLE_TTS_ACCESS_TOKEN", ""),
            quota_project=os.getenv("GOOGLE_TTS_QUOTA_PROJECT", ""),
            voice=os.getenv("GOOGLE_TTS_VOICE", ""),
            language=os.getenv("GOOGLE_TTS_LANGUAGE", "ru-RU"),
            audio_format=os.getenv("TTS_FORMAT", "oggopus"),
            speed=float(os.getenv("GOOGLE_TTS_SPEED", "1")),
            **kwargs,
        )

    def _refresh_credentials(self) -> str:
        import google.auth  # optional: pip install 'botkit[google]'
        from google.auth.transport.requests import Request

        if self.credentials is None:
            self.credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        if not self.credentials.valid:
            request = Request()
            # The worker remains bounded even if the coroutine is cancelled.
            try:
                self.credentials.refresh(
                    lambda *args, **kwargs: request(*args, **{**kwargs, "timeout": self.timeout})
                )
            finally:
                request.session.close()
        return self.credentials.token

    async def authorization(self) -> str:
        if self.access_token:
            return self.access_token
        async with self.auth_lock:
            try:
                return await asyncio.to_thread(self._refresh_credentials)
            except Exception:
                raise TTSError(
                    "Google TTS authentication failed; configure ADC or an OAuth token", retry=False
                ) from None

    async def synthesize(self, text: str, *, voice: str | None = None) -> SynthesizedAudio:
        if not text.strip() or len(text.encode("utf-8")) > 5000:
            raise ValueError("Google input must contain 1–5000 UTF-8 bytes")
        headers = {"Authorization": "Bearer " + await self.authorization()}
        project = self.quota_project or getattr(self.credentials, "quota_project_id", "")
        if project:
            headers["x-goog-user-project"] = project
        voice_config = {"languageCode": self.language}
        if name := (voice or self.voice):
            voice_config["name"] = name
        body, mime = await self.request(
            self.endpoint,
            limit=MAX_AUDIO_BYTES * 4 // 3 + 4096,
            headers=headers,
            json={
                "input": {"text": text},
                "voice": voice_config,
                "audioConfig": {
                    "audioEncoding": "OGG_OPUS" if self.audio_format == "oggopus" else "MP3",
                    "speakingRate": self.speed,
                },
            },
        )
        try:
            if mime != "application/json":
                raise ValueError()
            encoded = json.loads(body)["audioContent"]
            if not isinstance(encoded, str):
                raise ValueError()
            audio = base64.b64decode(encoded, validate=True)
            if not 0 < len(audio) <= MAX_AUDIO_BYTES:
                raise ValueError()
        except (ValueError, KeyError, TypeError, binascii.Error):
            raise TTSError("Google TTS returned invalid audioContent", retry=False) from None
        return SynthesizedAudio(audio, MIME[self.audio_format])
