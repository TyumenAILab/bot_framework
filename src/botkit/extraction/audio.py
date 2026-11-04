from __future__ import annotations

import asyncio

from botkit.llm.base import AudioTranscriptionProvider

from .base import ExtractedContent
from .config import ExtractionConfig, UnsupportedContentError


class AudioExtractor:
    kind = "audio"
    extensions = {
        "audio/wav": "wav",
        "audio/x-wav": "wav",
        "audio/mpeg": "mp3",
        "audio/ogg": "ogg",
        "audio/mp4": "m4a",
        "audio/flac": "flac",
        "audio/webm": "webm",
        "audio/aac": "aac",
        "audio/x-ms-wma": "wma",
    }

    def __init__(self, transcriber: AudioTranscriptionProvider, config: ExtractionConfig | None = None):
        self.transcriber, self.config = transcriber, config or ExtractionConfig()

    async def supports(self, mime_type: str) -> bool:
        return mime_type in self.extensions

    async def extract(self, data: bytes, mime_type: str) -> ExtractedContent:
        self.config.check_bytes(data)
        if mime_type not in self.extensions:
            raise UnsupportedContentError("Unsupported ASR audio type")
        async with asyncio.timeout(self.config.audio_timeout):
            response = await self.transcriber.atranscribe(
                data,
                mime_type=mime_type,
                filename="audio." + self.extensions[mime_type],
                timeout=self.config.audio_timeout,
            )
        self.config.check_text(response.response)
        warnings = ["audio_transcript_unverified"]
        if response.meta.get("postprocessed"):
            warnings.append("audio_transcript_postprocessed")
        elif "postprocess_reason" in response.meta:
            warnings.append("audio_transcript_postprocess_skipped")
        if not response.response.strip():
            warnings.append("empty_extraction")
        return ExtractedContent(
            "transcript",
            response.response,
            None,
            warnings,
            "low",
            {
                "method": "asr",
                "model": response.model,
                "provider": response.provider,
                **{
                    key: response.meta[key]
                    for key in (
                        "job_id",
                        "audio_duration_seconds",
                        "poll_count",
                        "upload_protocol",
                        "postprocessed",
                        "postprocess_reason",
                        "postprocessor_model",
                    )
                    if key in response.meta
                },
            },
        )
