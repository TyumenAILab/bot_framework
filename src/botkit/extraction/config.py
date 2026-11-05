from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

PDFMode = Literal["auto", "text", "vision"]


class ExtractionError(ValueError):
    """Safe, user-facing error; never contains document contents or credentials."""


class UnsupportedContentError(ExtractionError):
    pass


class ExtractionLimitError(ExtractionError):
    pass


@dataclass(frozen=True)
class ExtractionConfig:
    pdf_mode: PDFMode = "auto"
    max_bytes: int = 20 * 1024 * 1024
    max_pages: int = 30
    max_vision_pages: int = 10
    max_text_chars: int = 200_000
    max_image_pixels: int = 25_000_000
    render_max_pixels: int = 4_000_000
    render_dpi: int = 144
    max_rendered_bytes: int = 50 * 1024 * 1024
    max_archive_bytes: int = 100 * 1024 * 1024
    max_archive_entries: int = 2000
    max_cells: int = 50_000
    pdf_min_text_chars: int = 30
    vision_timeout: float = 180
    vision_max_tokens: int = 8192
    audio_timeout: float = 600

    def __post_init__(self):
        if self.pdf_mode not in {"auto", "text", "vision"}:
            raise ValueError("pdf_mode must be auto, text or vision")
        for name, value in vars(self).items():
            if name == "pdf_mode":
                continue
            if name in {"vision_timeout", "audio_timeout"}:
                try:
                    valid = type(value) in {int, float} and math.isfinite(value) and value > 0
                except OverflowError:
                    valid = False
            else:
                valid = type(value) is int and value > 0
            if not valid:
                raise ValueError(
                    f"{name} must be a positive finite {'number' if name.endswith('timeout') else 'integer'}"
                )

    @classmethod
    def from_env(cls) -> ExtractionConfig:
        return cls(
            pdf_mode=os.getenv("EXTRACTION_PDF_MODE", "auto"),
            max_bytes=int(os.getenv("MAX_ATTACHMENT_BYTES", str(20 * 1024 * 1024))),
            max_pages=int(os.getenv("EXTRACTION_MAX_PAGES", "30")),
            max_vision_pages=int(os.getenv("EXTRACTION_MAX_VISION_PAGES", "10")),
            max_text_chars=int(os.getenv("EXTRACTION_MAX_TEXT_CHARS", "200000")),
            render_dpi=int(os.getenv("EXTRACTION_RENDER_DPI", "144")),
            vision_timeout=float(os.getenv("EXTRACTION_VISION_TIMEOUT", "180")),
            audio_timeout=float(os.getenv("EXTRACTION_AUDIO_TIMEOUT", "600")),
        )

    def check_bytes(self, data: bytes) -> None:
        if not isinstance(data, bytes):
            raise ExtractionError("Attachment data must be bytes")
        if not data:
            raise ExtractionError("Empty attachment")
        if len(data) > self.max_bytes:
            raise ExtractionLimitError("Attachment exceeds extraction size limit")

    def check_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise ExtractionError("Extracted content must be text")
        if len(text) > self.max_text_chars:
            raise ExtractionLimitError("Extracted text exceeds character limit")
