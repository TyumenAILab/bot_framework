from __future__ import annotations

import asyncio
import math
import threading
from dataclasses import dataclass
from io import BytesIO

import pdfplumber
import pypdfium2 as pdfium

from botkit.usage import current_context, usage_context

from .base import ExtractedContent
from .config import ExtractionConfig, ExtractionError, ExtractionLimitError
from .vision import ImageExtractor

# PDFium forbids concurrent calls even for independent documents. All native
# handles are created and closed inside this lock, including on exceptions.
_pdfium_lock = threading.Lock()


@dataclass(frozen=True)
class PageImage:
    page_number: int
    data: bytes
    width: int
    height: int
    mime_type: str = "image/png"


def _native_pages(data: bytes, config: ExtractionConfig) -> list[dict]:
    config.check_bytes(data)
    try:
        with pdfplumber.open(BytesIO(data), unicode_norm="NFC") as pdf:
            if not pdf.pages:
                raise ExtractionError("PDF has no pages")
            if len(pdf.pages) > config.max_pages:
                raise ExtractionLimitError("PDF page limit exceeded")
            pages = []
            text_count = 0
            cells = 0
            for index, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                text_count += len(text)
                if text_count > config.max_text_chars:
                    raise ExtractionLimitError("PDF text exceeds character limit")
                raw_tables = page.extract_tables() if text.strip() else []
                cells += sum(len(row) for table in raw_tables for row in table)
                if cells > config.max_cells:
                    raise ExtractionLimitError("PDF table cell limit exceeded")
                tables = [{"rows": [[cell or "" for cell in row] for row in table]} for table in raw_tables]
                area = max(1, page.width * page.height)
                image_area = sum(
                    max(0, i["x1"] - i["x0"]) * max(0, i["bottom"] - i["top"]) for i in page.images
                )
                warnings = []
                if not text.strip():
                    warnings.append("no_text_layer")
                elif (
                    sum(c.isalnum() for c in text) < config.pdf_min_text_chars
                    or "(cid:" in text
                    or "\ufffd" in text
                ):
                    warnings.append("suspicious_text_layer")
                if image_area / area >= 0.2:
                    warnings.append("page_contains_images")
                pages.append(
                    {
                        "page_number": index,
                        "text": text,
                        "tables": tables,
                        "warnings": warnings,
                        "method": "text",
                    }
                )
                page.close()
            return pages
    except ExtractionError:
        raise
    except Exception:
        raise ExtractionError("PDF is damaged, encrypted, or unsupported") from None


def _page_count(data: bytes, config: ExtractionConfig) -> int:
    config.check_bytes(data)
    try:
        with _pdfium_lock, pdfium.PdfDocument(data) as doc:
            count = len(doc)
            if not count:
                raise ExtractionError("PDF has no pages")
            if count > config.max_pages:
                raise ExtractionLimitError("PDF page limit exceeded")
            return count
    except ExtractionError:
        raise
    except Exception:
        raise ExtractionError("PDF is damaged, encrypted, or unsupported") from None


def _render_page(data: bytes, number: int, config: ExtractionConfig) -> PageImage:
    try:
        with _pdfium_lock, pdfium.PdfDocument(data) as doc:
            if len(doc) > config.max_pages:
                raise ExtractionLimitError("PDF page limit exceeded")
            if number < 1 or number > len(doc):
                raise ExtractionError("PDF page number out of range")
            page = doc[number - 1]
            try:
                width, height = page.get_size()
                if not all(math.isfinite(n) and n > 0 for n in (width, height)):
                    raise ExtractionError("Invalid PDF page dimensions")
                scale = min(config.render_dpi / 72, math.sqrt(config.render_max_pixels / (width * height)))
                # Account for rounding of the bitmap dimensions before allocation.
                while math.ceil(width * scale) * math.ceil(height * scale) > config.render_max_pixels:
                    scale *= 0.99
                bitmap = page.render(scale=scale, rev_byteorder=True)
                try:
                    with bitmap.to_pil() as image:
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        png = buffer.getvalue()
                        if len(png) > config.max_bytes:
                            raise ExtractionLimitError("Rendered page exceeds image size limit")
                        return PageImage(number, png, image.width, image.height)
                finally:
                    bitmap.close()
            finally:
                page.close()
    except ExtractionError:
        raise
    except Exception:
        raise ExtractionError("Could not render PDF page") from None


async def render_pdf(data: bytes, *, config: ExtractionConfig | None = None) -> list[PageImage]:
    """Explicit PDF -> PNG API. Does not call a model or write files."""
    config = config or ExtractionConfig()
    count = await asyncio.to_thread(_page_count, data, config)
    result, size = [], 0
    for number in range(1, count + 1):
        image = await asyncio.to_thread(_render_page, data, number, config)
        size += len(image.data)
        if size > config.max_rendered_bytes:
            raise ExtractionLimitError("Rendered PDF exceeds total image size limit")
        result.append(image)
    return result


class PDFExtractor:
    kind = "document"

    def __init__(self, vision: ImageExtractor | None = None, config: ExtractionConfig | None = None):
        self.vision, self.config = vision, config or ExtractionConfig()

    async def supports(self, mime_type: str) -> bool:
        return mime_type == "application/pdf"

    async def extract(self, data: bytes, mime_type: str) -> ExtractedContent:
        self.config.check_bytes(data)
        if self.config.pdf_mode == "vision" and self.vision is None:
            raise ExtractionError("PDF vision mode requires an LLM provider")
        # pdfplumber.pages eagerly builds page objects. Bound the page count
        # with the existing native renderer before the heavier text/table pass.
        count = await asyncio.to_thread(_page_count, data, self.config)
        pages = None
        initial_warnings = []
        if self.config.pdf_mode != "vision":
            try:
                pages = await asyncio.to_thread(_native_pages, data, self.config)
            except ExtractionLimitError:
                raise
            except ExtractionError:
                if self.config.pdf_mode == "text" or self.vision is None:
                    raise
                initial_warnings = ["native_parser_failed"]
        if pages is None:
            pages = [
                {
                    "page_number": n,
                    "text": "",
                    "tables": [],
                    "method": "text",
                    "warnings": list(initial_warnings),
                }
                for n in range(1, count + 1)
            ]
        needs_vision = [
            p
            for p in pages
            if self.config.pdf_mode == "vision" or (self.config.pdf_mode == "auto" and p["warnings"])
        ]
        if self.config.pdf_mode == "text":
            needs_vision = []
        if len(needs_vision) > self.config.max_vision_pages and self.vision:
            raise ExtractionLimitError("PDF exceeds vision page budget; split the document or use text mode")
        if needs_vision and self.config.pdf_mode == "vision" and self.vision is None:
            raise ExtractionError("PDF vision mode requires an LLM provider")
        rendered_bytes = 0
        for page in needs_vision:
            if self.vision is None:
                page["warnings"].append("vision_unavailable")
                continue
            image = await asyncio.to_thread(_render_page, data, page["page_number"], self.config)
            rendered_bytes += len(image.data)
            if rendered_bytes > self.config.max_rendered_bytes:
                raise ExtractionLimitError("Rendered PDF exceeds total image size limit")
            ctx = current_context()
            with usage_context(meta={**ctx.get("meta", {}), "extraction_page": page["page_number"]}):
                extracted = await self.vision.extract(image.data, image.mime_type)
            page.update(
                text=extracted.text or "",
                tables=extracted.structured_data["tables"],
                method="vision",
                metadata=extracted.metadata,
            )
            page["warnings"].extend(extracted.warnings)
            self.config.check_text("\n".join(p["text"] for p in pages))
        has_content = any(p["text"].strip() or p["tables"] for p in pages)
        text = "\n\n".join(f"[Page {p['page_number']}]\n{p['text']}" for p in pages) if has_content else ""
        self.config.check_text(text)
        warnings = [f"page:{p['page_number']}:{w}" for p in pages for w in p["warnings"]]
        if not has_content:
            warnings.append("empty_extraction")
        has_vision = any(p["method"] == "vision" for p in pages)
        confidence = "low" if warnings or has_vision else "high"
        return ExtractedContent(
            "structured" if has_vision else "text",
            text,
            {"pages": pages},
            warnings,
            confidence,
            {
                "page_count": len(pages),
                "pdf_mode": self.config.pdf_mode,
                "vision_pages": sum(p["method"] == "vision" for p in pages),
            },
        )
