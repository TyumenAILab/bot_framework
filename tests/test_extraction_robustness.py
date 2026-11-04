"""Synthetic, offline regressions for extraction limits and failure boundaries."""

import asyncio
import builtins
import json
import time
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zipfile import ZIP_STORED, ZipFile

import pytest
from docx import Document
from PIL import Image
from reportlab.pdfgen import canvas

from botkit.extraction import (
    AudioExtractor,
    ContentExtractionService,
    ExtractedContent,
    ExtractionConfig,
    ExtractionError,
    ExtractionLimitError,
    ImageExtractor,
    PDFExtractor,
    UnsupportedContentError,
)
from botkit.extraction.documents import DOCX, PPTX, XLSX, inspect_archive
from botkit.extraction.pdf import PageImage
from botkit.extraction.vision import vision_payload
from botkit.llm import InvalidLLMResponse, LLMResponse


def make_pdf(pages=1, *, blank=False):
    buffer = BytesIO()
    document = canvas.Canvas(buffer)
    for _ in range(pages):
        if not blank:
            document.drawString(10, 10, "Synthetic readable document, no private content.")
        document.showPage()
    document.save()
    return buffer.getvalue()


def make_docx():
    buffer = BytesIO()
    document = Document()
    document.add_paragraph("Synthetic paragraph")
    document.save(buffer)
    return buffer.getvalue()


def make_png():
    buffer = BytesIO()
    with Image.new("RGB", (4, 4), "white") as image:
        image.save(buffer, "PNG")
    return buffer.getvalue()


def archive_xml(content, member="word/document.xml"):
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_STORED) as archive:
        archive.writestr(member, content)
    return buffer.getvalue()


INTEGER_LIMITS = [
    name
    for name in ExtractionConfig.__dataclass_fields__
    if name not in {"pdf_mode", "vision_timeout", "audio_timeout"}
]


@pytest.mark.parametrize("field", INTEGER_LIMITS)
@pytest.mark.parametrize("value", [True, 1.5, float("nan"), float("inf"), float("-inf")])
def test_count_limits_reject_non_integer_and_non_finite_values(field, value):
    with pytest.raises(ValueError):
        ExtractionConfig(**{field: value})


@pytest.mark.parametrize("field", ["vision_timeout", "audio_timeout"])
@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf"), 10**1000])
def test_provider_timeouts_must_be_positive_finite_numbers(field, value):
    with pytest.raises(ValueError):
        ExtractionConfig(**{field: value})


@pytest.mark.parametrize("data", [None, "SYNTHETIC_TEXT", bytearray(b"x"), 123])
async def test_invalid_byte_input_has_a_safe_extraction_error(data):
    with pytest.raises(ExtractionError, match="must be bytes"):
        await ContentExtractionService().extract_bytes(data, "text/plain")


@pytest.mark.parametrize("data", [b"", b"oversized"])
async def test_byte_limits_precede_mime_inspection_and_heavy_parsing(monkeypatch, data):
    def forbidden(*args):
        pytest.fail("Rejected bytes must not reach MIME or parser work")

    monkeypatch.setattr("botkit.extraction.service.detect_mime", forbidden)
    with pytest.raises(ExtractionError):
        await ContentExtractionService(config=ExtractionConfig(max_bytes=2)).extract_bytes(
            data, "application/pdf"
        )


async def test_missing_mime_can_use_filename_and_signatures():
    result = await ContentExtractionService().extract_bytes(b"Readable text", None, filename="note.TXT")
    assert result.text == "Readable text" and result.metadata["mime_type"] == "text/plain"
    assert "mime_type_inferred_from_content_or_filename" in result.warnings
    result = await ContentExtractionService().extract_bytes(make_docx(), "text/plain", filename="wrong.txt")
    assert "Synthetic paragraph" in result.text and result.metadata["mime_type"] == DOCX


@pytest.mark.parametrize("mime,filename", [(123, None), ("application/octet-stream", 123)])
async def test_invalid_metadata_does_not_escape_as_attribute_or_path_error(mime, filename):
    with pytest.raises(ExtractionError):
        await ContentExtractionService().extract_bytes(b"Readable text", mime, filename=filename)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-32"])
async def test_office_dtd_and_entities_are_rejected_before_document_parser(monkeypatch, encoding):
    content = (
        f'<?xml version="1.0" encoding="{encoding}"?>'
        '<!DOCTYPE root [<!ENTITY secret SYSTEM "file:///PRIVATE_SENTINEL">]>'
        "<root>&secret;</root>"
    ).encode(encoding)

    def forbidden(*args):
        pytest.fail("Unsafe XML cannot reach an Office parser")

    monkeypatch.setattr("botkit.extraction.documents.OfficeExtractor._docx", forbidden)
    with pytest.raises(ExtractionError) as caught:
        await ContentExtractionService().extract_bytes(archive_xml(content), DOCX)
    assert "SENTINEL" not in str(caught.value)


def test_unsupported_archive_compression_is_a_safe_extraction_error():
    data = bytearray(archive_xml(b"<root/>"))
    data[8:10] = (99).to_bytes(2, "little")
    central = data.index(b"PK\x01\x02")
    data[central + 10 : central + 12] = (99).to_bytes(2, "little")
    with pytest.raises(ExtractionError, match="Invalid Office archive"):
        inspect_archive(bytes(data), ExtractionConfig())


@pytest.mark.parametrize("data", [b"\xff\xfeA", b"\xef\xbb\xbf\xff", b"\xff\xfe\x00\x00A"])
async def test_invalid_bom_text_does_not_silently_turn_into_cp1251(data):
    with pytest.raises(ExtractionError, match="byte-order encoding"):
        await ContentExtractionService().extract_bytes(data, "text/plain")


async def test_missing_docx_dependency_is_not_misreported_as_document_corruption(monkeypatch):
    data = make_docx()
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "docx" or name.startswith("docx."):
            raise ImportError("PRIVATE_IMPORT_SENTINEL")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(UnsupportedContentError, match="python-docx") as caught:
        await ContentExtractionService().extract_bytes(data, DOCX)
    assert "SENTINEL" not in str(caught.value)


async def test_docx_text_limit_stops_iteration_before_later_blocks(monkeypatch):
    def blocks():
        yield SimpleNamespace(text="123456")
        pytest.fail("Text limit must stop before the next block")

    monkeypatch.setattr("docx.Document", lambda _: SimpleNamespace(iter_inner_content=blocks))
    with pytest.raises(ExtractionLimitError, match="DOCX text"):
        await ContentExtractionService(config=ExtractionConfig(max_text_chars=5)).extract_bytes(
            make_docx(), DOCX
        )


async def test_xlsx_text_limit_stops_rows_and_closes_workbook(monkeypatch):
    closed = []

    def rows(**kwargs):
        yield ("123456",)
        pytest.fail("Text limit must stop before the next row")

    book = SimpleNamespace(
        worksheets=[SimpleNamespace(max_row=1, max_column=1, iter_rows=rows)],
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr("openpyxl.load_workbook", lambda *args, **kwargs: book)
    data = archive_xml(b"<workbook/>", "xl/workbook.xml")
    with pytest.raises(ExtractionLimitError, match="Workbook text"):
        await ContentExtractionService(config=ExtractionConfig(max_text_chars=5)).extract_bytes(data, XLSX)
    assert closed == [True]


async def test_pptx_text_limit_stops_iteration_before_later_shapes(monkeypatch):
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    def shapes():
        yield SimpleNamespace(
            shape_type=MSO_SHAPE_TYPE.AUTO_SHAPE,
            has_table=False,
            has_text_frame=True,
            text_frame=SimpleNamespace(text="123456"),
        )
        pytest.fail("Text limit must stop before the next shape")

    monkeypatch.setattr(
        "pptx.Presentation", lambda _: SimpleNamespace(slides=[SimpleNamespace(shapes=shapes())])
    )
    data = archive_xml(b"<presentation/>", "ppt/presentation.xml")
    with pytest.raises(ExtractionLimitError, match="Presentation text"):
        await ContentExtractionService(config=ExtractionConfig(max_text_chars=5)).extract_bytes(data, PPTX)


@pytest.mark.parametrize(
    "module,mime,member", [("openpyxl", XLSX, "xl/workbook.xml"), ("pptx", PPTX, "ppt/presentation.xml")]
)
async def test_missing_optional_office_packages_have_actionable_safe_errors(
    monkeypatch, module, mime, member
):
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == module or name.startswith(module + "."):
            raise ImportError("PRIVATE_IMPORT_SENTINEL")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(UnsupportedContentError, match="botkit\\[office\\]") as caught:
        await ContentExtractionService().extract_bytes(archive_xml(b"<root/>", member), mime)
    assert "SENTINEL" not in str(caught.value)


async def test_pdf_page_limit_precedes_heavy_native_text_parser(monkeypatch):
    def forbidden(*args):
        pytest.fail("Too many pages must be rejected before text/table parsing")

    monkeypatch.setattr("botkit.extraction.pdf._native_pages", forbidden)
    with pytest.raises(ExtractionLimitError, match="page limit"):
        await ContentExtractionService(config=ExtractionConfig(max_pages=1)).extract_bytes(
            make_pdf(2), "application/pdf"
        )


async def test_pdf_without_readable_content_does_not_return_page_markers_as_text():
    result = await ContentExtractionService().extract_bytes(make_pdf(blank=True), "application/pdf")
    assert result.text == "" and "empty_extraction" in result.warnings
    assert result.structured_data["pages"][0]["text"] == ""
    assert "page:1:vision_unavailable" in result.warnings


async def test_pdf_vision_requires_provider_before_native_work(monkeypatch):
    def forbidden(*args):
        pytest.fail("Missing provider is a configuration error, not a PDF parse attempt")

    monkeypatch.setattr("botkit.extraction.pdf._page_count", forbidden)
    with pytest.raises(ExtractionError, match="requires an LLM"):
        await PDFExtractor(config=ExtractionConfig(pdf_mode="vision")).extract(b"nonempty", "application/pdf")


async def test_pdf_total_rendered_budget_applies_before_each_provider_call(monkeypatch):
    monkeypatch.setattr("botkit.extraction.pdf._page_count", lambda *_: 2)
    monkeypatch.setattr(
        "botkit.extraction.pdf._render_page", lambda data, number, config: PageImage(number, b"123", 1, 1)
    )
    vision = SimpleNamespace(
        extract=AsyncMock(return_value=ExtractedContent("structured", "OCR", {"tables": []}, [], "low"))
    )
    extractor = PDFExtractor(vision, ExtractionConfig(pdf_mode="vision", max_rendered_bytes=4))
    with pytest.raises(ExtractionLimitError, match="total image size"):
        await extractor.extract(b"nonempty", "application/pdf")
    assert vision.extract.await_count == 1


@pytest.mark.parametrize("kind", ["vision", "audio"])
async def test_custom_provider_ignoring_timeout_option_is_locally_bounded(kind):
    cancelled = asyncio.Event()

    async def slow(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    config = ExtractionConfig(vision_timeout=0.02, audio_timeout=0.02)
    operation = (
        ImageExtractor(SimpleNamespace(ainvoke_with_image_b64=slow), config).extract(make_png(), "image/png")
        if kind == "vision"
        else AudioExtractor(SimpleNamespace(atranscribe=slow), config).extract(b"audio", "audio/ogg")
    )
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(operation, timeout=1)
    assert time.monotonic() - started < 0.5 and cancelled.is_set()


@pytest.mark.parametrize("kind", ["vision", "audio"])
async def test_provider_cancellation_is_not_converted_to_a_parser_error(kind):
    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    operation = (
        ImageExtractor(SimpleNamespace(ainvoke_with_image_b64=cancelled)).extract(make_png(), "image/png")
        if kind == "vision"
        else AudioExtractor(SimpleNamespace(atranscribe=cancelled)).extract(b"audio", "audio/ogg")
    )
    with pytest.raises(asyncio.CancelledError):
        await operation


async def test_vision_table_cells_obey_config_even_when_provider_ignores_validator():
    raw = json.dumps({"text": "", "tables": [{"rows": [["a", "b"]]}], "warnings": []})
    provider = SimpleNamespace(
        ainvoke_with_image_b64=AsyncMock(return_value=LLMResponse(raw, raw, "raw_text_fallback"))
    )
    with pytest.raises(ExtractionLimitError, match="Vision table cell"):
        await ImageExtractor(provider, ExtractionConfig(max_cells=1)).extract(make_png(), "image/png")
    assert provider.ainvoke_with_image_b64.await_count == 1


async def test_vision_raw_size_limit_precedes_json_parsing(monkeypatch):
    def forbidden(*args):
        pytest.fail("Oversized raw result must not reach JSON/schema parsing")

    monkeypatch.setattr("botkit.extraction.vision.vision_payload", forbidden)
    raw = "x" * 20
    provider = SimpleNamespace(
        ainvoke_with_image_b64=AsyncMock(return_value=LLMResponse(raw, raw, "raw_text_fallback"))
    )
    with pytest.raises(ExtractionLimitError):
        await ImageExtractor(provider, ExtractionConfig(max_text_chars=10)).extract(make_png(), "image/png")


def test_deeply_nested_vision_json_is_a_schema_error_not_unhandled_recursion(monkeypatch):
    def recursive(*args):
        raise RecursionError("PRIVATE_JSON_SENTINEL")

    monkeypatch.setattr("botkit.extraction.vision.json.loads", recursive)
    raw = "{}"
    with pytest.raises(InvalidLLMResponse, match="not JSON"):
        vision_payload(LLMResponse(raw, raw, "raw_text_fallback"))
