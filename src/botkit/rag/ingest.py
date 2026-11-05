# A3. RAG Ingestion & Retrieval — ingest CSV/PDF/DOCX в список Chunk.
# CSV-путь на основе наработки build_raci_rag.py (строка = документ,
# остальные колонки = metadata); PDF/DOCX-путь на основе
# utils/croma_db_with_bge_model_restructuring.ipynb DocumentProcessor,
# с заменой посимвольного сплиттера на sentence_aware_chunks (splitting.py)
# и добавлением проверок качества (quality.py) постранично, а не на весь
# документ — одна проблемная страница не блокирует ingest остальных.

from __future__ import annotations

import csv as csv_module
import os
from datetime import UTC, datetime

from botkit.rag.base import Chunk, IngestReport, OnProgress
from botkit.rag.quality import page_quality_warning
from botkit.rag.splitting import sentence_aware_chunks


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def ingest_csv(
    csv_path: str, *, on_progress: OnProgress | None = None
) -> tuple[list[Chunk], IngestReport]:
    """Каждая строка CSV становится одним Chunk. Обязательна колонка
    'source' — её отсутствие останавливает ingest до записи чего-либо
    (fail fast), а не пропускает строки молча. Требует колонку 'text'
    (содержимое чанка); остальные колонки становятся metadata как есть.
    """
    chunks: list[Chunk] = []
    warnings: list[str] = []
    errors: list[str] = []

    # UTF-8 BOM is common in spreadsheet exports. Normalize the actual reader
    # keys, not only the header validation, or padded `text` columns are lost.
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv_module.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        fields = [name.strip() for name in reader.fieldnames]
        if len(set(fields)) != len(fields):
            raise ValueError(f"CSV has duplicate normalized column names: {csv_path}")
        reader.fieldnames = fields
        if "source" not in fields:
            raise ValueError(
                f"CSV is missing required 'source' column: {csv_path}"
            )
        if "text" not in fields:
            raise ValueError(f"CSV is missing required 'text' column: {csv_path}")

        ingested_at = _now_iso()
        total_rows = 0
        if on_progress is not None:
            # Count logical CSV records, not physical lines inside quoted text.
            with open(csv_path, newline="", encoding="utf-8-sig") as count_file:
                total_rows = sum(1 for _ in csv_module.DictReader(count_file))

        for row_idx, row in enumerate(reader):
            if None in row:
                # DictReader puts extra cells into a list under None. Never
                # call .strip() on it or silently ingest a truncated record.
                errors.append(f"extra_columns_row_{row_idx}")
                if on_progress is not None:
                    on_progress(row_idx + 1, total_rows)
                continue
            text = (row.get("text") or "").strip()
            if not text:
                warnings.append(f"empty_text_row_{row_idx}")
                if on_progress is not None:
                    on_progress(row_idx + 1, total_rows)
                continue

            metadata: dict = {}
            source_value = ""
            for key, value in row.items():
                if key == "text":
                    continue
                clean_value = (value or "").strip()
                metadata[key] = clean_value
                if key == "source":
                    source_value = clean_value

            if not source_value:
                # Инвариант A3: каждый Chunk обязан иметь source, иначе не
                # проходит в контекст — строка без source не индексируется,
                # а не проходит с пустым полем ("модель выдумала источник").
                warnings.append(f"empty_source_row_{row_idx}")
                if on_progress is not None:
                    on_progress(row_idx + 1, total_rows)
                continue

            metadata["ingested_at"] = ingested_at

            chunks.append(
                Chunk(
                    text=text,
                    source=source_value,
                    page=None,
                    chunk_index=row_idx,
                    metadata=metadata,
                )
            )
            if on_progress is not None:
                on_progress(row_idx + 1, total_rows)

    report = IngestReport(chunks_added=len(chunks), warnings=warnings, errors=errors)
    return chunks, report


def _extract_pdf_pages(pdf_path: str, *, on_progress: OnProgress | None = None) -> list[str]:
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for i, page in enumerate(pdf.pages, start=1):
            pages.append(page.extract_text() or "")
            if on_progress is not None:
                on_progress(i, total)
    return pages


def ingest_pdf(
    pdf_path: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    language: str = "russian",
    on_progress: OnProgress | None = None,
) -> tuple[list[Chunk], IngestReport]:
    """Извлекает текст постранично, проверяет каждую страницу на скан/
    кракозябры (quality.py) — проблемная страница пропускается с warning,
    остальные страницы того же документа индексируются нормально.
    Чанкинг — sentence-aware (splitting.py), никогда не обрывает предложение.

    on_progress(page_done, total_pages) вызывается во время извлечения
    текста pdfplumber-ом — это самый долгий этап на больших PDF (замерено:
    ~17s на 445-страничной книге против ~0.01s на сам чанкинг того же
    объёма текста), поэтому прогресс должен идти именно здесь, а не только
    после того как весь текст уже извлечён. Чанкинг (второй проход ниже)
    on_progress не вызывает — иначе один и тот же (done, total) повторялся
    бы дважды на странице (см. коммит, исправлявший именно это дублирование
    ранее — регрессия при повторном добавлении прогресса не нужна).
    """
    warnings: list[str] = []
    errors: list[str] = []
    chunks: list[Chunk] = []

    try:
        pages = _extract_pdf_pages(pdf_path, on_progress=on_progress)
    except Exception as e:
        errors.append(f"pdf_read_failed: {type(e).__name__}: {e}")
        return [], IngestReport(chunks_added=0, warnings=warnings, errors=errors)

    source = os.path.basename(pdf_path)
    ingested_at = _now_iso()
    chunk_index = 0

    for page_number, page_text in enumerate(pages, start=1):
        warning = page_quality_warning(page_text, page_number=page_number)
        if warning is not None:
            warnings.append(warning)
            continue

        for chunk_text in sentence_aware_chunks(
            page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, language=language
        ):
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    page=page_number,
                    chunk_index=chunk_index,
                    metadata={
                        "source": source,
                        "page": page_number,
                        "ingested_at": ingested_at,
                    },
                )
            )
            chunk_index += 1

    report = IngestReport(chunks_added=len(chunks), warnings=warnings, errors=errors)
    return chunks, report


def _extract_docx_paragraphs_with_headings(docx_path: str) -> list[tuple[str, bool]]:
    """Возвращает список (текст_параграфа, is_heading). is_heading — по
    стилю параграфа (Heading1/2/...) — используется как приблизительная
    граница главы/раздела (chapter), не гарантированно точная."""
    import docx as docx_lib

    doc = docx_lib.Document(docx_path)
    result: list[tuple[str, bool]] = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name or "") if para.style else ""
        is_heading = style_name.lower().startswith("heading")
        result.append((text, is_heading))
    return result


def ingest_docx(
    docx_path: str,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    language: str = "russian",
    on_progress: OnProgress | None = None,
) -> tuple[list[Chunk], IngestReport]:
    """DOCX не имеет "страниц" на уровне формата — chapter извлекается
    эвристически по параграфам со стилем Heading*, page остаётся None.

    on_progress(group_done, total_groups) — группа здесь соответствует
    главе (тексту между двумя заголовками Heading*), а не странице.
    """
    warnings: list[str] = []
    errors: list[str] = []
    chunks: list[Chunk] = []

    try:
        paragraphs = _extract_docx_paragraphs_with_headings(docx_path)
    except Exception as e:
        errors.append(f"docx_read_failed: {type(e).__name__}: {e}")
        return [], IngestReport(chunks_added=0, warnings=warnings, errors=errors)

    source = os.path.basename(docx_path)
    ingested_at = _now_iso()
    chunk_index = 0
    current_chapter: str | None = None

    # Группируем параграфы по текущему заголовку, чтобы каждая группа
    # чанковалась (и проверялась на качество) отдельно от следующей главы.
    groups: list[tuple[str | None, list[str]]] = []
    for text, is_heading in paragraphs:
        if is_heading:
            current_chapter = text
            groups.append((current_chapter, []))
            continue
        if not groups:
            groups.append((current_chapter, []))
        groups[-1][1].append(text)

    total_groups = len(groups)

    for group_index, (chapter, texts) in enumerate(groups, start=1):
        if not texts:
            if on_progress is not None:
                on_progress(group_index, total_groups)
            continue
        full_text = "\n".join(texts)

        warning = page_quality_warning(full_text, page_number=len(warnings) + 1)
        if warning is not None:
            # На DOCX нет номера страницы — переименовываем warning под chapter,
            # чтобы сообщение оставалось информативным без ложной "страницы".
            label = chapter or "document"
            warnings.append(warning.replace("_page_", f"_chapter_{label}_"))
            if on_progress is not None:
                on_progress(group_index, total_groups)
            continue

        for chunk_text in sentence_aware_chunks(
            full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, language=language
        ):
            metadata = {
                "source": source,
                "ingested_at": ingested_at,
            }
            if chapter is not None:
                metadata["chapter"] = chapter

            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    page=None,
                    chunk_index=chunk_index,
                    metadata=metadata,
                )
            )
            chunk_index += 1

        if on_progress is not None:
            on_progress(group_index, total_groups)

    report = IngestReport(chunks_added=len(chunks), warnings=warnings, errors=errors)
    return chunks, report
