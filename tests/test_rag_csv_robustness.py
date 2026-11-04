"""Regression coverage for ordinary CSV exports, without models or network."""

import pytest

from botkit.rag.ingest import ingest_csv


def test_utf8_bom_and_padded_headers_are_normalized(tmp_path):
    source = tmp_path / "excel.csv"
    source.write_text(" source , text , category \n document.pdf , Useful text , reading \n", encoding="utf-8-sig")
    chunks, report = ingest_csv(str(source))
    assert report.errors == report.warnings == []
    assert len(chunks) == 1
    assert chunks[0].source == "document.pdf"
    assert chunks[0].text == "Useful text"
    assert chunks[0].metadata["category"] == "reading"


@pytest.mark.parametrize("header", ["source,text, text", "source,text,source", "source,text, , "])
def test_duplicate_normalized_headers_fail_before_progress(tmp_path, header):
    source = tmp_path / "ambiguous.csv"
    source.write_text(header + "\nbook,first,second\n", encoding="utf-8")
    calls = []
    with pytest.raises(ValueError, match="duplicate"):
        ingest_csv(str(source), on_progress=lambda *args: calls.append(args))
    assert calls == []


def test_extra_columns_are_reported_without_crashing_or_losing_valid_rows(tmp_path):
    source = tmp_path / "mixed.csv"
    source.write_text("source,text\nbook,bad,unlabelled\nbook2,good\n", encoding="utf-8")
    calls = []
    chunks, report = ingest_csv(str(source), on_progress=lambda *args: calls.append(args))
    assert [chunk.text for chunk in chunks] == ["good"]
    assert report.errors == ["extra_columns_row_0"]
    assert report.chunks_added == 1
    assert calls == [(1, 2), (2, 2)]


def test_multiline_records_and_blank_lines_have_logical_progress_count(tmp_path):
    source = tmp_path / "multiline.csv"
    source.write_text('source,text\n\nbook,"first\nsecond"\n\nbook2,last\n', encoding="utf-8", newline="\n")
    calls = []
    chunks, report = ingest_csv(str(source), on_progress=lambda *args: calls.append(args))
    assert report.errors == []
    assert [chunk.text for chunk in chunks] == ["first\nsecond", "last"]
    assert calls == [(1, 2), (2, 2)]


def test_missing_optional_cell_stays_empty_and_missing_required_cell_is_skipped(tmp_path):
    source = tmp_path / "missing.csv"
    source.write_text("text,source,category\ngood,book\nincomplete\n", encoding="utf-8")
    chunks, report = ingest_csv(str(source))
    assert len(chunks) == 1
    assert chunks[0].metadata["category"] == ""
    assert report.warnings == ["empty_source_row_1"]
