from __future__ import annotations

import pytest

from arxiv_agent.services.chunker import chunk_document, chunk_section, chunks_from_abstract
from arxiv_agent.services.pdf_parser import ParsedDoc, Section, parse_pdf, segment_sections
from conftest import make_pdf


def test_parse_real_pdf_finds_sections(sample_pdf, settings):
    doc = parse_pdf(sample_pdf, settings)
    assert doc.parser in {"pymupdf", "pdfplumber"}
    assert not doc.degraded
    titles = " ".join(s.title.lower() for s in doc.sections)
    assert "method" in titles
    assert "experiments" in titles
    assert doc.quality > 0.4


def test_parse_extracts_references_separately(sample_pdf, settings):
    doc = parse_pdf(sample_pdf, settings)
    body = " ".join(s.text for s in doc.sections if s.canonical != "references")
    assert "StreamingLLM" not in body or doc.references


def test_empty_pdf_degrades_instead_of_raising(tmp_path, settings):
    blank = make_pdf(tmp_path / "blank.pdf", {"": ""})
    doc = parse_pdf(blank, settings)
    assert doc.degraded is True
    assert doc.warnings


def test_missing_file_degrades_gracefully(tmp_path, settings):
    doc = parse_pdf(tmp_path / "nope.pdf", settings)
    assert doc.degraded is True
    assert doc.parser == "none"


def test_page_cap_marks_truncation(sample_pdf, settings):
    settings.max_pdf_pages = 1
    doc = parse_pdf(sample_pdf, settings)
    assert doc.pages_parsed == 1
    if doc.pages_total > 1:
        assert doc.truncated
        assert any("pages" in w for w in doc.warnings)


def test_segment_sections_splits_on_headings():
    pages = [
        (
            1,
            "Front stuff\n1 Introduction\nWe begin.\n2 Method\nWe do things.\n"
            "References\n[1] Xu, L. Attention Mechanisms Revisited. In ICML, 2020.",
        )
    ]
    sections, refs = segment_sections(pages)
    titles = [s.title for s in sections]
    assert "1 Introduction" in titles
    assert "2 Method" in titles
    assert refs and refs[0].startswith("[1]")


# ------------------------------------------------------------------ #
def _section(text: str, title: str = "3 Method") -> Section:
    return Section(title=title, text=text, page_start=3, page_end=4)


def test_chunks_respect_the_size_budget():
    text = "\n\n".join(f"Paragraph {i}. " + ("word " * 60) for i in range(20))
    chunks = chunk_section(_section(text), chunk_chars=800, overlap=100)
    assert chunks
    # Overlap can push a chunk slightly past the target; it must not run away.
    assert all(len(c) <= 800 + 200 for c in chunks)


def test_long_paragraph_splits_on_sentences_not_mid_word():
    para = " ".join(f"This is sentence number {i} about caches." for i in range(80))
    chunks = chunk_section(_section(para), chunk_chars=400, overlap=0)
    assert len(chunks) > 1
    assert all(not c.endswith(" cach") for c in chunks)
    assert all(c.strip()[0].isupper() or c.strip()[0].isdigit() for c in chunks)


def test_chunks_never_cross_section_boundaries():
    doc = ParsedDoc(
        sections=[
            _section("Method text. " * 30, "3 Method"),
            _section("Limitation text. " * 30, "6 Limitations"),
        ]
    )
    chunks = chunk_document(doc, arxiv_id="1234.5678", chunk_chars=5000, overlap=100)
    assert len(chunks) == 2
    for chunk in chunks:
        assert not ("Method text" in chunk.text and "Limitation text" in chunk.text)


def test_chunk_metadata_is_complete():
    doc = ParsedDoc(sections=[_section("Method text. " * 40)])
    chunk = chunk_document(doc, arxiv_id="1234.5678", chunk_chars=600, overlap=80)[0]
    assert chunk.id.startswith("1234.5678::c")
    assert chunk.metadata["section"] == "3 Method"
    assert chunk.metadata["canonical_section"] == "method"
    assert chunk.metadata["page_start"] == 3
    assert chunk.text.startswith("[Section: 3 Method")


def test_references_are_flagged_low_value():
    doc = ParsedDoc(sections=[_section("[1] Smith. Paper. 2020. " * 20, "References")])
    chunks = chunk_document(doc, arxiv_id="1.1", chunk_chars=600, overlap=0)
    assert all(c.metadata["low_value"] for c in chunks)


def test_overlap_carries_context_between_chunks():
    paragraphs = "\n\n".join(f"Sentence block {i}. " + ("filler " * 40) for i in range(6))
    chunks = chunk_section(_section(paragraphs), chunk_chars=500, overlap=150)
    assert len(chunks) > 2
    # some text from the tail of chunk N reappears at the head of chunk N+1
    assert any(chunks[i][-60:].strip()[:20] in chunks[i + 1] for i in range(len(chunks) - 1))


def test_abstract_only_fallback_produces_one_chunk():
    chunks = chunks_from_abstract("2402.09876", "Title", "Abstract body text.")
    assert len(chunks) == 1
    assert chunks[0].metadata["degraded"] is True
