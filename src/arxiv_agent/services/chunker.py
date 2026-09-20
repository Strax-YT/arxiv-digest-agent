"""Chunking.

Strategy: **section-aware paragraph packing with sentence-boundary overlap.**

* Never merge across a section boundary. A chunk that straddles "Results" and
  "Limitations" produces answers that attribute a limitation to a result.
* Inside a section, pack whole paragraphs up to `chunk_chars`. A paragraph
  longer than the budget is split on sentence boundaries, never mid-sentence.
* Overlap is carried as trailing *sentences*, not a raw character slice, so a
  chunk always begins at a readable boundary.
* Every chunk keeps `section`, `canonical_section`, page range and position.
  That metadata is what lets the QA answer say "Section 4.2, p.6" instead of
  "somewhere in the paper", and lets retrieval down-weight the bibliography.

The chunk header line ("[Section: Experiments]") is embedded with the body on
purpose: it gives the lexical fallback embedder something to match on and
nudges the dense embedder toward the right part of the paper.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .pdf_parser import ParsedDoc, Section

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")
LOW_VALUE_SECTIONS = {"references", "bibliography", "acknowledgments", "acknowledgements"}


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def _overlap_tail(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    sents = _sentences(text)
    tail: list[str] = []
    size = 0
    for sent in reversed(sents):
        if size + len(sent) > budget:
            # A single sentence larger than the whole budget (equation dumps,
            # table rows): carry only its tail rather than blowing the budget.
            if not tail:
                return sent[-budget:].lstrip()
            break
        tail.insert(0, sent)
        size += len(sent) + 1
    return " ".join(tail)


def _split_paragraph(par: str, limit: int) -> list[str]:
    """Split an over-long paragraph on sentence boundaries; hard-split only if a
    single 'sentence' is still too big (tables, equation dumps)."""
    out: list[str] = []
    buf = ""
    for sent in _sentences(par) or [par]:
        if len(sent) > limit:
            if buf:
                out.append(buf.strip())
                buf = ""
            out.extend(sent[i : i + limit] for i in range(0, len(sent), limit))
            continue
        if len(buf) + len(sent) + 1 > limit and buf:
            out.append(buf.strip())
            buf = sent
        else:
            buf = f"{buf} {sent}".strip()
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_section(section: Section, *, chunk_chars: int, overlap: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.text) if p.strip()]
    pieces: list[str] = []
    for par in paragraphs:
        pieces.extend(_split_paragraph(par, chunk_chars) if len(par) > chunk_chars else [par])

    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        if buf and len(buf) + len(piece) + 2 > chunk_chars:
            chunks.append(buf.strip())
            carry = _overlap_tail(buf, overlap)
            buf = f"{carry}\n{piece}" if carry else piece
        else:
            buf = f"{buf}\n\n{piece}" if buf else piece
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def chunk_document(
    doc: ParsedDoc,
    *,
    arxiv_id: str,
    chunk_chars: int = 1200,
    overlap: int = 180,
    include_references: bool = True,
) -> list[Chunk]:
    out: list[Chunk] = []
    index = 0
    for section in doc.sections:
        canonical = section.canonical
        if canonical in LOW_VALUE_SECTIONS and not include_references:
            continue
        for local, body in enumerate(chunk_section(section, chunk_chars=chunk_chars, overlap=overlap)):
            if len(body.strip()) < 60:  # stray captions / page furniture
                continue
            index += 1
            chunk_id = f"{arxiv_id}::c{index:04d}"
            header = f"[Section: {section.title} | pages {section.page_start}-{section.page_end}]"
            out.append(
                Chunk(
                    id=chunk_id,
                    text=f"{header}\n{body}",
                    metadata={
                        "arxiv_id": arxiv_id,
                        "section": section.title,
                        "canonical_section": canonical,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                        "position": index,
                        "local_index": local,
                        "chars": len(body),
                        "low_value": canonical in LOW_VALUE_SECTIONS,
                    },
                )
            )
    return out


def chunks_from_abstract(arxiv_id: str, title: str, abstract: str) -> list[Chunk]:
    """Degraded mode: the abstract is all we could get, so index just that."""
    body = f"[Section: Abstract | pages 1-1]\nTitle: {title}\n\n{abstract}"
    return [
        Chunk(
            id=f"{arxiv_id}::c0001",
            text=body,
            metadata={
                "arxiv_id": arxiv_id,
                "section": "Abstract",
                "canonical_section": "abstract",
                "page_start": 1,
                "page_end": 1,
                "position": 1,
                "chars": len(abstract),
                "low_value": False,
                "degraded": True,
            },
        )
    ]
