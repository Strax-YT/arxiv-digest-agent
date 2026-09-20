"""PDF -> sections.

arXiv PDFs are LaTeX output, which is the good case, but the long tail is real:
two-column layouts, scanned survey scans, 100-page appendices, ligature soup.
The strategy is *extract, then measure, then decide*:

1. Try PyMuPDF (fast, gives font sizes -> good heading detection).
2. If the quality gate fails, retry with pdfplumber (different layout engine).
3. If both fail, return `degraded=True` with whatever we have. The caller then
   builds an abstract-only briefing and says so out loud, instead of
   hallucinating a method section.

Quality is scored on characters per page, alphabetic ratio, and whether any
recognisable section headings were found — a scanned paper trips all three.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

CANONICAL_HEADINGS = (
    "abstract",
    "introduction",
    "background",
    "related work",
    "preliminaries",
    "problem statement",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "architecture",
    "implementation",
    "experiments",
    "experimental setup",
    "evaluation",
    "results",
    "analysis",
    "ablation",
    "ablation study",
    "discussion",
    "limitations",
    "future work",
    "conclusion",
    "conclusions",
    "acknowledgments",
    "acknowledgements",
    "references",
    "bibliography",
    "appendix",
)

_HEADING_RE = re.compile(
    r"^\s*(?:(?P<num>\d+(?:\.\d+)*)\.?\s+)?(?P<title>[A-Z][A-Za-z0-9 \-:&/()]{2,70})\s*$"
)
_REF_START_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)
_REF_ITEM_RE = re.compile(r"^\s*(\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}\.)\s+\S")

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl", "\u00ad": ""}


@dataclass
class Section:
    title: str
    text: str
    page_start: int = 0
    page_end: int = 0
    level: int = 1

    @property
    def canonical(self) -> str:
        """Section title reduced to a comparable key: '4.1 Experiments' -> 'experiments'."""
        low = re.sub(r"^\s*\d+(?:\.\d+)*\.?\s*", "", self.title.lower()).strip(" .:")
        for name in CANONICAL_HEADINGS:
            if low.startswith(name):
                return name
        return low[:40]


@dataclass
class ParsedDoc:
    sections: list[Section] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    full_text: str = ""
    pages_total: int = 0
    pages_parsed: int = 0
    parser: str = ""
    truncated: bool = False
    degraded: bool = False
    quality: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def body_chars(self) -> int:
        return len(self.full_text)

    def section_map(self) -> dict[str, Section]:
        return {s.canonical: s for s in self.sections}

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sections"] = [asdict(s) for s in self.sections]
        return data


# --------------------------------------------------------------------- #
# raw extraction
# --------------------------------------------------------------------- #
def _clean(text: str) -> str:
    for bad, good in LIGATURES.items():
        text = text.replace(bad, good)
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    # de-hyphenate across line breaks: "compres-\nsion" -> "compression"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_pymupdf(path: Path, max_pages: int) -> tuple[list[tuple[int, str]], int, list[str]]:
    try:  # the `fitz` alias is deprecated in PyMuPDF >= 1.24
        import pymupdf as fitz
    except ImportError:
        import fitz

    warnings: list[str] = []
    doc = fitz.open(str(path))
    try:
        if doc.needs_pass:
            raise RuntimeError("PDF is password protected")
        total = doc.page_count
        pages: list[tuple[int, str]] = []
        body_size = _dominant_font_size(doc, min(total, 5))
        for page_no in range(min(total, max_pages)):
            page = doc.load_page(page_no)
            blocks = page.get_text("dict").get("blocks", [])
            lines: list[str] = []
            for block in blocks:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text:
                        continue
                    size = max((s.get("size", 0) for s in spans), default=0)
                    bold = any("bold" in (s.get("font", "").lower()) for s in spans)
                    # Mark visually-salient short lines so heading detection can
                    # use typography, not just regex.
                    if (size > body_size * 1.12 or bold) and len(text) < 90:
                        lines.append(f"\x00HEAD\x00{text}")
                    else:
                        lines.append(text)
            pages.append((page_no + 1, "\n".join(lines)))
        return pages, total, warnings
    finally:
        doc.close()


def _dominant_font_size(doc, sample_pages: int) -> float:
    counter: Counter[float] = Counter()
    for page_no in range(sample_pages):
        for block in doc.load_page(page_no).get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    counter[round(span.get("size", 0), 1)] += len(span.get("text", ""))
    return counter.most_common(1)[0][0] if counter else 10.0


def _extract_pdfplumber(path: Path, max_pages: int) -> tuple[list[tuple[int, str]], int, list[str]]:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        total = len(pdf.pages)
        pages = []
        for idx, page in enumerate(pdf.pages[:max_pages]):
            pages.append((idx + 1, page.extract_text() or ""))
        return pages, total, []


def _strip_running_heads(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop lines repeated on most pages (running headers/footers, page numbers)."""
    if len(pages) < 4:
        return pages
    counts: Counter[str] = Counter()
    for _, text in pages:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in lines[:2] + lines[-2:]:
            if len(line) < 120:
                counts[re.sub(r"\d+", "#", line)] += 1
    threshold = max(3, int(len(pages) * 0.6))
    noisy = {k for k, v in counts.items() if v >= threshold}
    if not noisy:
        return pages
    cleaned = []
    for page_no, text in pages:
        kept = [l for l in text.splitlines() if re.sub(r"\d+", "#", l.strip()) not in noisy]
        cleaned.append((page_no, "\n".join(kept)))
    return cleaned


# --------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------- #
def _is_heading(line: str) -> tuple[bool, str]:
    marked = line.startswith("\x00HEAD\x00")
    text = line.removeprefix("\x00HEAD\x00").strip()
    if not text or len(text) > 90:
        return False, text
    low = text.lower().strip(" .:")
    if any(low == h or low.startswith(h + " ") or low == h + "s" for h in CANONICAL_HEADINGS):
        return True, text
    match = _HEADING_RE.match(text)
    if match and match.group("num"):
        return True, text
    if marked and len(text.split()) <= 10 and text[0].isupper() and not text.endswith((".", ",", ";")):
        return True, text
    return False, text


def segment_sections(pages: list[tuple[int, str]]) -> tuple[list[Section], list[str]]:
    sections: list[Section] = []
    references: list[str] = []
    current = Section(title="Front matter", text="", page_start=pages[0][0] if pages else 0)
    in_refs = False
    buffer: list[str] = []
    ref_buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer
        current.text = _clean("\n".join(buffer))
        if current.text or current.title != "Front matter":
            sections.append(
                Section(
                    title=current.title,
                    text=current.text,
                    page_start=current.page_start,
                    page_end=current.page_end,
                    level=current.level,
                )
            )
        buffer = []

    for page_no, text in pages:
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if not line.strip():
                (ref_buffer if in_refs else buffer).append("")
                continue
            heading, clean_line = _is_heading(line)
            if heading:
                if _REF_START_RE.match(clean_line):
                    current.page_end = page_no
                    flush()
                    in_refs = True
                    current = Section(title="References", text="", page_start=page_no)
                    continue
                if in_refs and clean_line.lower().startswith("appendix"):
                    in_refs = False
                current.page_end = page_no
                if not in_refs:
                    flush()
                    current = Section(title=clean_line, text="", page_start=page_no)
                    continue
            (ref_buffer if in_refs else buffer).append(clean_line)
        current.page_end = page_no

    flush()
    references = _split_references(ref_buffer)
    return [s for s in sections if s.text.strip()], references


def _split_references(lines: list[str]) -> list[str]:
    """Group the bibliography into entries.

    Numbered lists ("[1] ...", "1. ...") are the common case and split cleanly
    on the marker. Only when *no* marker appears anywhere do we fall back to
    guessing entry boundaries from "Surname, I." patterns, because that
    heuristic happily shreds a correctly numbered list.
    """
    numbered = any(_REF_ITEM_RE.match(line) for line in lines)
    entries: list[str] = []
    current: list[str] = []

    if numbered:
        for line in lines:
            if _REF_ITEM_RE.match(line) and current:
                entries.append(" ".join(current).strip())
                current = [line.strip()]
            elif line.strip():
                current.append(line.strip())
        if current:
            entries.append(" ".join(current).strip())
    elif lines:
        blob = " ".join(l for l in lines if l.strip())
        entries = [e.strip() for e in re.split(r"(?<=\.)\s(?=[A-Z][a-z]+,? [A-Z])", blob)]

    return [e for e in entries if len(e) > 20][:400]


# --------------------------------------------------------------------- #
def _quality(pages_parsed: int, chars: int, alpha_ratio: float, sections: int) -> float:
    if pages_parsed == 0:
        return 0.0
    per_page = chars / pages_parsed
    score = 0.0
    score += min(1.0, per_page / 1500) * 0.5
    score += min(1.0, alpha_ratio / 0.7) * 0.25
    score += min(1.0, sections / 5) * 0.25
    return round(score, 3)


def parse_pdf(path: Path, settings) -> ParsedDoc:
    """Extract with the best available engine; never raise for a bad PDF."""
    path = Path(path)
    attempts = [("pymupdf", _extract_pymupdf), ("pdfplumber", _extract_pdfplumber)]
    best: ParsedDoc | None = None

    for name, extractor in attempts:
        try:
            pages, total, warnings = extractor(path, settings.max_pdf_pages)
        except ImportError as exc:
            log.info("%s not installed (%s)", name, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed on %s: %s", name, path.name, exc)
            continue

        pages = _strip_running_heads(pages)
        sections, references = segment_sections(pages)
        full_text = "\n\n".join(f"## {s.title}\n{s.text}" for s in sections)
        alpha = sum(c.isalpha() for c in full_text)
        alpha_ratio = alpha / max(1, len(full_text))
        doc = ParsedDoc(
            sections=sections,
            references=references,
            full_text=full_text,
            pages_total=total,
            pages_parsed=len(pages),
            parser=name,
            truncated=total > len(pages),
            quality=_quality(len(pages), len(full_text), alpha_ratio, len(sections)),
            warnings=list(warnings),
        )
        if doc.truncated:
            doc.warnings.append(
                f"paper has {total} pages; parsed the first {len(pages)} "
                f"(raise MAX_PDF_PAGES to include appendices)"
            )
        if len(full_text) / max(1, len(pages)) < settings.min_chars_per_page:
            doc.warnings.append(
                "very little extractable text per page — the PDF is probably scanned images; "
                "no OCR is performed by default"
            )
        if best is None or doc.quality > best.quality:
            best = doc
        if doc.quality >= 0.55:
            break

    if best is None:
        return ParsedDoc(
            parser="none",
            degraded=True,
            warnings=["no PDF parser could open the file (install pymupdf or pdfplumber)"],
        )
    if best.quality < 0.35 or not best.sections:
        best.degraded = True
        best.warnings.append(
            f"extraction quality {best.quality:.2f} is below the usable threshold; "
            "falling back to abstract-only mode"
        )
    return best
