"""Node 4 — fetch & parse.

This is the node most likely to hit reality: 400-page appendices, scanned
1998 preprints, PDFs that are one giant image, arXiv 503s.

The contract is that it **never halts the run**. Worst case it sets
`degraded=True`, and the downstream nodes build an abstract-only briefing and
label it as such. A briefing that says "I could only read the abstract" is far
more useful than either a crash or a confident hallucination.

The parsed document is written to a sidecar file next to the session state
rather than into the state itself — full text of a long paper is ~300 KB and
has no business being re-serialised after every node.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..graph.engine import Context
from ..services.pdf_parser import ParsedDoc, parse_pdf
from ..state import AgentState

log = logging.getLogger(__name__)


def fetch_and_parse(state: AgentState, ctx: Context) -> None:
    paper = state.paper_meta
    session_dir = ctx.settings.sessions_dir / state.session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    doc: ParsedDoc
    try:
        pdf_path = ctx.arxiv.download_pdf(paper, ctx.settings.pdf_cache_dir)
        state.pdf_path = str(pdf_path)
        doc = parse_pdf(pdf_path, ctx.settings)
    except Exception as exc:  # noqa: BLE001 - download or open failure
        log.warning("PDF stage failed for %s: %s", paper.arxiv_id, exc)
        state.record_error("fetch_and_parse", exc, recoverable=True)
        doc = ParsedDoc(
            parser="none",
            degraded=True,
            warnings=[f"could not download or open the PDF ({type(exc).__name__}: {exc})"],
        )

    if doc.degraded or not doc.sections:
        doc.degraded = True
        if paper.abstract:
            doc.warnings.append("using the arXiv abstract as the only source of content")
        else:
            doc.warnings.append("no abstract available either — the briefing will be near-empty")

    parsed_path = session_dir / "parsed.json"
    parsed_path.write_text(json.dumps(doc.to_dict(), ensure_ascii=False), encoding="utf-8")
    state.parsed_path = str(parsed_path)
    state.parse_report = {
        "parser": doc.parser,
        "pages_total": doc.pages_total,
        "pages_parsed": doc.pages_parsed,
        "chars": doc.body_chars,
        "section_count": len(doc.sections),
        "reference_count": len(doc.references),
        "truncated": doc.truncated,
        "degraded": doc.degraded,
        "quality_score": doc.quality,
        "warnings": doc.warnings,
        "sections": [s.title for s in doc.sections][:40],
    }
    state.note(
        "fetch_and_parse",
        f"parser={doc.parser} quality={doc.quality} sections={len(doc.sections)} "
        f"chars={doc.body_chars} degraded={doc.degraded}",
    )


def load_parsed(state: AgentState) -> ParsedDoc:
    """Rehydrate the sidecar. Used by chunking and summarisation."""
    from ..services.pdf_parser import Section

    if not state.parsed_path or not Path(state.parsed_path).exists():
        return ParsedDoc(parser="none", degraded=True, warnings=["parsed sidecar missing"])
    data = json.loads(Path(state.parsed_path).read_text(encoding="utf-8"))
    doc = ParsedDoc(
        **{k: v for k, v in data.items() if k not in {"sections"} and k in ParsedDoc.__dataclass_fields__}
    )
    doc.sections = [Section(**s) for s in data.get("sections", [])]
    return doc
