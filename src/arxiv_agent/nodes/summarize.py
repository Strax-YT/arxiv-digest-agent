"""Node 6 — the executive briefing.

Two paths, chosen by size:

* **Short paper** (< `map_reduce_threshold` chars of body): build a labelled
  evidence pack directly from the highest-value sections and summarise in one
  call. Fewer calls, less lossy.
* **Long paper**: map each priority section to dense notes, then reduce the
  notes into the briefing. Sections are visited in *importance* order, not page
  order, so if the budget runs out it is the appendix that gets dropped.

Two grounding decisions worth calling out:

1. Bibliographic fields are never asked of the model — the header of the
   briefing is assembled from arXiv metadata.
2. `limitations` is enforced. Models love to skip it or write "none". If the
   normalised briefing comes back without real limitations, a second targeted
   call runs with an explicit instruction to mark inferred ones as
   "(reviewer-inferred)". Honest labelling beats a silent omission.
"""

from __future__ import annotations

import logging

from ..graph.engine import Context
from ..models import normalise_briefing, render_json, render_markdown
from ..prompts import (
    BRIEFING_SYSTEM,
    BRIEFING_USER,
    LIMITATIONS_RETRY_USER,
    MAP_SECTION_SYSTEM,
    MAP_SECTION_USER,
)
from ..services.llm import LLMError
from ..state import AgentState
from .fetch_parse import load_parsed

log = logging.getLogger(__name__)

# Importance order for evidence selection. Anything not listed sorts last.
SECTION_PRIORITY = [
    "abstract",
    "introduction",
    "method",
    "methods",
    "methodology",
    "approach",
    "model",
    "architecture",
    "results",
    "experiments",
    "evaluation",
    "analysis",
    "ablation",
    "ablation study",
    "limitations",
    "discussion",
    "conclusion",
    "conclusions",
    "future work",
    "experimental setup",
    "implementation",
    "related work",
    "background",
]
EXCLUDED = {"references", "bibliography", "acknowledgments", "acknowledgements"}
PER_SECTION_BUDGET = 3500
TOTAL_EVIDENCE_BUDGET = 18_000


def _priority(canonical: str) -> int:
    try:
        return SECTION_PRIORITY.index(canonical)
    except ValueError:
        return len(SECTION_PRIORITY) + 1


def build_evidence(doc, paper, *, budget: int = TOTAL_EVIDENCE_BUDGET) -> tuple[str, list[dict]]:
    """Labelled evidence blocks (C1, C2, ...) in importance order."""
    blocks: list[dict] = []
    if paper.abstract:
        blocks.append({"section": "Abstract (arXiv metadata)", "text": paper.abstract})

    sections = [s for s in doc.sections if s.canonical not in EXCLUDED and s.text.strip()]
    sections.sort(key=lambda s: _priority(s.canonical))

    used = sum(len(b["text"]) for b in blocks)
    for section in sections:
        if used >= budget:
            break
        text = section.text[:PER_SECTION_BUDGET]
        if len(section.text) > PER_SECTION_BUDGET:
            text += "\n[... section truncated ...]"
        blocks.append({"section": f"{section.title} (p.{section.page_start}-{section.page_end})", "text": text})
        used += len(text)

    rendered = "\n\n".join(
        f"[C{i}] {b['section']}\n{b['text']}" for i, b in enumerate(blocks, start=1)
    )
    labelled = [{"label": f"C{i}", **b} for i, b in enumerate(blocks, start=1)]
    return rendered, labelled


def _map_sections(state: AgentState, ctx: Context, doc, paper) -> str:
    sections = [s for s in doc.sections if s.canonical not in EXCLUDED and len(s.text) > 200]
    sections.sort(key=lambda s: _priority(s.canonical))
    sections = sections[: ctx.settings.max_sections_mapped]

    notes: list[str] = []
    if paper.abstract:
        notes.append(f"[C1] Abstract (arXiv metadata)\n{paper.abstract}")
    for idx, section in enumerate(sections, start=len(notes) + 1):
        try:
            summary = ctx.llm.complete(
                MAP_SECTION_USER.format(
                    title=paper.title,
                    section=section.title,
                    text=section.text[:9000],
                    bullets=6,
                ),
                MAP_SECTION_SYSTEM,
                temperature=0.0,
            ).strip()
        except LLMError as exc:
            log.warning("map step failed for section %s: %s", section.title, exc)
            state.record_error("summarize", exc, recoverable=True)
            summary = section.text[:800]
        if "(no substantive content)" in summary:
            continue
        notes.append(f"[C{idx}] {section.title} (p.{section.page_start}-{section.page_end})\n{summary}")
    state.note("summarize", f"map-reduce over {len(notes)} section(s)")
    return "\n\n".join(notes)


def summarize(state: AgentState, ctx: Context) -> None:
    paper = state.paper_meta
    doc = load_parsed(state)
    degraded = bool(state.parse_report.get("degraded")) or not doc.sections

    if degraded:
        evidence = f"[C1] Abstract (arXiv metadata)\n{paper.abstract or '(no abstract available)'}"
        state.note("summarize", "degraded: briefing from abstract + metadata only")
    elif doc.body_chars > ctx.settings.map_reduce_threshold:
        evidence = _map_sections(state, ctx, doc, paper)
    else:
        evidence, _ = build_evidence(doc, paper)

    raw = ctx.llm.complete_json(
        BRIEFING_USER.format(
            title=paper.title,
            categories=", ".join(paper.categories) or paper.primary_category,
            abstract=paper.abstract or "(unavailable)",
            evidence=evidence,
        ),
        BRIEFING_SYSTEM,
    )
    briefing = normalise_briefing(raw if isinstance(raw, dict) else {})

    # Enforce the limitations section rather than letting the model skip it.
    if not briefing["limitations"]:
        try:
            retry = ctx.llm.complete_json(
                LIMITATIONS_RETRY_USER.format(title=paper.title, evidence=evidence[:12000]),
                BRIEFING_SYSTEM,
                temperature=0.2,
            )
            extra = normalise_briefing({"limitations": (retry or {}).get("limitations")})["limitations"]
            if extra:
                briefing["limitations"] = extra
                briefing["_issues"] = [i for i in briefing["_issues"] if "limitations" not in i]
                state.note("summarize", "limitations recovered on second pass")
        except (LLMError, ValueError) as exc:
            log.warning("limitations retry failed: %s", exc)

    if degraded:
        briefing["confidence"] = "low"
        briefing["confidence_reason"] = (
            "full text unavailable — briefing derived from the abstract and arXiv metadata only"
        )
        briefing["limitations"].append(
            "(reviewer-inferred) This briefing could not read the paper body; verify method "
            "and results against the PDF before relying on them."
        )

    state.briefing = briefing

    meta = {
        "session_id": state.session_id,
        "parser": state.parse_report.get("parser", "none"),
        "pages_parsed": f"{state.parse_report.get('pages_parsed', 0)}/{state.parse_report.get('pages_total', 0)}",
        "chunk_count": state.chunk_count,
        "embedder": state.embedder,
        "vector_backend": state.vector_backend,
        "llm": f"{ctx.settings.provider}:{ctx.settings.model}",
        "degraded": degraded,
        "warnings": state.parse_report.get("warnings", []),
        "selection_rationale": state.selection_rationale,
    }

    session_dir = ctx.settings.sessions_dir / state.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    md_path = session_dir / "briefing.md"
    json_path = session_dir / "briefing.json"
    md_path.write_text(render_markdown(paper, briefing, meta), encoding="utf-8")
    json_path.write_text(render_json(paper, briefing, meta), encoding="utf-8")
    state.briefing_md_path = str(md_path)
    state.briefing_json_path = str(json_path)
    state.status = "briefed"
    state.note("summarize", f"briefing written to {md_path}")
