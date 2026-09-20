"""Briefing schema, validation and rendering.

The LLM returns free-form JSON; `normalise_briefing` coerces it into the shape
the rest of the system (and the grader) expects. Missing required fields are
recorded in `_issues` rather than silently defaulted, so degraded output is
visible instead of plausible-looking.
"""

from __future__ import annotations

import json
from typing import Any

from .state import PaperMeta

REQUIRED_LIST_FIELDS = ("method", "key_results", "limitations", "follow_up_questions")
REQUIRED_TEXT_FIELDS = ("why_it_matters", "problem_statement")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        lines = [l.strip(" -*\t") for l in value.splitlines() if l.strip()]
        return lines or [value.strip()]
    return [value]


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(str(v) for v in value).strip()
    return "" if value is None else str(value)


def normalise_briefing(raw: dict[str, Any]) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    issues: list[str] = []

    briefing: dict[str, Any] = {
        "why_it_matters": _as_text(raw.get("why_it_matters") or raw.get("summary")),
        "problem_statement": _as_text(raw.get("problem_statement") or raw.get("problem")),
        "method": [_as_text(x) for x in _as_list(raw.get("method") or raw.get("approach"))],
        "key_results": [],
        "limitations": [_as_text(x) for x in _as_list(raw.get("limitations"))],
        "follow_up_questions": [
            _as_text(x) for x in _as_list(raw.get("follow_up_questions") or raw.get("questions"))
        ],
        "confidence": _as_text(raw.get("confidence") or "medium").lower() or "medium",
        "confidence_reason": _as_text(raw.get("confidence_reason")),
    }

    for item in _as_list(raw.get("key_results") or raw.get("results")):
        if isinstance(item, dict):
            briefing["key_results"].append(
                {
                    "claim": _as_text(item.get("claim") or item.get("result") or item.get("text")),
                    "evidence": _as_text(item.get("evidence") or item.get("citation")),
                }
            )
        else:
            briefing["key_results"].append({"claim": _as_text(item), "evidence": ""})

    for field in REQUIRED_TEXT_FIELDS:
        if len(briefing[field]) < 20:
            issues.append(f"{field} missing or too short")
    briefing["key_results"] = [r for r in briefing["key_results"] if r.get("claim")]
    for field in REQUIRED_LIST_FIELDS:
        if field != "key_results":
            briefing[field] = [x for x in briefing[field] if x and len(x.strip()) > 2]
        if not briefing[field]:
            issues.append(f"{field} is empty")
    if briefing["confidence"] not in {"high", "medium", "low"}:
        briefing["confidence"] = "medium"

    briefing["_issues"] = issues
    return briefing


def render_markdown(paper: PaperMeta, briefing: dict[str, Any], meta: dict[str, Any]) -> str:
    authors = ", ".join(paper.authors[:8]) + (" et al." if len(paper.authors) > 8 else "")
    lines: list[str] = [
        f"# {paper.title or '(untitled)'}",
        "",
        f"**arXiv:** [{paper.arxiv_id}{paper.version}]({paper.abs_url})  ",
        f"**Authors:** {authors or 'unknown'}  ",
        f"**Published:** {paper.published[:10] or 'unknown'}"
        + (f" (updated {paper.updated[:10]})" if paper.updated[:10] != paper.published[:10] else "")
        + "  ",
        f"**Categories:** {', '.join(paper.categories) or paper.primary_category or 'n/a'}  ",
        f"**PDF:** {paper.pdf_url}  ",
    ]
    if paper.comment:
        lines.append(f"**Author comment:** {paper.comment}  ")
    lines += ["", "---", "", "## Why this paper matters", "", briefing.get("why_it_matters", "")]
    lines += ["", "## Problem statement", "", briefing.get("problem_statement", "")]

    lines += ["", "## Method / approach", ""]
    lines += [f"- {b}" for b in briefing.get("method", [])] or ["- (not extracted)"]

    lines += ["", "## Key results and claims", ""]
    for item in briefing.get("key_results", []) or []:
        cite = f" _[{item['evidence']}]_" if item.get("evidence") else ""
        lines.append(f"- {item.get('claim', '')}{cite}")
    if not briefing.get("key_results"):
        lines.append("- (not extracted)")

    lines += ["", "## Limitations", ""]
    lines += [f"- {b}" for b in briefing.get("limitations", [])] or ["- (not extracted)"]

    lines += ["", "## Suggested follow-up questions", ""]
    lines += [f"{i}. {q}" for i, q in enumerate(briefing.get("follow_up_questions", []), 1)]

    lines += ["", "---", "", "## Provenance"]
    lines.append("")
    lines.append(f"- Briefing confidence: **{briefing.get('confidence', 'medium')}**"
                 + (f" — {briefing['confidence_reason']}" if briefing.get("confidence_reason") else ""))
    for key, label in (
        ("parser", "PDF parser"),
        ("pages_parsed", "Pages parsed"),
        ("chunk_count", "Chunks indexed"),
        ("embedder", "Embedding model"),
        ("vector_backend", "Vector store"),
        ("llm", "LLM"),
        ("session_id", "Session id"),
    ):
        if meta.get(key) not in (None, ""):
            lines.append(f"- {label}: `{meta[key]}`")
    for warning in meta.get("warnings", []) or []:
        lines.append(f"- ⚠️ {warning}")
    for issue in briefing.get("_issues", []) or []:
        lines.append(f"- ⚠️ schema: {issue}")
    if meta.get("degraded"):
        lines.append(
            "- ⚠️ **Degraded run:** the full text could not be parsed, so this briefing is "
            "based on the abstract and metadata only. Treat method and results as indicative."
        )
    lines += ["", f"_Ask follow-up questions with:_ `python -m arxiv_agent chat {meta.get('session_id', '')}`", ""]
    return "\n".join(lines)


def render_json(paper: PaperMeta, briefing: dict[str, Any], meta: dict[str, Any]) -> str:
    payload = {
        "paper": {
            "arxiv_id": paper.arxiv_id,
            "version": paper.version,
            "title": paper.title,
            "authors": paper.authors,
            "published": paper.published,
            "updated": paper.updated,
            "categories": paper.categories,
            "url": paper.abs_url,
            "pdf_url": paper.pdf_url,
        },
        "briefing": {k: v for k, v in briefing.items() if not k.startswith("_")},
        "provenance": meta,
        "schema_issues": briefing.get("_issues", []),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)
