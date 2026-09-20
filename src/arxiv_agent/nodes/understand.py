"""Node 1 — query understanding.

An arXiv id is a *regex problem*, not a reasoning problem. Detecting it
deterministically means the "paste a paper id" path costs zero LLM calls and
cannot be derailed by a model deciding `2401.12345` is a topic.

Only the topic path calls the LLM, and even then a keyword heuristic backs it
up so a dead LLM downgrades the search rather than killing the run.
"""

from __future__ import annotations

import logging
import re

from ..graph.engine import Context
from ..prompts import QUERY_PLAN_SYSTEM, QUERY_PLAN_USER
from ..services.arxiv_client import extract_arxiv_id
from ..services.llm import LLMError
from ..state import AgentState, QueryPlan

log = logging.getLogger(__name__)

STOPWORDS = {
    "recent", "work", "on", "for", "the", "a", "an", "of", "in", "and", "or", "with",
    "about", "papers", "paper", "find", "me", "show", "latest", "new", "research",
    "using", "how", "what", "to", "is", "are", "that", "this", "any", "some",
}

# Coarse routing hints, only used when the LLM is unavailable.
CATEGORY_HINTS = {
    "cs.CL": {"llm", "llms", "language", "nlp", "token", "tokens", "prompt", "translation", "text"},
    "cs.CV": {"image", "vision", "video", "segmentation", "diffusion", "visual", "detection"},
    "cs.LG": {"training", "learning", "optimizer", "gradient", "generalization", "scaling"},
    "cs.CR": {"security", "privacy", "attack", "adversarial", "watermark", "jailbreak"},
    "cs.IR": {"retrieval", "rag", "ranking", "search", "recommendation", "embedding"},
    "cs.DC": {"distributed", "parallel", "inference", "serving", "throughput", "cache"},
    "cs.RO": {"robot", "manipulation", "policy", "locomotion"},
}


def heuristic_plan(text: str) -> QueryPlan:
    lowered = text.lower()
    quoted = re.findall(r'"([^"]{3,60})"', text)
    words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", lowered) if w not in STOPWORDS]

    # Adjacent non-stopword pairs make decent phrase candidates.
    phrases = list(quoted)
    tokens = [w for w in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]+", lowered)]
    for a, b in zip(tokens, tokens[1:]):
        if a not in STOPWORDS and b not in STOPWORDS and len(a) > 2 and len(b) > 2:
            phrases.append(f"{a} {b}")
    phrases = list(dict.fromkeys(phrases))[:3]

    categories = [
        cat for cat, hints in CATEGORY_HINTS.items() if hints & set(words)
    ][:2]
    since = None
    if any(k in lowered for k in ("recent", "latest", "2024", "2025", "state of the art", "sota")):
        since = 2024
    return QueryPlan(
        phrases=phrases,
        terms=list(dict.fromkeys(words))[:6],
        categories=categories,
        since_year=since,
        source="heuristic",
    )


def understand_query(state: AgentState, ctx: Context) -> None:
    text = (state.raw_input or "").strip()
    if not text:
        state.status = "failed"
        state.halt_reason = "empty input"
        state.intent = "unknown"
        return

    arxiv_id = extract_arxiv_id(text)
    # A bare id (or a URL) is a lookup. An id buried in a sentence with lots of
    # other words is more likely a topic mentioning a paper — but a lookup is
    # cheap and reversible, so we prefer it.
    if arxiv_id:
        state.intent = "paper_lookup"
        state.arxiv_id = arxiv_id
        state.note("understand_query", f"detected arXiv id {arxiv_id} (no LLM call needed)")
        return

    state.intent = "topic_search"
    plan = heuristic_plan(text)

    if ctx.llm is not None:
        try:
            raw = ctx.llm.complete_json(
                QUERY_PLAN_USER.format(request=text), QUERY_PLAN_SYSTEM, temperature=0.0
            )
            if isinstance(raw, dict):
                llm_plan = QueryPlan(
                    phrases=[str(p) for p in (raw.get("phrases") or [])][:3],
                    terms=[str(t) for t in (raw.get("terms") or [])][:6],
                    categories=[str(c) for c in (raw.get("categories") or []) if re.match(r"^[a-z-]+\.[A-Z]", str(c))][:3],
                    authors=[str(a) for a in (raw.get("authors") or [])][:3],
                    since_year=raw.get("since_year") if isinstance(raw.get("since_year"), int) else None,
                    source="llm",
                )
                if llm_plan.phrases or llm_plan.terms:
                    plan = llm_plan
                    state.note("understand_query", f"LLM plan: {raw.get('notes', '')}"[:200])
        except (LLMError, ValueError) as exc:
            log.warning("query planning via LLM failed (%s); using keyword heuristic", exc)
            state.record_error("understand_query", exc, recoverable=True)
            state.note("understand_query", "LLM unavailable — fell back to keyword heuristic")

    state.query_plan = plan.__dict__.copy()


def route_after_understanding(state: AgentState) -> str:
    if state.status == "failed":
        return "fail"
    return "lookup" if state.intent == "paper_lookup" else "search"
