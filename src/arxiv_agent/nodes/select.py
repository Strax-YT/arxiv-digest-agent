"""Node 3 — selection / ranking.

A single signal picks the wrong paper often enough to matter, so the score is a
blend of four cheap ones:

    0.45  semantic similarity  (query vs title+abstract embedding)
    0.15  arXiv's own rank     (lexical relevance, decayed by position)
    0.15  recency              (soft preference; "recent work on X" should not
                               return a 2017 paper when 2025 ones exist)
    0.25  LLM re-rank          (top 8 only — reads the abstract and judges fit)

The LLM term is dropped and the rest renormalised if no LLM is available, so
ranking degrades instead of failing.

Ambiguity: if the top two are within `AMBIGUITY_EPS` and we are attached to a
terminal, we ask the user. Non-interactive runs take the top hit and record the
runners-up in the state so the choice is auditable.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

from ..graph.engine import Context
from ..prompts import RERANK_SYSTEM, RERANK_USER
from ..services.llm import LLMError
from ..state import AgentState

log = logging.getLogger(__name__)

AMBIGUITY_EPS = 0.06
LLM_RERANK_POOL = 8
HALF_LIFE_DAYS = 550.0


def _recency(published: str) -> float:
    try:
        when = datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        return 0.5
    age_days = (datetime.now(timezone.utc) - when).days
    return math.exp(-max(0, age_days) / HALF_LIFE_DAYS)


def _llm_scores(state: AgentState, ctx: Context, pool: list[dict]) -> dict[int, tuple[float, str]]:
    listing = "\n\n".join(
        f"[{i}] {p['title']}\n{p.get('abstract', '')[:700]}" for i, p in enumerate(pool)
    )
    raw = ctx.llm.complete_json(
        RERANK_USER.format(request=state.raw_input, candidates=listing), RERANK_SYSTEM, temperature=0.0
    )
    out: dict[int, tuple[float, str]] = {}
    for row in (raw or {}).get("scores", []):
        try:
            idx = int(row["index"])
            score = max(0.0, min(1.0, float(row.get("score", 0))))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < len(pool):
            out[idx] = (score, str(row.get("reason", ""))[:120])
    return out


def select_paper(state: AgentState, ctx: Context) -> None:
    candidates = list(state.candidates)
    if not candidates:
        state.status = "no_results"
        state.halt_reason = "no candidates to rank"
        return
    if len(candidates) == 1:
        state.paper = candidates[0]
        state.selection_rationale = "only one candidate returned by arXiv"
        return

    query_vec = ctx.embedder.embed([state.raw_input], is_query=True)[0]
    doc_vecs = ctx.embedder.embed([f"{c['title']}. {c.get('abstract', '')[:1200]}" for c in candidates])

    llm_scores: dict[int, tuple[float, str]] = {}
    if ctx.llm is not None:
        try:
            llm_scores = _llm_scores(state, ctx, candidates[:LLM_RERANK_POOL])
        except (LLMError, ValueError) as exc:
            log.warning("LLM rerank unavailable (%s); ranking on similarity + recency only", exc)
            state.note("select_paper", "LLM rerank skipped")

    weights = {"sim": 0.45, "rank": 0.15, "recency": 0.15, "llm": 0.25}
    if not llm_scores:
        total = weights["sim"] + weights["rank"] + weights["recency"]
        weights = {k: (v / total if k != "llm" else 0.0) for k, v in weights.items()}

    scored: list[dict] = []
    for idx, cand in enumerate(candidates):
        sim = sum(a * b for a, b in zip(query_vec, doc_vecs[idx]))
        sim = (sim + 1) / 2  # map cosine to 0..1
        rank_score = 1.0 / (1.0 + 0.25 * idx)
        recency = _recency(cand.get("published", ""))
        llm_score, reason = llm_scores.get(idx, (0.0, ""))
        total = (
            weights["sim"] * sim
            + weights["rank"] * rank_score
            + weights["recency"] * recency
            + weights["llm"] * llm_score
        )
        scored.append(
            {
                "paper": cand,
                "score": round(total, 4),
                "parts": {
                    "similarity": round(sim, 3),
                    "arxiv_rank": round(rank_score, 3),
                    "recency": round(recency, 3),
                    "llm": round(llm_score, 3),
                },
                "reason": reason,
            }
        )

    scored.sort(key=lambda r: r["score"], reverse=True)
    top = scored[0]
    runner_up_gap = top["score"] - scored[1]["score"] if len(scored) > 1 else 1.0

    state.runners_up = [
        {"title": r["paper"]["title"], "arxiv_id": r["paper"]["arxiv_id"], "score": r["score"], "parts": r["parts"]}
        for r in scored[1:5]
    ]

    if runner_up_gap < AMBIGUITY_EPS and ctx.ask_user and ctx.settings.interactive:
        options = [
            f"{r['paper']['title'][:90]}  ({r['paper']['arxiv_id']}, {r['paper'].get('published', '')[:7]})"
            for r in scored[:4]
        ]
        choice = ctx.ask_user(
            "Several papers match about equally well. Which should I brief?", options
        )
        top = scored[choice]
        state.selection_rationale = "user picked from an ambiguous candidate set"
    else:
        state.selection_rationale = (
            f"top of {len(scored)} candidates (score {top['score']:.3f}, "
            f"gap to runner-up {runner_up_gap:.3f}); components={top['parts']}"
            + (f"; LLM: {top['reason']}" if top["reason"] else "")
        )
        if runner_up_gap < AMBIGUITY_EPS:
            state.note("select_paper", "close call — see runners_up in state for alternatives")

    state.paper = top["paper"]
    state.note("select_paper", f"selected {top['paper']['arxiv_id']}: {top['paper']['title'][:80]}")


def route_after_selection(state: AgentState) -> str:
    return "no_results" if state.status == "no_results" or not state.paper else "fetch"
