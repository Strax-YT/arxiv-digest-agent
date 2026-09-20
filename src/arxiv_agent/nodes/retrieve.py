"""Nodes 2a/2b — arXiv retrieval.

`lookup_paper` handles a known id. `search_papers` handles a topic and owns the
answer to "what if arXiv returns zero, or three thousand, results?":

* **Zero** → walk down the relaxation ladder (strict AND → OR → drop category →
  broad sweep). Every attempt is recorded in `state.query_attempts`, so the
  README's "here's what it actually asked arXiv" story is inspectable. If the
  ladder bottoms out, the run halts cleanly with `status="no_results"` and
  suggested rephrasings — it does not invent a paper.
* **Many** → cap at `arxiv_max_results`, deduplicate by base id (arXiv returns
  v1/v2 of the same work), drop withdrawn papers, and hand the pool to the
  ranking node. Relevance sort is requested from the API, but we re-rank
  locally because arXiv's relevance is lexical.
"""

from __future__ import annotations

import logging

from ..graph.engine import Context
from ..services.arxiv_client import MAX_RELAXATION_LEVEL, render_query
from ..state import AgentState, QueryPlan

log = logging.getLogger(__name__)

MIN_CANDIDATES = 3
WITHDRAWN_MARKERS = ("this paper has been withdrawn", "paper withdrawn", "this submission has been withdrawn")


def lookup_paper(state: AgentState, ctx: Context) -> None:
    paper = ctx.arxiv.get_by_id(state.arxiv_id)
    if paper is None:
        state.status = "no_results"
        state.halt_reason = (
            f"arXiv has no record of id {state.arxiv_id!r}. Check the id, or paste the abs/ URL."
        )
        state.note("lookup_paper", "id not found")
        return
    state.candidates = [paper.__dict__]
    state.paper = paper.__dict__
    state.selection_rationale = "explicit arXiv id supplied by the user"
    state.note("lookup_paper", f"resolved {paper.arxiv_id}: {paper.title[:80]}")


def _dedupe(papers: list) -> list:
    seen: set[str] = set()
    out = []
    for paper in papers:
        base = paper.arxiv_id.split("v")[0]
        if base in seen:
            continue
        if any(m in paper.abstract.lower()[:200] for m in WITHDRAWN_MARKERS):
            continue
        seen.add(base)
        out.append(paper)
    return out


def search_papers(state: AgentState, ctx: Context) -> None:
    plan = QueryPlan(**{k: v for k, v in state.query_plan.items() if k in QueryPlan.__dataclass_fields__})
    collected: list = []

    first_hit_level: int | None = None

    for level in range(MAX_RELAXATION_LEVEL + 1):
        query = render_query(plan, level)
        try:
            results = ctx.arxiv.search(query, max_results=ctx.settings.arxiv_max_results)
        except Exception as exc:  # noqa: BLE001 - the graph retries this node
            state.query_attempts.append({"level": level, "query": query, "error": str(exc)[:200]})
            raise
        results = _dedupe(results)
        state.query_attempts.append({"level": level, "query": query, "hits": len(results)})
        log.info("relaxation level %d -> %d hits for %s", level, len(results), query)

        if len(results) > len(collected):
            collected = results
        if results and first_hit_level is None:
            first_hit_level = level
        if len(collected) >= MIN_CANDIDATES:
            break
        # A thin pool is worth one extra widening step, but only one: each level
        # costs a 3-second rate-limited round trip.
        if first_hit_level is not None and level > first_hit_level:
            break

    if not collected:
        plan_terms = ", ".join(plan.phrases + plan.terms) or state.raw_input
        state.status = "no_results"
        state.halt_reason = (
            f"arXiv returned nothing for any relaxation of: {plan_terms}.\n"
            "Try: (a) the authors' own terminology rather than a description of the idea, "
            "(b) dropping the date/category constraint, or (c) pasting a specific arXiv id."
        )
        state.note("search_papers", f"exhausted {len(state.query_attempts)} query variants, 0 hits")
        return

    state.candidates = [p.__dict__ for p in collected[: ctx.settings.arxiv_max_results]]
    state.note(
        "search_papers",
        f"{len(state.candidates)} candidates after {len(state.query_attempts)} query variant(s)",
    )


def route_after_retrieval(state: AgentState) -> str:
    if state.status == "no_results":
        return "no_results"
    if state.paper:  # id lookup already settled the choice
        return "fetch"
    return "select"
