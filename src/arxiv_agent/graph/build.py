"""Graph wiring.

Two graphs share one state object and one checkpointer:

  build_briefing_graph()  input -> ... -> briefing   (runs once)
  build_qa_graph()        question -> retrieve -> grounded answer (runs per turn)

They are separate because their lifecycles differ. The briefing graph is a
one-shot DAG; the QA graph is re-entered many times, possibly days later in a
different process, restored from `sessions/<id>/state.json`. Splitting them
keeps the briefing DAG acyclic and makes "resume a session" trivial: load the
state, run the QA graph.

Retry/criticality policy per node:
  understand_query  critical, 0 retries  (a failure here is a bad input)
  search / lookup   critical, 2 retries  (network flakiness is expected)
  fetch_and_parse   NON-critical         (degrades to abstract-only)
  chunk_and_embed   critical, 1 retry
  summarize         critical, 1 retry
"""

from __future__ import annotations

from ..nodes.chunk_embed import chunk_and_embed
from ..nodes.fetch_parse import fetch_and_parse
from ..nodes.qa import answer_question
from ..nodes.retrieve import lookup_paper, route_after_retrieval, search_papers
from ..nodes.select import route_after_selection, select_paper
from ..nodes.summarize import summarize
from ..nodes.terminal import report_failure, report_no_results
from ..nodes.understand import route_after_understanding, understand_query
from .engine import END, Graph


def build_briefing_graph() -> Graph:
    g = Graph("briefing")

    g.add_node("understand_query", understand_query, critical=True)
    g.add_node("lookup_paper", lookup_paper, retries=2, critical=True, on_error="report_failure")
    g.add_node("search_papers", search_papers, retries=2, critical=True, on_error="report_failure")
    g.add_node("select_paper", select_paper, retries=1, critical=True, on_error="report_failure")
    # Non-critical on purpose: a broken PDF degrades the run, it does not end it.
    g.add_node("fetch_and_parse", fetch_and_parse, retries=1, critical=False)
    g.add_node("chunk_and_embed", chunk_and_embed, retries=1, critical=True, on_error="report_failure")
    g.add_node("summarize", summarize, retries=1, critical=True, on_error="report_failure")
    g.add_node("report_no_results", report_no_results, critical=False)
    g.add_node("report_failure", report_failure, critical=False)

    g.set_entry("understand_query")
    g.add_conditional_edges(
        "understand_query",
        route_after_understanding,
        {"lookup": "lookup_paper", "search": "search_papers", "fail": "report_failure"},
    )
    g.add_conditional_edges(
        "lookup_paper",
        route_after_retrieval,
        {"fetch": "fetch_and_parse", "select": "select_paper", "no_results": "report_no_results"},
    )
    g.add_conditional_edges(
        "search_papers",
        route_after_retrieval,
        {"fetch": "fetch_and_parse", "select": "select_paper", "no_results": "report_no_results"},
    )
    g.add_conditional_edges(
        "select_paper",
        route_after_selection,
        {"fetch": "fetch_and_parse", "no_results": "report_no_results"},
    )
    g.add_edge("fetch_and_parse", "chunk_and_embed")
    g.add_edge("chunk_and_embed", "summarize")
    g.add_edge("summarize", END)
    g.add_edge("report_no_results", END)
    g.add_edge("report_failure", END)

    return g.compile()


def build_qa_graph() -> Graph:
    g = Graph("qa")
    g.add_node("answer_question", answer_question, retries=1, critical=True, on_error="report_failure")
    g.add_node("report_failure", report_failure, critical=False)
    g.set_entry("answer_question")
    g.add_edge("answer_question", END)
    g.add_edge("report_failure", END)
    return g.compile()
