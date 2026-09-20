from __future__ import annotations

import pytest

from arxiv_agent.graph.engine import Context
from arxiv_agent.nodes.retrieve import search_papers
from arxiv_agent.nodes.understand import heuristic_plan, understand_query
from arxiv_agent.services.arxiv_client import ArxivClient, extract_arxiv_id, render_query
from arxiv_agent.state import AgentState, QueryPlan
from conftest import ATOM_FEED, EmptyThenFullArxiv, FakeArxiv


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2401.12345", "2401.12345"),
        ("arXiv:2401.12345v3", "2401.12345"),
        ("https://arxiv.org/abs/2401.12345", "2401.12345"),
        ("https://arxiv.org/pdf/2401.12345v2.pdf", "2401.12345"),
        ("http://arxiv.org/abs/cs/0112017", "cs/0112017"),
        ("summarise 2401.12345 for me", "2401.12345"),
        ("recent work on KV cache compression", None),
        ("papers from 2024 about attention", None),
    ],
)
def test_extract_arxiv_id(text, expected):
    assert extract_arxiv_id(text) == expected


def test_id_lookup_needs_no_llm(settings):
    state = AgentState(raw_input="https://arxiv.org/abs/2402.09876")
    understand_query(state, Context(settings=settings, llm=None))
    assert state.intent == "paper_lookup"
    assert state.arxiv_id == "2402.09876"


def test_topic_falls_back_to_heuristic_plan_without_llm(settings):
    state = AgentState(raw_input="recent work on KV-cache compression for LLMs")
    understand_query(state, Context(settings=settings, llm=None))
    assert state.intent == "topic_search"
    assert state.query_plan["source"] == "heuristic"
    assert state.query_plan["terms"]


def test_heuristic_plan_guesses_a_category():
    plan = heuristic_plan("adversarial attacks on image segmentation models")
    assert "cs.CV" in plan.categories or "cs.CR" in plan.categories


def test_relaxation_ladder_gets_progressively_looser():
    plan = QueryPlan(
        phrases=["kv cache compression", "long context"],
        terms=["quantization", "inference"],
        categories=["cs.CL"],
        since_year=2024,
    )
    l0, l2, l3, l4 = (render_query(plan, i) for i in (0, 2, 3, 4))
    assert " AND " in l0 and "submittedDate" in l0
    assert " OR " in l2 and "cat:cs.CL" in l2
    assert "cat:" not in l3  # category dropped
    assert len(l4) < len(l0)  # last-ditch sweep is the broadest


def test_query_rendering_quotes_phrases():
    q = render_query(QueryPlan(phrases=["speculative decoding"]), 0)
    assert 'all:"speculative decoding"' in q


def test_zero_results_walks_the_ladder_then_succeeds(settings):
    fake = EmptyThenFullArxiv(succeed_at=2)
    state = AgentState(raw_input="obscure topic", query_plan=QueryPlan(phrases=["obscure topic"]).__dict__)
    search_papers(state, Context(settings=settings, arxiv=fake))
    # two empty levels, a hit at level 2, then exactly one extra widening attempt
    assert len(fake.queries) == 4
    assert len(state.candidates) == 2
    assert [a["hits"] for a in state.query_attempts] == [0, 0, 2, 2]


def test_exhausted_ladder_halts_cleanly_instead_of_inventing(settings):
    fake = EmptyThenFullArxiv(succeed_at=99)
    state = AgentState(raw_input="zzz", query_plan=QueryPlan(phrases=["zzz"]).__dict__)
    search_papers(state, Context(settings=settings, arxiv=fake))
    assert state.status == "no_results"
    assert state.candidates == []
    assert "Try:" in state.halt_reason


def test_atom_parsing_extracts_metadata():
    papers = ArxivClient._parse_feed(ATOM_FEED)
    first = papers[0]
    assert first.arxiv_id == "2402.09876"
    assert first.version == "v2"
    assert first.authors == ["A. Researcher", "B. Coauthor"]
    assert first.primary_category == "cs.CL"
    assert first.pdf_url.endswith("2402.09876v2")
    assert "PagedCache" in first.title
    assert first.comment == "14 pages, 6 figures"


def test_fake_client_get_by_id():
    client = FakeArxiv()
    assert client.get_by_id("2402.09876").title.startswith("PagedCache")
    assert client.get_by_id("9999.99999") is None
