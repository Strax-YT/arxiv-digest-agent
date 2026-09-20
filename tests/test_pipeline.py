"""End-to-end runs through the real graph with fakes at the network boundary."""

from __future__ import annotations

import json

from arxiv_agent.agent import Agent
from arxiv_agent.models import normalise_briefing
from arxiv_agent.services.llm import MockLLM, parse_json_loose
from conftest import BRIEFING_JSON, FakeArxiv


def _agent(settings, mock_llm, embedder, store, arxiv):
    return Agent(settings, llm=mock_llm, embedder=embedder, store=store, arxiv=arxiv, emit=lambda *_: None)


def test_full_run_by_arxiv_id(settings, mock_llm, embedder, store, sample_pdf):
    agent = _agent(settings, mock_llm, embedder, store, FakeArxiv(pdf_path=sample_pdf))
    state = agent.brief("https://arxiv.org/abs/2402.09876")

    assert state.status == "briefed"
    assert state.paper_meta.arxiv_id == "2402.09876"
    assert state.chunk_count > 1
    assert state.briefing["limitations"]
    assert state.briefing["key_results"]

    md = open(state.briefing_md_path, encoding="utf-8").read()
    assert "PagedCache" in md
    assert "## Limitations" in md
    assert "arxiv.org/abs/2402.09876" in md

    payload = json.loads(open(state.briefing_json_path, encoding="utf-8").read())
    assert payload["paper"]["authors"] == ["A. Researcher", "B. Coauthor"]
    assert payload["provenance"]["chunk_count"] == state.chunk_count

    # visited nodes, in order
    nodes = [t["node"] for t in state.trace if "outcome" in t]
    assert nodes == ["understand_query", "lookup_paper", "fetch_and_parse", "chunk_and_embed", "summarize"]


def test_full_run_by_topic_selects_the_method_paper(settings, mock_llm, embedder, store, sample_pdf):
    agent = _agent(settings, mock_llm, embedder, store, FakeArxiv(pdf_path=sample_pdf))
    state = agent.brief("recent work on KV-cache compression for LLMs")

    assert state.status == "briefed"
    assert state.intent == "topic_search"
    assert state.paper_meta.arxiv_id == "2402.09876"  # method paper beats the survey
    assert state.runners_up
    assert "score" in state.selection_rationale


def test_pdf_failure_degrades_to_abstract_only(settings, mock_llm, embedder, store):
    # FakeArxiv with no pdf_path raises on download.
    agent = _agent(settings, mock_llm, embedder, store, FakeArxiv(pdf_path=None))
    state = agent.brief("2402.09876")

    assert state.status == "briefed"  # the run survived
    assert state.parse_report["degraded"] is True
    assert state.chunk_count == 1
    assert state.briefing["confidence"] == "low"
    assert any("reviewer-inferred" in l for l in state.briefing["limitations"])
    md = open(state.briefing_md_path, encoding="utf-8").read()
    assert "Degraded run" in md


def test_unknown_id_halts_without_a_briefing(settings, mock_llm, embedder, store):
    agent = _agent(settings, mock_llm, embedder, store, FakeArxiv())
    state = agent.brief("9999.99999")
    assert state.status == "no_results"
    assert state.briefing is None


def test_session_resumes_in_a_new_agent(settings, mock_llm, embedder, store, sample_pdf):
    agent = _agent(settings, mock_llm, embedder, store, FakeArxiv(pdf_path=sample_pdf))
    state = agent.brief("2402.09876")
    session_id = state.session_id

    # Simulate a completely new process: new Agent, new LLM, same workdir + store.
    llm2 = MockLLM(settings, {"Question:": "The blocks are 64 tokens [C1]."})
    agent2 = _agent(settings, llm2, embedder, store, FakeArxiv(pdf_path=sample_pdf))
    restored = agent2.load(session_id)
    assert restored.collection == state.collection
    assert restored.chunk_count == state.chunk_count

    restored = agent2.ask(restored, "How large are the cache blocks?")
    assert restored.qa_history[-1]["grounded"]
    assert agent2.load(session_id).qa_history  # persisted again


def test_reembedding_is_skipped_on_a_second_run(settings, mock_llm, embedder, store, sample_pdf):
    arxiv = FakeArxiv(pdf_path=sample_pdf)
    first = _agent(settings, mock_llm, embedder, store, arxiv).brief("2402.09876")
    second = _agent(settings, mock_llm, embedder, store, arxiv).brief("2402.09876")
    assert first.collection == second.collection
    assert any("reusing" in t.get("note", "") for t in second.trace)


# ------------------------------------------------------------------ #
def test_normalise_coerces_string_lists():
    briefing = normalise_briefing(
        {
            "why_it_matters": "x" * 40,
            "problem_statement": "y" * 40,
            "method": "- one\n- two",
            "key_results": ["a plain string result"],
            "limitations": ["only English"],
            "follow_up_questions": ["why?"],
            "confidence": "VERY HIGH",
        }
    )
    assert briefing["method"] == ["one", "two"]
    assert briefing["key_results"][0] == {"claim": "a plain string result", "evidence": ""}
    assert briefing["confidence"] == "medium"  # invalid value normalised
    assert briefing["_issues"] == []


def test_normalise_reports_missing_fields_instead_of_faking_them():
    briefing = normalise_briefing({"why_it_matters": "short"})
    assert "limitations is empty" in briefing["_issues"]
    assert "why_it_matters missing or too short" in briefing["_issues"]


def test_limitations_retry_fires_when_the_model_omits_them(settings, embedder, store, sample_pdf):
    stripped = json.loads(BRIEFING_JSON)
    stripped["limitations"] = []
    llm = MockLLM(
        settings,
        {
            "Write an executive briefing": json.dumps(stripped),
            "had an empty or trivial": json.dumps({"limitations": ["(reviewer-inferred) English only"]}),
        },
    )
    agent = _agent(settings, llm, embedder, store, FakeArxiv(pdf_path=sample_pdf))
    state = agent.brief("2402.09876")
    assert state.briefing["limitations"] == ["(reviewer-inferred) English only"]
    assert any("limitations recovered" in t.get("note", "") for t in state.trace)


def test_json_parser_survives_fences_and_truncation():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert parse_json_loose('{"a": 1, "b": [2, 3')["a"] == 1  # truncated generation
    assert parse_json_loose('{"a": 1,}')["a"] == 1  # trailing comma
