from __future__ import annotations

import pytest

from arxiv_agent.graph.checkpoint import FileCheckpointer
from arxiv_agent.graph.engine import END, Context, Graph, GraphConfigError
from arxiv_agent.state import AgentState


def _ctx(settings):
    return Context(settings=settings, emit=lambda *_: None)


def test_compile_rejects_dangling_edges():
    g = Graph()
    g.add_node("a", lambda s, c: None)
    g.set_entry("a")
    g.add_edge("a", "nowhere")
    with pytest.raises(GraphConfigError):
        g.compile()


def test_conditional_routing_picks_the_mapped_branch(settings):
    order: list[str] = []
    g = Graph()
    g.add_node("start", lambda s, c: order.append("start"))
    g.add_node("left", lambda s, c: order.append("left"))
    g.add_node("right", lambda s, c: order.append("right"))
    g.set_entry("start")
    g.add_conditional_edges("start", lambda s: "r" if s.raw_input == "go-right" else "l",
                            {"l": "left", "r": "right"})
    g.add_edge("left", END)
    g.add_edge("right", END)
    g.compile()

    g.invoke(AgentState(raw_input="go-right"), _ctx(settings))
    assert order == ["start", "right"]


def test_node_retries_then_succeeds(settings):
    attempts = {"n": 0}

    def flaky(state, ctx):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")

    g = Graph()
    g.add_node("flaky", flaky, retries=3)
    g.set_entry("flaky")
    g.add_edge("flaky", END)
    g.compile()

    state = g.invoke(AgentState(), _ctx(settings))
    assert attempts["n"] == 3
    assert state.status == "running"
    assert not state.errors


def test_non_critical_failure_routes_to_handler_and_continues(settings):
    reached: list[str] = []

    def boom(state, ctx):
        raise ValueError("pdf exploded")

    g = Graph()
    g.add_node("boom", boom, critical=False, on_error="recover")
    g.add_node("recover", lambda s, c: reached.append("recover"))
    g.set_entry("boom")
    g.add_edge("boom", "after")
    g.add_node("after", lambda s, c: reached.append("after"))
    g.add_edge("after", END)
    g.add_edge("recover", END)
    g.compile()

    state = g.invoke(AgentState(), _ctx(settings))
    assert reached == ["recover"]
    assert state.errors and state.errors[0]["type"] == "ValueError"
    assert state.status != "failed"  # the graph stayed alive


def test_critical_failure_halts_with_reason(settings):
    g = Graph()
    g.add_node("boom", lambda s, c: (_ for _ in ()).throw(RuntimeError("fatal")), critical=True)
    g.set_entry("boom")
    g.add_edge("boom", END)
    g.compile()

    state = g.invoke(AgentState(), _ctx(settings))
    assert state.status == "failed"
    assert "fatal" in state.halt_reason


def test_cycle_guard_stops_runaway_graphs(settings):
    g = Graph()
    g.add_node("a", lambda s, c: None)
    g.add_node("b", lambda s, c: None)
    g.set_entry("a")
    g.add_edge("a", "b")
    g.add_edge("b", "a")
    g.compile()

    state = g.invoke(AgentState(), _ctx(settings), max_steps=8)
    assert state.status == "failed"
    assert "step limit" in state.halt_reason


def test_state_is_checkpointed_after_every_node(settings):
    cp = FileCheckpointer(settings.sessions_dir)
    g = Graph()
    g.add_node("a", lambda s, c: s.note("a", "hello"))
    g.set_entry("a")
    g.add_edge("a", END)
    g.compile()

    state = g.invoke(AgentState(raw_input="x"), _ctx(settings), checkpointer=cp)
    restored = cp.load(state.session_id)
    assert restored.raw_input == "x"
    assert any(t.get("note") == "hello" for t in restored.trace)


def test_mermaid_includes_every_node_and_edge():
    from arxiv_agent.graph.build import build_briefing_graph

    diagram = build_briefing_graph().to_mermaid()
    for node in ("understand_query", "search_papers", "fetch_and_parse", "summarize"):
        assert node in diagram
    assert "flowchart TD" in diagram
