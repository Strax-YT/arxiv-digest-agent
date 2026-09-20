"""Terminal nodes.

Failure has a node of its own rather than being an exception that escapes the
graph. That keeps the exit path uniform: every run ends with a checkpointed
state whose `status` and `halt_reason` explain what happened, which is what the
CLI prints and what a future orchestrator would branch on.
"""

from __future__ import annotations

from ..graph.engine import Context
from ..state import AgentState


def report_no_results(state: AgentState, ctx: Context) -> None:
    state.status = "no_results"
    if not state.halt_reason:
        state.halt_reason = "no papers matched the request"
    tried = [a.get("query", "") for a in state.query_attempts]
    ctx.emit("\nNo usable paper found.\n")
    ctx.emit(state.halt_reason)
    if tried:
        ctx.emit("\nQueries attempted against the arXiv API:")
        for i, query in enumerate(tried):
            ctx.emit(f"  {i}. {query}")
    ctx.emit("")


def report_failure(state: AgentState, ctx: Context) -> None:
    state.status = "failed"
    if not state.halt_reason and state.errors:
        last = state.errors[-1]
        state.halt_reason = f"{last['node']}: {last['type']}: {last['message']}"
    ctx.emit(f"\nRun failed: {state.halt_reason or 'unknown error'}")
    if state.errors:
        ctx.emit("\nError trail:")
        for err in state.errors[-5:]:
            ctx.emit(f"  - [{err['node']}] {err['type']}: {err['message'][:200]}")
    ctx.emit(f"\nSession {state.session_id} is saved; state.json holds the full trace.\n")
