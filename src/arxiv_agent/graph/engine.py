"""A tiny, explicit state-graph runtime.

Why not LangGraph? See README "Design Decisions". Short version: the graph here
is ~15 nodes with simple routing, and writing the runtime by hand costs ~150
lines while making the control flow, the retry policy and the checkpoint
boundary completely legible in one file. The public surface intentionally
mirrors LangGraph (`add_node` / `add_edge` / `add_conditional_edges` / `invoke`)
so swapping it out later is mechanical.

Semantics
---------
* A node is ``fn(state, ctx) -> None``. Nodes mutate the shared state in place.
* Routing is separate from work: static edges, or a router function that reads
  state and returns the next node name. Nodes never decide their own successor,
  which keeps the graph drawable from the code (`to_mermaid`).
* Every node is wrapped: timing, retry-with-backoff, exception capture. An
  exception in a non-critical node routes to ``on_error`` instead of killing
  the run; a critical node halts the graph with ``status="failed"``.
* After every node the state is checkpointed. That is the persistence boundary
  between the briefing run and a later QA session.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from ..state import AgentState

log = logging.getLogger(__name__)

END = "__end__"

NodeFn = Callable[[AgentState, "Context"], None]
Router = Callable[[AgentState], str]


class Checkpointer(Protocol):
    def save(self, state: AgentState) -> None: ...


class _NullCheckpointer:
    def save(self, state: AgentState) -> None:  # pragma: no cover - trivial
        return None


@dataclass
class Context:
    """Everything a node needs that is *not* state: clients, settings, io."""

    settings: Any
    llm: Any = None
    embedder: Any = None
    store: Any = None
    arxiv: Any = None
    ask_user: Callable[[str, list[str]], int] | None = None
    emit: Callable[[str], None] = print


@dataclass
class NodeSpec:
    name: str
    fn: NodeFn
    retries: int = 0
    backoff: float = 1.5
    critical: bool = True
    on_error: str | None = None


class GraphConfigError(RuntimeError):
    pass


class Graph:
    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self.nodes: dict[str, NodeSpec] = {}
        self.edges: dict[str, str] = {}
        self.conditional: dict[str, tuple[Router, dict[str, str]]] = {}
        self.entry: str | None = None

    # -- construction ------------------------------------------------- #
    def add_node(
        self,
        name: str,
        fn: NodeFn,
        *,
        retries: int = 0,
        critical: bool = True,
        on_error: str | None = None,
    ) -> "Graph":
        if name in self.nodes:
            raise GraphConfigError(f"duplicate node: {name}")
        self.nodes[name] = NodeSpec(name, fn, retries=retries, critical=critical, on_error=on_error)
        return self

    def set_entry(self, name: str) -> "Graph":
        self.entry = name
        return self

    def add_edge(self, src: str, dst: str) -> "Graph":
        if src in self.conditional:
            raise GraphConfigError(f"{src} already has conditional edges")
        self.edges[src] = dst
        return self

    def add_conditional_edges(self, src: str, router: Router, mapping: dict[str, str]) -> "Graph":
        if src in self.edges:
            raise GraphConfigError(f"{src} already has a static edge")
        self.conditional[src] = (router, mapping)
        return self

    def compile(self) -> "Graph":
        """Fail loudly at startup rather than half-way through a paper."""
        if not self.entry:
            raise GraphConfigError("no entry node set")
        known = set(self.nodes) | {END}
        if self.entry not in known:
            raise GraphConfigError(f"entry node {self.entry!r} is not defined")
        for src, dst in self.edges.items():
            if src not in self.nodes or dst not in known:
                raise GraphConfigError(f"dangling edge {src} -> {dst}")
        for src, (_, mapping) in self.conditional.items():
            if src not in self.nodes:
                raise GraphConfigError(f"conditional edge from unknown node {src}")
            for label, dst in mapping.items():
                if dst not in known:
                    raise GraphConfigError(f"dangling conditional edge {src} -[{label}]-> {dst}")
        for spec in self.nodes.values():
            if spec.on_error and spec.on_error not in known:
                raise GraphConfigError(f"unknown error handler {spec.on_error}")
        return self

    # -- execution ---------------------------------------------------- #
    def _next(self, node: str, state: AgentState) -> str:
        if node in self.conditional:
            router, mapping = self.conditional[node]
            label = router(state)
            if label not in mapping:
                raise GraphConfigError(f"router for {node} returned unmapped label {label!r}")
            return mapping[label]
        return self.edges.get(node, END)

    def invoke(
        self,
        state: AgentState,
        ctx: Context,
        *,
        checkpointer: Checkpointer | None = None,
        start: str | None = None,
        max_steps: int = 60,
    ) -> AgentState:
        cp = checkpointer or _NullCheckpointer()
        current = start or self.entry or END
        steps = 0

        while current != END:
            if steps >= max_steps:
                state.status = "failed"
                state.halt_reason = f"step limit ({max_steps}) exceeded — probable cycle"
                break
            steps += 1
            spec = self.nodes[current]
            outcome, elapsed = self._run_node(spec, state, ctx)
            state.trace.append(
                {"node": spec.name, "outcome": outcome, "seconds": round(elapsed, 3), "ts": time.time()}
            )
            state.updated_at = time.time()
            cp.save(state)

            if outcome == "error":
                if spec.on_error:
                    current = spec.on_error
                    continue
                if spec.critical:
                    state.status = "failed"
                    state.halt_reason = f"{spec.name} failed: {state.errors[-1]['message'] if state.errors else ''}"
                    break
            if state.status == "awaiting_user":
                break
            current = self._next(spec.name, state)

        cp.save(state)
        return state

    def _run_node(self, spec: NodeSpec, state: AgentState, ctx: Context) -> tuple[str, float]:
        started = time.time()
        attempt = 0
        while True:
            try:
                log.debug("node %s attempt %d", spec.name, attempt + 1)
                spec.fn(state, ctx)
                return "ok", time.time() - started
            except Exception as exc:  # noqa: BLE001 - boundary: we log and route
                attempt += 1
                if attempt <= spec.retries:
                    delay = spec.backoff**attempt
                    log.warning("node %s failed (%s), retry %d in %.1fs", spec.name, exc, attempt, delay)
                    time.sleep(delay)
                    continue
                log.exception("node %s failed permanently", spec.name)
                state.record_error(spec.name, exc, recoverable=not spec.critical or bool(spec.on_error))
                return "error", time.time() - started

    # -- docs ---------------------------------------------------------- #
    def to_mermaid(self) -> str:
        """Render the graph from the actual wiring, so the README can't drift."""

        def ident(name: str) -> str:
            return "END" if name == END else name

        lines = ["flowchart TD"]
        for name in self.nodes:
            shape = f'{name}["{name}"]'
            lines.append(f"    {shape}")
        lines.append('    END([" done "])')
        if self.entry:
            lines.append(f"    START([" + '"' + "input" + '"' + f"]) --> {self.entry}")
        for src, dst in self.edges.items():
            lines.append(f"    {src} --> {ident(dst)}")
        for src, (_, mapping) in self.conditional.items():
            for label, dst in mapping.items():
                lines.append(f"    {src} -->|{label}| {ident(dst)}")
        for spec in self.nodes.values():
            if spec.on_error:
                lines.append(f"    {spec.name} -.->|on_error| {ident(spec.on_error)}")
        return "\n".join(lines)
