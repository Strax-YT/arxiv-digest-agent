"""The facade the CLI (or a notebook) talks to."""

from __future__ import annotations

import logging
import os
from typing import Callable

from .config import Settings
from .graph.build import build_briefing_graph, build_qa_graph
from .graph.checkpoint import FileCheckpointer
from .graph.engine import Context
from .services.arxiv_client import ArxivClient
from .services.embeddings import build_embedder
from .services.llm import LLMUnavailable, build_llm
from .state import AgentState

log = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        settings: Settings,
        *,
        llm=None,
        embedder=None,
        store=None,
        arxiv=None,
        ask_user: Callable[[str, list[str]], int] | None = None,
        emit: Callable[[str], None] = print,
    ) -> None:
        settings.ensure_dirs()
        self.settings = settings
        self.checkpointer = FileCheckpointer(settings.sessions_dir)

        if store is None:
            from .services.vectorstore import build_store

            store = build_store(settings)

        embedder = embedder if embedder is not None else build_embedder(settings)
        if getattr(embedder, "degraded", False) and "MIN_SIMILARITY" not in os.environ:
            # The abstention threshold is calibrated for dense cosine scores.
            # The lexical fallback scores far lower, so keeping 0.15 would make
            # it refuse to answer almost anything. Lower the bar and say so.
            settings.min_similarity = 0.02
            log.warning(
                "using the lexical fallback embedder; lowering MIN_SIMILARITY to %.2f "
                "(set MIN_SIMILARITY explicitly to override)",
                settings.min_similarity,
            )

        self.ctx = Context(
            settings=settings,
            llm=llm if llm is not None else build_llm(settings),
            embedder=embedder,
            store=store,
            arxiv=arxiv if arxiv is not None else ArxivClient(settings),
            ask_user=ask_user,
            emit=emit,
        )
        self.briefing_graph = build_briefing_graph()
        self.qa_graph = build_qa_graph()

    # ------------------------------------------------------------------ #
    def brief(self, user_input: str) -> AgentState:
        state = AgentState(raw_input=user_input.strip())
        self.ctx.emit(f"Session {state.session_id} · {self.settings.provider}:{self.settings.model}")
        return self.briefing_graph.invoke(state, self.ctx, checkpointer=self.checkpointer)

    def ask(self, state: AgentState, question: str) -> AgentState:
        if not state.collection:
            raise RuntimeError(
                f"session {state.session_id} has no indexed chunks — run a briefing first"
            )
        state.pending_question = question
        return self.qa_graph.invoke(state, self.ctx, checkpointer=self.checkpointer)

    def load(self, session_id: str) -> AgentState:
        return self.checkpointer.load(session_id)

    def latest(self) -> AgentState | None:
        return self.checkpointer.latest()

    def sessions(self) -> list[dict]:
        return self.checkpointer.list_sessions()


def try_build_llm(settings: Settings, emit: Callable[[str], None] = print):
    """Build an LLM but tolerate absence: query planning and ranking can degrade."""
    try:
        return build_llm(settings)
    except LLMUnavailable as exc:
        emit(f"\n! LLM unavailable: {exc}\n")
        raise
