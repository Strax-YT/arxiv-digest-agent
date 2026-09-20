"""Command-line interface.

    python -m arxiv_agent brief "KV-cache compression for LLMs"
    python -m arxiv_agent brief 2401.12345 --no-chat
    python -m arxiv_agent chat <session_id>
    python -m arxiv_agent ask  <session_id> "what datasets did they use?"
    python -m arxiv_agent sessions
    python -m arxiv_agent show <session_id> [--json]
    python -m arxiv_agent graph            # mermaid diagram of the state graph
    python -m arxiv_agent doctor           # check the local environment
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from .agent import Agent
from .config import DEFAULT_BASE_URLS, DEFAULT_MODELS, Settings
from .graph.build import build_briefing_graph
from .services.llm import LLMUnavailable
from .state import AgentState

BANNER = "arXiv Digest & QA Agent"


# --------------------------------------------------------------------- #
def _load_dotenv() -> None:
    """Minimal .env loader so the project has no python-dotenv dependency."""
    for path in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
        return


def _settings(args: argparse.Namespace) -> Settings:
    _load_dotenv()
    if getattr(args, "provider", None):
        os.environ["LLM_PROVIDER"] = args.provider
        os.environ.pop("LLM_BASE_URL", None)
        if not getattr(args, "model", None):
            os.environ["LLM_MODEL"] = DEFAULT_MODELS.get(args.provider, "")
            os.environ["LLM_BASE_URL"] = DEFAULT_BASE_URLS.get(args.provider, "")
    if getattr(args, "model", None):
        os.environ["LLM_MODEL"] = args.model
    settings = Settings.from_env()
    if getattr(args, "workdir", None):
        settings.workdir = Path(args.workdir).expanduser()
    if getattr(args, "max_pages", None):
        settings.max_pdf_pages = args.max_pages
    if getattr(args, "no_chat", False) or getattr(args, "json", False):
        settings.interactive = False
    return settings


def _ask_user(prompt: str, options: list[str]) -> int:
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"  {i}) {option}")
    while True:
        raw = input(f"Choose 1-{len(options)} [1]: ").strip() or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return int(raw) - 1
        print("  Please enter one of the listed numbers.")


def _print_briefing(state: AgentState) -> None:
    if state.briefing_md_path and Path(state.briefing_md_path).exists():
        print("\n" + Path(state.briefing_md_path).read_text(encoding="utf-8"))
        print(f"[saved] {state.briefing_md_path}")
        print(f"[saved] {state.briefing_json_path}\n")


def _print_answer(turn: dict) -> None:
    print(f"\n{turn['answer']}\n")
    if turn.get("citations"):
        print("Sources:")
        for c in turn["citations"]:
            pages = "-".join(str(p) for p in c["pages"] if p is not None)
            print(f"  [{c['label']}] {c['section']} (p.{pages}) · similarity {c['score']}")
    elif turn.get("abstained"):
        print("(no sufficiently similar passage was found — the agent declined to guess)")
    if not turn.get("grounded"):
        print("  !! ungrounded answer — no valid citations were produced")
    print()


# --------------------------------------------------------------------- #
def cmd_brief(args: argparse.Namespace) -> int:
    settings = _settings(args)
    try:
        agent = Agent(settings, ask_user=_ask_user)
    except LLMUnavailable as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    started = time.time()
    state = agent.brief(args.query)

    if state.status != "briefed":
        return 1
    print(f"\nBriefed in {time.time() - started:.1f}s")
    _print_briefing(state)

    if args.no_chat or not sys.stdin.isatty():
        print(f"Ask follow-ups with: python -m arxiv_agent chat {state.session_id}")
        return 0
    return _chat_loop(agent, state)


def cmd_chat(args: argparse.Namespace) -> int:
    settings = _settings(args)
    agent = Agent(settings)
    state = agent.load(args.session_id) if args.session_id else agent.latest()
    if state is None:
        print("No sessions yet. Run `brief` first.", file=sys.stderr)
        return 1
    paper = state.paper_meta
    print(f"\n{paper.title}\narXiv:{paper.arxiv_id} · {state.chunk_count} chunks indexed\n")
    return _chat_loop(agent, state)


def _chat_loop(agent: Agent, state: AgentState) -> int:
    print("QA mode — grounded in the indexed paper. Type 'exit' to quit, 'suggest' for ideas.\n")
    while True:
        try:
            question = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            break
        if question.lower() == "suggest":
            for q in (state.briefing or {}).get("follow_up_questions", []):
                print(f"  - {q}")
            print()
            continue
        try:
            state = agent.ask(state, question)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            print(f"  (error: {exc})\n")
            continue
        if state.qa_history:
            _print_answer(state.qa_history[-1])
    print(f"Session saved: {state.session_id}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    settings = _settings(args)
    agent = Agent(settings)
    state = agent.load(args.session_id) if args.session_id else agent.latest()
    if state is None:
        print("No sessions yet. Run `brief` first.", file=sys.stderr)
        return 1
    state = agent.ask(state, args.question)
    turn = state.qa_history[-1] if state.qa_history else {}
    if args.json:
        import json

        print(json.dumps(turn, indent=2, ensure_ascii=False))
    else:
        _print_answer(turn)
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    agent = Agent(_settings(args), llm=_NoLLM(), embedder=_NoEmbedder(), arxiv=object())
    rows = agent.sessions()
    if not rows:
        print("No sessions yet.")
        return 0
    print(f"{'SESSION':<22} {'STATUS':<10} {'ARXIV':<12} {'Q':<3} TITLE")
    for row in rows:
        print(
            f"{row['session_id']:<22} {row['status']:<10} {row['arxiv_id']:<12} "
            f"{row['questions']:<3} {(row['title'] or row['input'])[:60]}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    agent = Agent(_settings(args), llm=_NoLLM(), embedder=_NoEmbedder(), arxiv=object())
    state = agent.load(args.session_id) if args.session_id else agent.latest()
    if state is None:
        print("No sessions yet.", file=sys.stderr)
        return 1
    path = Path(state.briefing_json_path if args.json else state.briefing_md_path)
    if not path.exists():
        print(f"session {state.session_id} has no briefing ({state.status}: {state.halt_reason})")
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    print("```mermaid")
    print(build_briefing_graph().to_mermaid())
    print("```")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings(args)
    print(f"{BANNER} — environment check\n")
    print(f"workdir           : {settings.workdir}")
    print(f"LLM provider      : {settings.provider} ({settings.model})")
    print(f"LLM base url      : {settings.base_url or 'n/a'}")
    print(f"API key present   : {bool(settings.api_key)}")

    for module, label in (
        ("pymupdf", "PyMuPDF (PDF parsing)"),
        ("pdfplumber", "pdfplumber (fallback parser)"),
        ("chromadb", "Chroma (vector store)"),
        ("sentence_transformers", "sentence-transformers (embeddings)"),
    ):
        try:
            __import__(module)
            print(f"{label:<34}: installed")
        except Exception as exc:  # noqa: BLE001
            print(f"{label:<34}: MISSING ({type(exc).__name__}) -> a fallback will be used")

    try:
        from .services.embeddings import build_embedder

        emb = build_embedder(settings)
        print(f"embedder resolved : {emb.name} (dim {emb.dim})")
    except Exception as exc:  # noqa: BLE001
        print(f"embedder resolved : FAILED {exc}")

    try:
        from .services.vectorstore import build_store

        print(f"vector store      : {build_store(settings).backend}")
    except Exception as exc:  # noqa: BLE001
        print(f"vector store      : FAILED {exc}")

    try:
        from .services.llm import build_llm

        llm = build_llm(settings)
        reply = llm.complete("Reply with the single word: ready", "You are terse.")
        print(f"LLM round-trip    : ok -> {reply.strip()[:40]!r}")
    except Exception as exc:  # noqa: BLE001
        print(f"LLM round-trip    : FAILED {exc}")
        return 1
    return 0


class _NoLLM:  # stand-ins for read-only commands that must not need a model
    def complete(self, *a, **k):
        raise RuntimeError("no LLM configured for this command")

    def complete_json(self, *a, **k):
        raise RuntimeError("no LLM configured for this command")


class _NoEmbedder:
    name = "none"
    dim = 0

    def embed(self, texts, is_query: bool = False):
        raise RuntimeError("no embedder configured for this command")


# --------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arxiv_agent", description=BANNER)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--provider", choices=["ollama", "groq", "gemini", "openrouter", "mock"])
        p.add_argument("--model")
        p.add_argument("--workdir")

    p_brief = sub.add_parser("brief", help="fetch, read and brief a paper")
    p_brief.add_argument("query", help="a topic, an arXiv id, or an arxiv.org URL")
    p_brief.add_argument("--no-chat", action="store_true", help="exit after the briefing")
    p_brief.add_argument("--max-pages", type=int, help="page cap for PDF parsing")
    common(p_brief)
    p_brief.set_defaults(func=cmd_brief)

    p_chat = sub.add_parser("chat", help="interactive grounded QA over a session")
    p_chat.add_argument("session_id", nargs="?")
    common(p_chat)
    p_chat.set_defaults(func=cmd_chat)

    p_ask = sub.add_parser("ask", help="one grounded question, then exit")
    p_ask.add_argument("session_id", nargs="?")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true")
    common(p_ask)
    p_ask.set_defaults(func=cmd_ask)

    p_sessions = sub.add_parser("sessions", help="list saved sessions")
    common(p_sessions)
    p_sessions.set_defaults(func=cmd_sessions)

    p_show = sub.add_parser("show", help="print a saved briefing")
    p_show.add_argument("session_id", nargs="?")
    p_show.add_argument("--json", action="store_true")
    common(p_show)
    p_show.set_defaults(func=cmd_show)

    p_graph = sub.add_parser("graph", help="print the state graph as mermaid")
    p_graph.set_defaults(func=cmd_graph)

    p_doctor = sub.add_parser("doctor", help="check local setup")
    common(p_doctor)
    p_doctor.set_defaults(func=cmd_doctor)
    return parser


def _force_utf8_stdio() -> None:
    """Keep the briefing printable on a legacy-codepage console.

    Windows hands Python a cp1252 stdout, which dies on the em dashes and box
    rules the briefing is full of.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except BrokenPipeError:
        # `... | head` closed the pipe. Silence the interpreter's shutdown noise.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
