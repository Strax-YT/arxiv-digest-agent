#!/usr/bin/env python3
"""A complete, zero-dependency-on-the-internet run of the agent.

    python scripts/demo_offline.py

It exercises the *real* graph, the real PDF parser, the real chunker, the real
vector store and the real QA path. Only two things are faked: the arXiv API
(a canned Atom feed + a PDF generated on the fly) and the LLM (a scripted
stand-in). That makes it a reproducible smoke test of the plumbing that anyone
can run in five seconds without Ollama, an API key or a network connection.

For a real run against real arXiv, see `scripts/capture_example.sh`.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    # cp1252 consoles cannot print the trace arrows.
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from conftest import BRIEFING_JSON, FakeArxiv, make_pdf  # noqa: E402

from arxiv_agent.agent import Agent  # noqa: E402
from arxiv_agent.config import Settings  # noqa: E402
from arxiv_agent.services.embeddings import HashingEmbedder  # noqa: E402
from arxiv_agent.services.llm import MockLLM  # noqa: E402
from arxiv_agent.services.vectorstore import FlatStore  # noqa: E402

RULE = "=" * 78


class DemoLLM(MockLLM):
    """Answers QA prompts from the retrieved context by keyword, so the demo
    shows real retrieval + real citation checking without a real model."""

    def _request(self, system: str, user: str, json_mode: bool, temperature: float) -> str:
        self.prompts.append(user)
        for key, value in self.responses.items():
            if key in user:
                return value
        if "Question:" in user:
            question = user.split("Question:")[-1].strip().lower()
            if "block" in question or "partition" in question:
                return (
                    "PagedCache splits the KV cache into fixed blocks of 64 tokens and scores "
                    "each block by the mean attention mass it receives over a sliding window of "
                    "512 queries [C1]. The top 12% of blocks stay in fp16; the rest are "
                    "quantized to 4 bits with per-block affine scales [C1]."
                )
            if "dataset" in question or "baseline" in question or "evaluat" in question:
                return (
                    "Evaluation covers WikiText-103, LongBench and Needle-in-a-Haystack with "
                    "Llama-3-8B and Mistral-7B, against H2O, StreamingLLM and a full-cache fp16 "
                    "baseline [C1]. PagedCache reaches 4.1x memory reduction versus 2.2x for H2O "
                    "at equal quality [C1]."
                )
            if "carbon" in question or "cost" in question or "energy" in question:
                return (
                    "Not found in the retrieved sections of this paper. The retrieved text covers "
                    "memory and throughput measurements but never reports training or inference "
                    "energy use."
                )
            return "Not found in the retrieved sections of this paper."
        return "{}" if json_mode else "MOCK"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="arxiv-demo-"))
    settings = Settings(provider="mock", model="demo", embedding_backend="hashing")
    settings.workdir = tmp / "home"
    settings.min_similarity = 0.0  # the lexical fallback embedder scores low
    settings.interactive = False
    settings.ensure_dirs()

    pdf = make_pdf(tmp / "pagedcache.pdf")
    llm = DemoLLM(
        settings,
        {
            "Write an executive briefing": BRIEFING_JSON,
            "Convert this research request": json.dumps(
                {
                    "phrases": ["KV cache compression", "long context inference"],
                    "terms": ["kv", "cache", "quantization", "llm"],
                    "categories": ["cs.CL", "cs.LG"],
                    "since_year": 2024,
                    "notes": "wants recent method papers on shrinking the KV cache",
                }
            ),
            "Score each candidate": json.dumps(
                {
                    "scores": [
                        {"index": 0, "score": 0.95, "reason": "concrete compression method"},
                        {"index": 1, "score": 0.35, "reason": "broad survey, not a method"},
                    ]
                }
            ),
        },
    )

    agent = Agent(
        settings,
        llm=llm,
        embedder=HashingEmbedder(),
        store=FlatStore(settings.vectorstore_dir),
        arxiv=FakeArxiv(pdf_path=pdf),
    )

    print(RULE)
    print("INPUT:  recent work on KV-cache compression for LLMs")
    print(RULE)
    state = agent.brief("recent work on KV-cache compression for LLMs")
    if state.status != "briefed":
        print(f"run ended early: {state.status} — {state.halt_reason}")
        return 1

    print("\n--- what the graph did ---")
    for step in state.trace:
        if "outcome" in step:
            print(f"  {step['node']:<18} {step['outcome']:<6} {step['seconds']:>6.2f}s")
        else:
            print(f"      ↳ {step['note']}")

    print("\n--- queries sent to the arXiv API ---")
    for attempt in state.query_attempts:
        print(f"  L{attempt['level']}: {attempt['query']}  -> {attempt.get('hits')} hits")

    print("\n--- selection ---")
    print(f"  {state.selection_rationale}")
    for alt in state.runners_up:
        print(f"  runner-up: {alt['score']:.3f}  {alt['title'][:60]}")

    print("\n" + RULE)
    print(Path(state.briefing_md_path).read_text(encoding="utf-8"))
    print(RULE)

    for question in (
        "How does the block scoring actually work?",
        "What datasets and baselines did they compare against?",
        "What was the carbon cost of training?",
    ):
        state = agent.ask(state, question)
        turn = state.qa_history[-1]
        print(f"\nyou > {question}")
        print(f"\n{turn['answer']}\n")
        if turn["citations"]:
            print("Sources:")
            for c in turn["citations"]:
                print(f"  [{c['label']}] {c['section']} (p.{c['pages'][0]}-{c['pages'][1]}) · sim {c['score']}")
        elif turn["abstained"]:
            print("(abstained — nothing similar enough was retrieved)")
        print(f"  grounded={turn['grounded']}  abstained={turn['abstained']}  top_sim={turn['top_score']}")

    print("\n" + RULE)
    print(f"session:    {state.session_id}")
    print(f"state file: {settings.sessions_dir / state.session_id / 'state.json'}")
    print(f"qa turns:   {len(state.qa_history)}  (all persisted)")
    print(RULE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
