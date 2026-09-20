"""Shared fixtures.

The whole graph is exercised without touching the network: a fake arXiv client
serving a canned Atom feed, a synthetic PDF built at test time, the hashing
embedder, the flat vector store, and a scripted mock LLM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arxiv_agent.config import Settings  # noqa: E402
from arxiv_agent.services.arxiv_client import ArxivClient  # noqa: E402
from arxiv_agent.services.embeddings import HashingEmbedder  # noqa: E402
from arxiv_agent.services.llm import MockLLM  # noqa: E402
from arxiv_agent.services.vectorstore import FlatStore  # noqa: E402
from arxiv_agent.state import PaperMeta  # noqa: E402

ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2402.09876v2</id>
    <updated>2024-03-01T10:00:00Z</updated>
    <published>2024-02-15T09:00:00Z</published>
    <title>PagedCache: Block-Sparse KV Cache Compression for Long-Context LLMs</title>
    <summary>We introduce PagedCache, a block-sparse compression scheme for the
    key-value cache of autoregressive transformers. PagedCache reduces memory by
    4.1x at 128k context with 0.3 perplexity degradation on WikiText-103.</summary>
    <author><name>A. Researcher</name></author>
    <author><name>B. Coauthor</name></author>
    <arxiv:comment>14 pages, 6 figures</arxiv:comment>
    <link href="http://arxiv.org/abs/2402.09876v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2402.09876v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2309.11111v1</id>
    <updated>2023-09-20T10:00:00Z</updated>
    <published>2023-09-20T10:00:00Z</published>
    <title>A Survey of Memory-Efficient Inference</title>
    <summary>This survey reviews memory-efficient inference techniques.</summary>
    <author><name>C. Surveyor</name></author>
    <link title="pdf" href="http://arxiv.org/pdf/2309.11111v1" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
  </entry>
</feed>
"""

EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>ArXiv Query</title></feed>
"""

PAPER_BODY = {
    "Abstract": (
        "We introduce PagedCache, a block-sparse compression scheme for the key-value cache "
        "of autoregressive transformers. PagedCache reduces memory by 4.1x at 128k context "
        "with 0.3 perplexity degradation on WikiText-103."
    ),
    "1 Introduction": (
        "Serving long-context language models is bottlenecked by the key-value cache, which "
        "grows linearly with sequence length and quickly dominates GPU memory. Existing "
        "eviction policies discard tokens irreversibly and degrade recall on retrieval tasks. "
        "We ask whether the cache can instead be compressed in place. "
    ) * 3,
    "3 Method": (
        "PagedCache partitions the cache into fixed blocks of 64 tokens. Each block is scored "
        "by the mean attention mass it receives over a sliding window of 512 queries. "
        "Low-scoring blocks are quantized to 4 bits using per-block affine scales, while the "
        "top 12 percent of blocks are retained in fp16. A lightweight router reconstructs "
        "blocks on demand during decoding, adding 3 percent latency overhead. "
    ) * 4,
    "4 Experiments": (
        "We evaluate on WikiText-103, LongBench and Needle-in-a-Haystack using Llama-3-8B and "
        "Mistral-7B. Baselines are H2O, StreamingLLM and full-cache fp16. PagedCache achieves "
        "4.1x memory reduction at 128k context with 0.3 perplexity degradation, versus 2.2x "
        "for H2O at equal quality. Throughput improves by 1.7x at batch size 32. "
    ) * 4,
    "6 Limitations": (
        "Our evaluation covers only English text and two model families below 10B parameters. "
        "We did not test on mixture-of-experts architectures, and the router adds a small but "
        "measurable latency cost at short context lengths."
    ),
    "References": (
        "[1] Smith, J., Doe, A. and Roe, P. Efficient Transformers for Long Sequences. "
        "In Proceedings of NeurIPS, 2023.\n"
        "[2] Lee, K. and Park, S. StreamingLLM: Efficient Streaming Language Models with "
        "Attention Sinks. In ICLR, 2024.\n"
    ),
}


class FakeArxiv:
    """Drop-in for ArxivClient with no network."""

    def __init__(self, feed: str = ATOM_FEED, pdf_path: Path | None = None) -> None:
        self.feed = feed
        self.pdf_path = pdf_path
        self.queries: list[str] = []

    def search(self, query: str, max_results: int = 20, sort: str = "relevance"):
        self.queries.append(query)
        return ArxivClient._parse_feed(self.feed)

    def get_by_id(self, arxiv_id: str):
        papers = ArxivClient._parse_feed(self.feed)
        return next((p for p in papers if p.arxiv_id.startswith(arxiv_id)), None)

    def download_pdf(self, paper, dest_dir):
        if self.pdf_path is None:
            raise RuntimeError("simulated download failure")
        return self.pdf_path


class EmptyThenFullArxiv(FakeArxiv):
    """Returns nothing until the relaxation ladder reaches `succeed_at`."""

    def __init__(self, succeed_at: int = 2, **kw) -> None:
        super().__init__(**kw)
        self.succeed_at = succeed_at

    def search(self, query: str, max_results: int = 20, sort: str = "relevance"):
        self.queries.append(query)
        if len(self.queries) <= self.succeed_at:
            return ArxivClient._parse_feed(EMPTY_FEED)
        return ArxivClient._parse_feed(self.feed)


def make_pdf(path: Path, sections: dict[str, str] | None = None) -> Path:
    """Render a small LaTeX-ish paper to a real PDF so parsing is tested for real."""
    try:
        fitz = pytest.importorskip("pymupdf")
    except Exception:  # pragma: no cover - older PyMuPDF
        fitz = pytest.importorskip("fitz")
    sections = sections or PAPER_BODY
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    page.insert_text((60, y), "PagedCache: Block-Sparse KV Cache Compression", fontsize=16, fontname="hebo")
    y += 40
    for heading, body in sections.items():
        if y > 700:
            page = doc.new_page()
            y = 60
        page.insert_text((60, y), heading, fontsize=13, fontname="hebo")
        y += 20
        wrapped = fitz.TextWriter(page.rect)
        rect = fitz.Rect(60, y, 540, min(y + 320, 780))
        rc = page.insert_textbox(rect, body, fontsize=9, fontname="helv")
        y = rect.y1 + 20 if rc >= 0 else y + 320
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(provider="mock", model="mock", embedding_backend="hashing", vector_backend="numpy")
    s.workdir = tmp_path / "home"
    s.ensure_dirs()
    s.interactive = False
    return s


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def store(tmp_path):
    return FlatStore(tmp_path / "vs")


@pytest.fixture
def paper() -> PaperMeta:
    return ArxivClient._parse_feed(ATOM_FEED)[0]


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    return make_pdf(tmp_path / "paper.pdf")


BRIEFING_JSON = json.dumps(
    {
        "why_it_matters": (
            "Long-context serving is memory bound, and this work shows the KV cache can be "
            "compressed in place rather than evicted, keeping recall while cutting memory 4x. "
            "That matters to anyone running long-context inference on fixed hardware budgets."
        ),
        "problem_statement": (
            "The KV cache grows linearly with context length and dominates GPU memory. "
            "Existing eviction policies discard tokens permanently and hurt retrieval recall."
        ),
        "method": [
            "Partition the KV cache into fixed 64-token blocks",
            "Score blocks by mean attention mass over a 512-query sliding window",
            "Quantize low-scoring blocks to 4 bits with per-block affine scales",
            "Keep the top 12 percent of blocks in fp16",
        ],
        "key_results": [
            {"claim": "4.1x memory reduction at 128k context with 0.3 PPL degradation", "evidence": "C4"},
            {"claim": "1.7x throughput improvement at batch size 32", "evidence": "C4"},
        ],
        "limitations": [
            "English-only evaluation",
            "Only two model families below 10B parameters",
            "No mixture-of-experts results",
        ],
        "follow_up_questions": [
            "How does PagedCache interact with speculative decoding?",
            "What is the overhead at short context lengths?",
        ],
        "confidence": "high",
        "confidence_reason": "method and results sections parsed cleanly",
    }
)


@pytest.fixture
def mock_llm(settings):
    return MockLLM(
        settings,
        {
            "Write an executive briefing": BRIEFING_JSON,
            "Convert this research request": json.dumps(
                {
                    "phrases": ["KV cache compression", "long context"],
                    "terms": ["kv", "cache", "compression", "llm"],
                    "categories": ["cs.CL", "cs.LG"],
                    "authors": [],
                    "since_year": 2024,
                    "notes": "user wants recent KV-cache compression work",
                }
            ),
            "Score each candidate": json.dumps(
                {"scores": [{"index": 0, "score": 0.95, "reason": "direct method paper"},
                            {"index": 1, "score": 0.3, "reason": "broad survey"}]}
            ),
        },
    )
