"""Backend-swap tests.

The flat store is the default in CI, so these cover the *other* half: that the
Chroma path behaves identically behind the same interface, and that swapping
the embedder recalibrates the abstention threshold instead of silently making
the agent refuse every question.
"""

from __future__ import annotations

import pytest

from arxiv_agent.agent import Agent
from arxiv_agent.services.embeddings import HashingEmbedder, cosine
from arxiv_agent.services.vectorstore import ChromaStore, FlatStore, collection_name
from conftest import FakeArxiv

chromadb = pytest.importorskip("chromadb")


@pytest.fixture
def chroma_store(tmp_path):
    return ChromaStore(tmp_path / "chroma")


def _seed(store, embedder, collection):
    texts = [
        "[Section: 3 Method]\nPagedCache partitions the cache into fixed 64-token blocks.",
        "[Section: 4 Experiments]\nWe evaluate on WikiText-103 and LongBench with Llama-3-8B.",
    ]
    store.add(
        collection,
        ids=["a", "b"],
        texts=texts,
        metadatas=[{"section": "3 Method", "low_value": False}, {"section": "4 Experiments", "low_value": False}],
        embeddings=embedder.embed(texts),
    )
    return texts


def test_chroma_roundtrip(chroma_store, embedder):
    collection = collection_name("2402.09876", embedder.name)
    texts = _seed(chroma_store, embedder, collection)
    assert chroma_store.count(collection) == 2

    hits = chroma_store.query(collection, embedder.embed([texts[0]], is_query=True)[0], 2)
    assert hits[0].id == "a"  # exact text is its own nearest neighbour
    assert hits[0].score > 0.9
    assert hits[0].metadata["section"] == "3 Method"
    assert hits[0].embedding and len(hits[0].embedding) == embedder.dim  # MMR needs these


def test_chroma_upsert_is_idempotent(chroma_store, embedder):
    collection = collection_name("2402.09876", embedder.name)
    _seed(chroma_store, embedder, collection)
    _seed(chroma_store, embedder, collection)
    assert chroma_store.count(collection) == 2


def test_chroma_drop_clears_the_collection(chroma_store, embedder):
    collection = collection_name("2402.09876", embedder.name)
    _seed(chroma_store, embedder, collection)
    chroma_store.drop(collection)
    assert chroma_store.count(collection) == 0


def test_both_backends_rank_identically(chroma_store, embedder, tmp_path):
    flat = FlatStore(tmp_path / "flat")
    collection = collection_name("2402.09876", embedder.name)
    texts = _seed(chroma_store, embedder, collection)
    _seed(flat, embedder, collection)

    query = embedder.embed(["which datasets were used for evaluation"], is_query=True)[0]
    assert [h.id for h in chroma_store.query(collection, query, 2)] == [
        h.id for h in flat.query(collection, query, 2)
    ]
    assert texts  # sanity


def test_collection_name_is_namespaced_by_embedder():
    a = collection_name("2402.09876", "hashing:384")
    b = collection_name("2402.09876", "sentence-transformers/all-MiniLM-L6-v2")
    assert a != b  # two vector spaces must never share a collection
    assert a.startswith("p_") and " " not in a


def test_full_pipeline_on_chroma(settings, mock_llm, embedder, chroma_store, sample_pdf):
    settings.min_similarity = 0.0
    agent = Agent(
        settings,
        llm=mock_llm,
        embedder=embedder,
        store=chroma_store,
        arxiv=FakeArxiv(pdf_path=sample_pdf),
        emit=lambda *_: None,
    )
    state = agent.brief("2402.09876")
    assert state.status == "briefed"
    assert state.vector_backend == "chroma"
    assert chroma_store.count(state.collection) == state.chunk_count

    state = agent.ask(state, "How are cache blocks scored?")
    assert state.qa_history[-1]["chunks"]


def test_fallback_embedder_lowers_the_abstention_threshold(settings, mock_llm, store, monkeypatch):
    monkeypatch.delenv("MIN_SIMILARITY", raising=False)
    settings.min_similarity = 0.15
    Agent(settings, llm=mock_llm, embedder=HashingEmbedder(), store=store, arxiv=object(), emit=lambda *_: None)
    assert settings.min_similarity == 0.02


def test_explicit_threshold_is_respected(settings, mock_llm, store, monkeypatch):
    monkeypatch.setenv("MIN_SIMILARITY", "0.4")
    settings.min_similarity = 0.4
    Agent(settings, llm=mock_llm, embedder=HashingEmbedder(), store=store, arxiv=object(), emit=lambda *_: None)
    assert settings.min_similarity == 0.4


def test_normalised_vectors_make_cosine_a_dot_product(embedder):
    a, b = embedder.embed(["attention is all you need", "attention is all you need"])
    assert cosine(a, b) == pytest.approx(1.0, abs=1e-6)
