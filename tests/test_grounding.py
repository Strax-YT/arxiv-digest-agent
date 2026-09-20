"""The grounding tests are the point of the whole exercise, so they are explicit
about the three failure modes we care about: answering from memory, answering
with no evidence, and answering confidently when the paper is silent."""

from __future__ import annotations

from arxiv_agent.graph.engine import Context
from arxiv_agent.nodes.qa import (
    CITATION_RE,
    NOT_FOUND,
    answer_question,
    mmr,
    normalize_citations,
    verify_citation_support,
)
from arxiv_agent.services.llm import MockLLM
from arxiv_agent.services.vectorstore import Hit
from arxiv_agent.state import AgentState


def _seed(store, embedder, collection="test_col", settings=None):
    """Index three chunks. Callers that want to exercise the *citation* logic
    pass settings so the similarity gate is opened; the gate has its own tests."""
    if settings is not None:
        settings.min_similarity = 0.0
    chunks = [
        ("c1", "[Section: 4 Experiments]\nWe evaluate on WikiText-103 and LongBench using Llama-3-8B.",
         {"section": "4 Experiments", "page_start": 5, "page_end": 6, "low_value": False}),
        ("c2", "[Section: 3 Method]\nPagedCache partitions the cache into fixed blocks of 64 tokens.",
         {"section": "3 Method", "page_start": 3, "page_end": 4, "low_value": False}),
        ("c3", "[Section: References]\n[1] Smith. Efficient Transformers. 2023.",
         {"section": "References", "page_start": 12, "page_end": 12, "low_value": True}),
    ]
    store.add(
        collection,
        ids=[c[0] for c in chunks],
        texts=[c[1] for c in chunks],
        metadatas=[c[2] for c in chunks],
        embeddings=embedder.embed([c[1] for c in chunks]),
    )
    return collection


def _state(collection, **kw):
    state = AgentState(collection=collection, chunk_count=3, **kw)
    state.paper = {"arxiv_id": "2402.09876", "title": "PagedCache", "authors": [], "abstract": ""}
    return state


def test_answer_cites_retrieved_chunks(settings, store, embedder):
    collection = _seed(store, embedder, settings=settings)
    llm = MockLLM(settings, {"Question:": "They evaluate on WikiText-103 and LongBench [C1]."})
    state = _state(collection, pending_question="What datasets are used?")
    answer_question(state, Context(settings=settings, llm=llm, embedder=embedder, store=store))

    turn = state.qa_history[-1]
    assert turn["grounded"] is True
    assert turn["citations"][0]["label"] == "C1"
    assert turn["citations"][0]["chunk_id"] in {"c1", "c2", "c3"}


def test_no_retrieval_hits_abstains_without_calling_the_llm(settings, store, embedder):
    llm = MockLLM(settings, {"Question:": "I happen to know the answer is 42."})
    state = _state("empty_collection", pending_question="What is the batch size?")
    answer_question(state, Context(settings=settings, llm=llm, embedder=embedder, store=store))

    turn = state.qa_history[-1]
    assert turn["abstained"] is True
    assert NOT_FOUND in turn["answer"]
    assert llm.prompts == []  # the gate fired before any generation


def test_low_similarity_abstains(settings, store, embedder):
    collection = _seed(store, embedder)
    settings.min_similarity = 0.99  # nothing can clear this bar
    llm = MockLLM(settings, {"Question:": "Definitely 1024 tokens [C1]."})
    state = _state(collection, pending_question="What is the melting point of tungsten?")
    answer_question(state, Context(settings=settings, llm=llm, embedder=embedder, store=store))

    assert state.qa_history[-1]["abstained"] is True
    assert llm.prompts == []


def test_uncited_answer_is_retried_then_flagged(settings, store, embedder):
    collection = _seed(store, embedder, settings=settings)
    llm = MockLLM(
        settings,
        {
            "previous answer cited no excerpts": "Still no citation.",
            "Question:": "It uses 64-token blocks.",
        },
    )
    state = _state(collection, pending_question="How big are the blocks?")
    answer_question(state, Context(settings=settings, llm=llm, embedder=embedder, store=store))

    turn = state.qa_history[-1]
    assert len(llm.prompts) == 2  # one generation + one strict retry
    assert turn["grounded"] is False
    assert turn["answer"].startswith("⚠️")


def test_model_abstention_is_accepted_as_grounded(settings, store, embedder):
    collection = _seed(store, embedder, settings=settings)
    llm = MockLLM(settings, {"Question:": NOT_FOUND + " The paper discusses block scoring instead."})
    state = _state(collection, pending_question="What is the training cost?")
    answer_question(state, Context(settings=settings, llm=llm, embedder=embedder, store=store))

    turn = state.qa_history[-1]
    assert turn["grounded"] is True
    assert turn["abstained"] is True


def test_followup_is_condensed_using_history(settings, store, embedder):
    collection = _seed(store, embedder, settings=settings)
    llm = MockLLM(
        settings,
        {
            "Rewrite the follow-up": "What block size does PagedCache use?",
            "Question:": "Blocks of 64 tokens [C1].",
        },
    )
    state = _state(collection, pending_question="and how big is it?")
    state.qa_history.append({"question": "What is PagedCache?", "answer": "A KV cache scheme [C2]."})
    answer_question(state, Context(settings=settings, llm=llm, embedder=embedder, store=store))

    turn = state.qa_history[-1]
    assert turn["standalone"] == "What block size does PagedCache use?"
    assert any("Rewrite the follow-up" in p for p in llm.prompts)


def test_first_question_is_not_condensed(settings, store, embedder):
    collection = _seed(store, embedder, settings=settings)
    llm = MockLLM(settings, {"Question:": "Yes [C1]."})
    state = _state(collection, pending_question="what is it?")
    answer_question(state, Context(settings=settings, llm=llm, embedder=embedder, store=store))
    assert not any("Rewrite the follow-up" in p for p in llm.prompts)


def test_mmr_prefers_diverse_chunks():
    a = Hit(id="a", text="x", metadata={}, score=0.90, embedding=[1.0, 0.0])
    near_duplicate = Hit(id="b", text="x", metadata={}, score=0.89, embedding=[0.99, 0.01])
    different = Hit(id="c", text="y", metadata={}, score=0.70, embedding=[0.0, 1.0])
    picked = mmr([a, near_duplicate, different], [1.0, 0.0], k=2, lambda_=0.5)
    assert [h.id for h in picked] == ["a", "c"]


def test_mmr_demotes_bibliography_chunks():
    body = Hit(id="body", text="x", metadata={"low_value": False}, score=0.60, embedding=[1.0, 0.0])
    refs = Hit(id="refs", text="y", metadata={"low_value": True}, score=0.65, embedding=[0.0, 1.0])
    picked = mmr([refs, body], [1.0, 0.0], k=1, lambda_=1.0)
    assert picked[0].id == "body"


def test_qa_history_persists_across_turns(settings, store, embedder):
    collection = _seed(store, embedder, settings=settings)
    llm = MockLLM(settings, {"Question:": "Answer [C1]."})
    ctx = Context(settings=settings, llm=llm, embedder=embedder, store=store)
    state = _state(collection, pending_question="q1")
    answer_question(state, ctx)
    state.pending_question = "q2"
    answer_question(state, ctx)
    assert len(state.qa_history) == 2
    assert state.pending_question == ""


# ------------------------------------------------------------------ #
# citation parsing and support
# ------------------------------------------------------------------ #


def test_normalize_citations_accepts_bracket_variants():
    """gpt-oss emits fullwidth/CJK brackets. A citation is a citation."""
    out = normalize_citations("see 【C3】 and ［C10］ plus [C1]")
    assert CITATION_RE.findall(out) == ["C3", "C10", "C1"]


def test_normalize_citations_leaves_plain_text_alone():
    assert normalize_citations("no citations here") == "no citations here"


class _StubEmbedder:
    """Two orthogonal topics, so support is unambiguous in the test."""

    dim = 2

    def embed(self, texts, *, is_query=False):
        out = []
        for t in texts:
            low = t.lower()
            out.append([1.0, 0.0] if "cache" in low or "block" in low else [0.0, 1.0])
        return out


def _ctx_with(threshold):
    class _S:
        citation_support_min = threshold

    class _C:
        settings = _S()
        embedder = _StubEmbedder()

    return _C()


def _label_map():
    return {
        "C1": Hit(id="c1", text="PagedCache partitions the cache into blocks", metadata={}, score=0.9)
    }


def test_citation_support_reports_no_caveat_for_a_supported_claim():
    answer = "The method partitions the cache into fixed blocks of 64 tokens [C1]."
    cited, weak = verify_citation_support(answer, _label_map(), _ctx_with(0.4))
    assert cited == ["C1"] and weak == []


def test_citation_support_flags_a_claim_the_chunk_does_not_back():
    """The label is real and retrieved; the cited text simply does not say this.

    It stays cited — the check advises, it does not overrule — but it is named
    so the reader knows which sentence to verify.
    """
    answer = "The authors report a human evaluation of translation fluency [C1]."
    cited, weak = verify_citation_support(answer, _label_map(), _ctx_with(0.4))
    assert cited == ["C1"]
    assert weak and weak[0][0] == "C1"


def test_citation_support_disabled_by_zero_threshold():
    answer = "An entirely unrelated claim [C1]."
    cited, weak = verify_citation_support(answer, _label_map(), _ctx_with(0.0))
    assert cited == ["C1"] and weak == []


def test_citation_support_keeps_everything_if_the_embedder_fails():
    """The guard must never be the reason a good answer gets flagged."""
    class _Boom:
        def embed(self, texts, *, is_query=False):
            raise RuntimeError("model gone")

    class _S:
        citation_support_min = 0.4

    class _C:
        settings = _S()
        embedder = _Boom()

    cited, weak = verify_citation_support("A claim [C1].", _label_map(), _C())
    assert cited == ["C1"] and weak == []
