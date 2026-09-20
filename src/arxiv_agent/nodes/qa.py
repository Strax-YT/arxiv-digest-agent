"""Node 7 — grounded QA.

Five layers keep answers tied to the paper, from cheapest to most expensive:

1. **Retrieval gate.** If the best-matching chunk scores below
   `min_similarity`, we do not call the LLM at all — we answer "not found in
   the retrieved sections" and name the sections we do have. No call, no
   hallucination.
2. **Context-only prompt.** The model sees only the retrieved excerpts and is
   told explicitly that "the paper does not say" is a correct answer. It is also
   told not to use outside knowledge about a paper it may have memorised.
3. **Citation contract.** Every claim must carry a `[C2]`-style label. After
   generation we verify the cited labels exist in what we actually retrieved.
   No valid citation → one strict retry → flag the answer as ungrounded.
4. **Diversified context.** MMR over the top-k so the model sees several parts
   of the paper rather than five near-duplicate paragraphs from one page.
5. **Citation support.** A recognised label only proves the model wrote a label
   we know — one that has memorised the paper can answer from memory and attach
   a plausible marker. Each citing sentence is checked against the chunk it
   points at. Supported and unrelated text overlap too much to settle
   groundedness outright, so a weak match adds a caveat rather than a verdict.

Conversation history lives in the state and is persisted, so a follow-up like
"and how does that compare to the baseline?" is condensed into a standalone
question before retrieval — vector search cannot resolve "that" on its own.
"""

from __future__ import annotations

import logging
import re
import time

from ..graph.engine import Context
from ..prompts import CONDENSE_SYSTEM, CONDENSE_USER, QA_STRICT_RETRY, QA_SYSTEM, QA_USER
from ..services.embeddings import cosine
from ..services.llm import LLMError
from ..state import AgentState

log = logging.getLogger(__name__)

CITATION_RE = re.compile(r"\[(C\d{1,2})\]")

# Some models reach for fullwidth or CJK brackets instead of ASCII ones. A
# 【C3】 means the same thing as [C3] and should not cost us a retry.
_CITATION_ANY = re.compile(r"[\[【［]\s*(C\d{1,2})\s*[\]】］]")


def normalize_citations(text: str) -> str:
    """Rewrite bracket variants of a citation marker to the canonical [Cn]."""
    return _CITATION_ANY.sub(lambda m: f"[{m.group(1)}]", text)


ANAPHORA = re.compile(
    r"\b(it|its|they|them|their|that|this|those|these|the approach|the method|the model|there)\b",
    re.IGNORECASE,
)
NOT_FOUND = "Not found in the retrieved sections of this paper."


def _needs_condensing(question: str, history: list[dict]) -> bool:
    if not history:
        return False
    return bool(ANAPHORA.search(question)) or len(question.split()) <= 6


def condense_question(state: AgentState, ctx: Context, question: str) -> str:
    if ctx.llm is None or not _needs_condensing(question, state.qa_history):
        return question
    history = "\n".join(
        f"Q: {turn['question']}\nA: {turn['answer'][:280]}" for turn in state.qa_history[-3:]
    )
    try:
        rewritten = ctx.llm.complete(
            CONDENSE_USER.format(history=history, question=question), CONDENSE_SYSTEM, temperature=0.0
        ).strip().strip('"')
        # Guard against a model that "helpfully" answers instead of rewriting.
        if 3 <= len(rewritten.split()) <= 60:
            return rewritten
    except LLMError as exc:
        log.warning("condense failed (%s); using the raw question", exc)
    return question


def mmr(hits: list, query_vec: list[float], k: int, lambda_: float) -> list:
    """Maximal marginal relevance: relevance minus redundancy."""
    selected: list = []
    pool = list(hits)
    while pool and len(selected) < k:
        best, best_score = None, -1e9
        for hit in pool:
            relevance = hit.score
            redundancy = 0.0
            if selected and hit.embedding:
                redundancy = max(
                    cosine(hit.embedding, s.embedding) for s in selected if s.embedding
                ) if any(s.embedding for s in selected) else 0.0
            score = lambda_ * relevance - (1 - lambda_) * redundancy
            # Bibliography chunks are rarely the answer; nudge them down.
            if hit.metadata.get("low_value"):
                score -= 0.1
            if score > best_score:
                best, best_score = hit, score
        selected.append(best)
        pool.remove(best)
    return selected


def retrieve(state: AgentState, ctx: Context, question: str):
    query_vec = ctx.embedder.embed([question], is_query=True)[0]
    hits = ctx.store.query(state.collection, query_vec, ctx.settings.retrieval_k)
    if not hits:
        return [], query_vec
    return mmr(hits, query_vec, ctx.settings.rerank_k, ctx.settings.mmr_lambda), query_vec


def format_context(hits) -> tuple[str, dict[str, object]]:
    blocks: list[str] = []
    label_map: dict[str, object] = {}
    for i, hit in enumerate(hits, start=1):
        label = f"C{i}"
        label_map[label] = hit
        section = hit.metadata.get("section", "?")
        pages = f"p.{hit.metadata.get('page_start', '?')}-{hit.metadata.get('page_end', '?')}"
        body = hit.text.split("\n", 1)[-1] if hit.text.startswith("[Section:") else hit.text
        blocks.append(f"[{label}] {section} ({pages}, similarity {hit.score:.2f})\n{body}")
    return "\n\n".join(blocks), label_map


# The lexical fallback scores far lower than dense cosine, so it needs its own
# bar or every citation looks unsupported.
LEXICAL_SUPPORT_MIN = 0.15

# A sentence ends at ., ! or ? followed by whitespace. Crude, but citations sit
# at clause level and we only need the neighbourhood of the marker.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def verify_citation_support(answer, label_map, ctx) -> tuple[list[str], list[tuple[str, float]]]:
    """Flag citations whose chunk does not resemble the sentence citing it.

    A recognised label only proves the model wrote a label we know. One that has
    memorised the paper can answer from its own weights and pin a plausible
    marker to an unrelated chunk.

    This annotates rather than gates. The two populations overlap enough that a
    hard cutoff throws out roughly a fifth of good citations, and a warning that
    fires on correct answers is one nobody reads. Weak support is a caveat the
    reader can act on.

    Returns (cited, weak): every valid label, and those whose best citing
    sentence fell short, as (label, score).
    """
    cited = sorted({c for c in CITATION_RE.findall(answer) if c in label_map})
    threshold = getattr(ctx.settings, "citation_support_min", 0.0)
    embedder = getattr(ctx, "embedder", None)
    if threshold and getattr(embedder, "degraded", False):
        # Same story as MIN_SIMILARITY in agent.py.
        threshold = min(threshold, LEXICAL_SUPPORT_MIN)
    if not cited or not threshold or embedder is None:
        return cited, []

    # Score each label by the best sentence that cites it: one good use is
    # enough, and a trailing "Sources:" list should not sink a real citation.
    best: dict[str, float] = {}
    for sentence in _SENTENCE_SPLIT.split(answer):
        labels = [c for c in CITATION_RE.findall(sentence) if c in label_map]
        if not labels:
            continue
        claim = CITATION_RE.sub("", sentence).strip()
        if len(claim) < 25:  # too short to judge; give it the benefit of the doubt
            for label in labels:
                best[label] = max(best.get(label, 0.0), 1.0)
            continue
        try:
            claim_vec = embedder.embed([claim], is_query=True)[0]
            for label in labels:
                hit = label_map[label]
                chunk_vec = getattr(hit, "embedding", None)
                if chunk_vec is None:
                    chunk_vec = embedder.embed([hit.text])[0]
                best[label] = max(best.get(label, 0.0), float(cosine(claim_vec, chunk_vec)))
        except Exception:  # noqa: BLE001 - never fail the answer over the guard
            log.warning("citation support check failed; keeping citations", exc_info=True)
            return cited, []

    weak = sorted((label, round(float(sc), 3)) for label, sc in best.items() if sc < threshold)
    if weak:
        log.info("weakly supported citations %s (threshold %.2f)", weak, threshold)
    return cited, weak


def answer_question(state: AgentState, ctx: Context) -> None:
    question = (state.pending_question or "").strip()
    if not question:
        return
    paper = state.paper_meta
    started = time.time()
    standalone = condense_question(state, ctx, question)
    hits, _ = retrieve(state, ctx, standalone)

    top_score = hits[0].score if hits else 0.0
    if not hits or top_score < ctx.settings.min_similarity:
        sections = sorted({h.metadata.get("section", "?") for h in hits})[:4]
        hint = f" The closest sections indexed are: {', '.join(sections)}." if sections else ""
        state.qa_history.append(
            {
                "question": question,
                "standalone": standalone,
                "answer": f"{NOT_FOUND} Nothing in the indexed text matched closely enough"
                f" (best similarity {top_score:.2f}).{hint}",
                "citations": [],
                "chunks": [h.id for h in hits],
                "grounded": True,  # abstaining *is* the grounded behaviour
                "abstained": True,
                "top_score": round(top_score, 3),
                "seconds": round(time.time() - started, 2),
                "ts": time.time(),
            }
        )
        state.pending_question = ""
        return

    context, label_map = format_context(hits)
    prompt = QA_USER.format(
        title=paper.title, arxiv_id=paper.arxiv_id, context=context, question=standalone
    )
    answer = normalize_citations(ctx.llm.complete(prompt, QA_SYSTEM, temperature=0.0).strip())

    cited, weak = verify_citation_support(answer, label_map, ctx)
    abstained = answer.startswith(NOT_FOUND[:30])

    if not cited and not abstained:
        log.info("answer had no valid citation; retrying once with a strict prompt")
        answer = normalize_citations(
            ctx.llm.complete(
                QA_STRICT_RETRY.format(
                    original=answer[:1500], context=context, question=standalone
                ),
                QA_SYSTEM,
                temperature=0.0,
            ).strip()
        )
        cited, weak = verify_citation_support(answer, label_map, ctx)
        abstained = answer.startswith(NOT_FOUND[:30])

    grounded = bool(cited) or abstained
    if not grounded:
        answer = (
            "⚠️ The model did not cite the retrieved excerpts, so this answer is NOT verified "
            "against the paper — treat it as unreliable.\n\n" + answer
        )
    elif weak:
        # Cited, but the cited text does not look much like the claim. Name the
        # labels so the reader knows exactly which sentences to check.
        labels = ", ".join(f"[{label}]" for label, _ in weak)
        answer += (
            f"\n\n_Note: the excerpt(s) behind {labels} are only weakly similar to the "
            "claim citing them. Worth checking those directly against the paper._"
        )

    state.qa_history.append(
        {
            "question": question,
            "standalone": standalone if standalone != question else "",
            "answer": answer,
            "citations": [
                {
                    "label": label,
                    "chunk_id": label_map[label].id,
                    "section": label_map[label].metadata.get("section", ""),
                    "pages": [
                        label_map[label].metadata.get("page_start"),
                        label_map[label].metadata.get("page_end"),
                    ],
                    "score": round(label_map[label].score, 3),
                }
                for label in dict.fromkeys(cited)
            ],
            "chunks": [h.id for h in hits],
            "grounded": grounded,
            "abstained": abstained,
            "top_score": round(top_score, 3),
            "seconds": round(time.time() - started, 2),
            "ts": time.time(),
        }
    )
    state.pending_question = ""
