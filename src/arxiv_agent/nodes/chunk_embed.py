"""Node 5 — chunk & embed.

Idempotent by construction: the collection name is derived from the arXiv id
*and* the embedder name, and a populated collection is reused rather than
re-embedded. Re-briefing the same paper is therefore nearly free, and switching
embedding models can never mix two vector spaces in one collection.
"""

from __future__ import annotations

import logging

from ..graph.engine import Context
from ..services.chunker import chunk_document, chunks_from_abstract
from ..services.vectorstore import collection_name
from ..state import AgentState
from .fetch_parse import load_parsed

log = logging.getLogger(__name__)


def chunk_and_embed(state: AgentState, ctx: Context) -> None:
    paper = state.paper_meta
    doc = load_parsed(state)

    if doc.degraded or not doc.sections:
        chunks = chunks_from_abstract(paper.arxiv_id, paper.title, paper.abstract)
        state.note("chunk_and_embed", "degraded mode: indexing the abstract only")
    else:
        chunks = chunk_document(
            doc,
            arxiv_id=paper.arxiv_id,
            chunk_chars=ctx.settings.chunk_chars,
            overlap=ctx.settings.chunk_overlap,
        )

    if not chunks:
        chunks = chunks_from_abstract(paper.arxiv_id, paper.title, paper.abstract or paper.title)

    collection = collection_name(paper.arxiv_id, ctx.embedder.name)
    state.collection = collection
    state.embedder = ctx.embedder.name
    state.vector_backend = ctx.store.backend

    existing = ctx.store.count(collection)
    if existing >= len(chunks) > 0:
        state.chunk_count = existing
        state.note("chunk_and_embed", f"reusing {existing} already-embedded chunks in {collection}")
        return

    if existing:  # partial/stale collection — rebuild rather than merge
        ctx.store.drop(collection)

    texts = [c.text for c in chunks]
    vectors = ctx.embedder.embed(texts)
    ctx.store.add(
        collection,
        ids=[c.id for c in chunks],
        texts=texts,
        metadatas=[c.metadata for c in chunks],
        embeddings=vectors,
    )
    state.chunk_count = len(chunks)
    avg = sum(len(t) for t in texts) // max(1, len(texts))
    state.note(
        "chunk_and_embed",
        f"embedded {len(chunks)} chunks (avg {avg} chars) into {collection} "
        f"via {ctx.embedder.name} on {ctx.store.backend}",
    )
