"""Local vector storage with one collection per paper.

Two interchangeable backends behind the same tiny interface:

* ``ChromaStore``  — persistent DuckDB+Parquet/SQLite Chroma client, cosine space.
* ``FlatStore``    — brute-force cosine over a JSON file. A few thousand chunks
  per paper is nothing, so exact search is fast and has zero dependencies. It is
  the automatic fallback when Chroma is not installed, and what the tests use.

Collections are keyed by arXiv id + embedder name, so switching embedding model
does not silently mix vector spaces.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity in [-1, 1]; higher is better
    embedding: list[float] | None = None


def collection_name(arxiv_id: str, embedder_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{arxiv_id}_{embedder_name}").strip("_").lower()
    return f"p_{slug}"[:60]


class VectorStore(Protocol):
    backend: str

    def count(self, collection: str) -> int: ...
    def add(self, collection: str, ids, texts, metadatas, embeddings) -> None: ...
    def query(self, collection: str, embedding, k: int) -> list[Hit]: ...
    def drop(self, collection: str) -> None: ...


# --------------------------------------------------------------------- #
class FlatStore:
    backend = "flat"

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, collection: str) -> Path:
        return self.root / f"{collection}.json"

    def _load(self, collection: str) -> dict[str, Any]:
        path = self._path(collection)
        if not path.exists():
            return {"ids": [], "texts": [], "metadatas": [], "embeddings": []}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("corrupt collection %s; rebuilding", collection)
            return {"ids": [], "texts": [], "metadatas": [], "embeddings": []}

    def count(self, collection: str) -> int:
        return len(self._load(collection)["ids"])

    def add(self, collection, ids, texts, metadatas, embeddings) -> None:
        data = self._load(collection)
        existing = set(data["ids"])
        for i, t, m, e in zip(ids, texts, metadatas, embeddings):
            if i in existing:
                continue
            data["ids"].append(i)
            data["texts"].append(t)
            data["metadatas"].append(m)
            data["embeddings"].append([round(float(x), 6) for x in e])
        self._path(collection).write_text(json.dumps(data), encoding="utf-8")

    def query(self, collection, embedding, k) -> list[Hit]:
        data = self._load(collection)
        if not data["ids"]:
            return []
        norm_q = math.sqrt(sum(x * x for x in embedding)) or 1.0
        scored: list[Hit] = []
        for idx, vec in enumerate(data["embeddings"]):
            norm_v = math.sqrt(sum(x * x for x in vec)) or 1.0
            sim = sum(a * b for a, b in zip(embedding, vec)) / (norm_q * norm_v)
            scored.append(
                Hit(
                    id=data["ids"][idx],
                    text=data["texts"][idx],
                    metadata=data["metadatas"][idx],
                    score=sim,
                    embedding=vec,
                )
            )
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def drop(self, collection: str) -> None:
        self._path(collection).unlink(missing_ok=True)


class ChromaStore:
    backend = "chroma"

    def __init__(self, root: Path) -> None:
        import chromadb  # lazy import

        self._client = chromadb.PersistentClient(path=str(root))

    def _collection(self, name: str):
        return self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def count(self, collection: str) -> int:
        return int(self._collection(collection).count())

    def add(self, collection, ids, texts, metadatas, embeddings) -> None:
        col = self._collection(collection)
        batch = 256
        for i in range(0, len(ids), batch):
            col.upsert(
                ids=list(ids[i : i + batch]),
                documents=list(texts[i : i + batch]),
                metadatas=list(metadatas[i : i + batch]),
                embeddings=[list(map(float, e)) for e in embeddings[i : i + batch]],
            )

    def query(self, collection, embedding, k) -> list[Hit]:
        col = self._collection(collection)
        n = min(k, max(1, col.count()))
        if n == 0:
            return []
        res = col.query(
            query_embeddings=[list(map(float, embedding))],
            n_results=n,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        hits: list[Hit] = []
        embs = (res.get("embeddings") or [[]])[0]
        for idx, doc_id in enumerate(res["ids"][0]):
            distance = res["distances"][0][idx]
            hits.append(
                Hit(
                    id=doc_id,
                    text=res["documents"][0][idx],
                    metadata=res["metadatas"][0][idx] or {},
                    score=1.0 - float(distance),  # cosine space
                    embedding=list(embs[idx]) if idx < len(embs) else None,
                )
            )
        return hits

    def drop(self, collection: str) -> None:
        try:
            self._client.delete_collection(collection)
        except Exception:  # noqa: BLE001 - deleting a missing collection is fine
            pass


def build_store(settings) -> VectorStore:
    root = settings.vectorstore_dir
    root.mkdir(parents=True, exist_ok=True)
    backend = settings.vector_backend
    if backend in {"auto", "chroma"}:
        try:
            return ChromaStore(root / "chroma")
        except Exception as exc:  # noqa: BLE001
            if backend != "auto":
                raise
            log.warning("chromadb unavailable (%s); using the built-in flat store", exc)
    return FlatStore(root / "flat")
