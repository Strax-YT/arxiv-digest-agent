"""Embeddings.

Primary: sentence-transformers `all-MiniLM-L6-v2` — small, CPU-friendly, no key.
Fallback: a deterministic hashed character-n-gram embedder. It is genuinely
worse at semantics, but it means the pipeline (and the test suite) still runs on
a machine with no model download, and the degradation is *reported* in the
briefing rather than silently pretended away.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Iterable, Protocol

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily: heavy

        self._model = SentenceTransformer(model_name)
        self.name = f"sentence-transformers:{model_name}"
        # Renamed in sentence-transformers 6.x.
        _dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dim = int(_dim())
        self.degraded = False

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return [list(map(float, v)) for v in vectors]


class HashingEmbedder:
    """Normalised bag of word-unigrams + character 4-grams, hashed into `dim` buckets.

    Lexical-only, so it handles "what batch size did they use" fine and
    "why is this better than prior work" poorly. Good enough as a safety net.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.name = f"hashing:{dim}"
        self.degraded = True

    @staticmethod
    def _features(text: str) -> Iterable[str]:
        lowered = text.lower()
        words = re.findall(r"[a-z0-9]+", lowered)
        yield from words
        for a, b in zip(words, words[1:]):
            yield f"{a}_{b}"
        squashed = re.sub(r"\s+", " ", lowered)
        for i in range(0, max(0, len(squashed) - 4), 2):
            yield squashed[i : i + 4]

    def embed(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for feat in self._features(text or ""):
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "big") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def build_embedder(settings) -> Embedder:
    backend = settings.embedding_backend
    if backend in {"auto", "sentence-transformers"}:
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception as exc:  # noqa: BLE001 - any import/download failure
            if backend != "auto":
                raise
            log.warning(
                "sentence-transformers unavailable (%s); falling back to the hashing embedder. "
                "Retrieval quality will be lexical only.",
                exc,
            )
    return HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are L2-normalised
