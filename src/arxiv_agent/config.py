"""Central configuration.

Everything is environment-driven with defaults that work with zero API keys
(Ollama on localhost + a local sentence-transformers embedder). See .env.example.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "true" if default else "false").lower() in _TRUTHY


# Per-provider default model. Kept here so `--provider groq` alone is enough.
DEFAULT_MODELS = {
    "ollama": "qwen2.5:7b-instruct",
    # Groq retires models without much notice; check GET /openai/v1/models
    # if this starts 404ing.
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-2.0-flash",
    "openrouter": "meta-llama/llama-3.3-70b-instruct:free",
    "mock": "mock",
}

DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "openrouter": "https://openrouter.ai/api/v1",
    "mock": "",
}

API_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


@dataclass
class Settings:
    # --- LLM ---
    provider: str = "ollama"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048
    llm_timeout: int = 180
    llm_retries: int = 3

    # --- Embeddings / vector store ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_backend: str = "auto"  # auto | sentence-transformers | hashing
    vector_backend: str = "auto"  # auto | chroma | numpy

    # --- Chunking / retrieval ---
    chunk_chars: int = 1200
    chunk_overlap: int = 180
    retrieval_k: int = 8
    rerank_k: int = 5
    min_similarity: float = 0.15
    # How much a cited chunk has to resemble the claim citing it before we stop
    # trusting the citation. Raise it to catch more invented citations at the
    # cost of nagging about good ones; 0 turns the check off.
    citation_support_min: float = 0.40
    mmr_lambda: float = 0.65

    # --- arXiv ---
    arxiv_max_results: int = 20
    arxiv_min_interval: float = 3.0  # arXiv asks for >=3s between API calls
    arxiv_timeout: int = 30
    arxiv_retries: int = 3

    # --- PDF ---
    max_pdf_pages: int = 60
    max_pdf_mb: int = 40
    min_chars_per_page: int = 180  # below this we treat the page as image-only

    # --- Summarisation ---
    map_reduce_threshold: int = 24_000  # chars of body text
    max_sections_mapped: int = 12

    # --- Paths ---
    workdir: Path = field(default_factory=lambda: Path(_env("ARXIV_AGENT_HOME", "~/.arxiv_agent")).expanduser())

    interactive: bool = True

    # ------------------------------------------------------------------ #
    @property
    def sessions_dir(self) -> Path:
        return self.workdir / "sessions"

    @property
    def vectorstore_dir(self) -> Path:
        return self.workdir / "vectorstore"

    @property
    def pdf_cache_dir(self) -> Path:
        return self.workdir / "pdf_cache"

    def ensure_dirs(self) -> None:
        for path in (self.sessions_dir, self.vectorstore_dir, self.pdf_cache_dir):
            path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        provider = _env("LLM_PROVIDER", "ollama").lower()
        model = _env("LLM_MODEL", DEFAULT_MODELS.get(provider, ""))
        base_url = _env("LLM_BASE_URL", DEFAULT_BASE_URLS.get(provider, ""))
        api_key = os.getenv(API_KEY_ENV.get(provider, "__none__"), "") or ""
        return cls(
            provider=provider,
            model=model,
            base_url=base_url.rstrip("/"),
            api_key=api_key.strip(),
            temperature=_env_float("LLM_TEMPERATURE", 0.1),
            max_tokens=_env_int("LLM_MAX_TOKENS", 2048),
            llm_timeout=_env_int("LLM_TIMEOUT", 180),
            llm_retries=_env_int("LLM_RETRIES", 3),
            embedding_model=_env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            embedding_backend=_env("EMBEDDING_BACKEND", "auto").lower(),
            vector_backend=_env("VECTOR_BACKEND", "auto").lower(),
            chunk_chars=_env_int("CHUNK_CHARS", 1200),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 180),
            retrieval_k=_env_int("RETRIEVAL_K", 8),
            rerank_k=_env_int("RERANK_K", 5),
            min_similarity=_env_float("MIN_SIMILARITY", 0.15),
            citation_support_min=_env_float("CITATION_SUPPORT_MIN", 0.40),
            arxiv_max_results=_env_int("ARXIV_MAX_RESULTS", 20),
            max_pdf_pages=_env_int("MAX_PDF_PAGES", 60),
            interactive=_env_bool("INTERACTIVE", True),
        )

    def redacted(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key_set": bool(self.api_key),
            "embedding_model": self.embedding_model,
            "vector_backend": self.vector_backend,
            "workdir": str(self.workdir),
        }
