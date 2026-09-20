"""Thin client over the official arXiv Atom API (no scraping).

Two things live here that matter for behaviour:

1. **Politeness.** arXiv asks for >= 3 seconds between calls from one client.
   The client enforces that itself, so the relaxation ladder below cannot
   accidentally hammer the API.
2. **The relaxation ladder.** A vague topic often returns zero hits when
   rendered as a strict conjunction of quoted phrases. Rather than giving up,
   `render_query` emits progressively looser queries and the retrieval node
   walks down the ladder until it has enough candidates, recording every
   attempt in the state so the user can see what was actually asked.
"""

from __future__ import annotations

import logging
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import requests

from ..state import PaperMeta, QueryPlan

log = logging.getLogger(__name__)

API_URL = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "arxiv-digest-agent/1.0 (assessment project; contact: local user)"

# 2401.12345, 2401.12345v2, or the legacy cs/0112017 form.
ID_RE = re.compile(r"\b(\d{4}\.\d{4,5})(v\d+)?\b|\b([a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?\b")
URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([^\s?#]+)", re.IGNORECASE)

MAX_RELAXATION_LEVEL = 4


class ArxivError(RuntimeError):
    pass


def extract_arxiv_id(text: str) -> str | None:
    """Pull an arXiv id out of a URL or bare string. Returns id without version."""
    url_match = URL_RE.search(text or "")
    candidate = url_match.group(1) if url_match else (text or "")
    candidate = candidate.removesuffix(".pdf")
    match = ID_RE.search(candidate)
    if not match:
        return None
    return match.group(1) or match.group(3)


def render_query(plan: QueryPlan, level: int = 0) -> str:
    """Build an arXiv `search_query` string at a given relaxation level.

    level 0: all phrases AND'ed, category filtered, optional date floor
    level 1: same, no date floor
    level 2: phrases OR'ed, category filtered
    level 3: phrases + terms OR'ed, no category filter
    level 4: two strongest terms OR'ed (last-ditch broad sweep)
    """
    phrases = [p for p in plan.phrases if p.strip()]
    terms = [t for t in plan.terms if t.strip()]
    cats = [c for c in plan.categories if c.strip()]

    def phrase_clause(p: str) -> str:
        return f'all:"{p}"'

    if level <= 1:
        parts = [phrase_clause(p) for p in phrases] or [f"all:{t}" for t in terms[:3]]
        core = " AND ".join(parts)
    elif level == 2:
        parts = [phrase_clause(p) for p in phrases] or [f"all:{t}" for t in terms[:4]]
        core = " OR ".join(parts)
    elif level == 3:
        pool = [phrase_clause(p) for p in phrases] + [f"all:{t}" for t in terms[:5]]
        core = " OR ".join(pool)
        cats = []
    else:
        pool = [phrase_clause(p) for p in phrases[:1]] + [f"all:{t}" for t in terms[:2]]
        core = " OR ".join(pool) or "all:machine learning"
        cats = []

    if not core:
        core = "all:machine learning"
    query = f"({core})"
    if cats:
        query += " AND (" + " OR ".join(f"cat:{c}" for c in cats) + ")"
    if plan.authors:
        query += " AND (" + " OR ".join(f'au:"{a}"' for a in plan.authors) + ")"
    if level == 0 and plan.since_year:
        query += f" AND submittedDate:[{plan.since_year}01010000 TO 299912312359]"
    return query


@dataclass
class _RateLimiter:
    min_interval: float
    _last: float = 0.0

    def wait(self) -> None:
        delta = time.monotonic() - self._last
        if self._last and delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


class ArxivClient:
    def __init__(self, settings) -> None:
        self.s = settings
        self._limiter = _RateLimiter(settings.arxiv_min_interval)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    # -- HTTP ---------------------------------------------------------- #
    def _get(self, params: dict) -> str:
        last: Exception | None = None
        for attempt in range(1, self.s.arxiv_retries + 1):
            self._limiter.wait()
            try:
                resp = self._session.get(API_URL, params=params, timeout=self.s.arxiv_timeout)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise ArxivError(f"arXiv HTTP {resp.status_code}")
                resp.raise_for_status()
                return resp.text
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("arXiv request failed (%s), attempt %d", exc, attempt)
                time.sleep(2**attempt)
        raise ArxivError(f"arXiv API unreachable after {self.s.arxiv_retries} attempts: {last}")

    # -- public -------------------------------------------------------- #
    def search(self, query: str, max_results: int = 20, sort: str = "relevance") -> list[PaperMeta]:
        xml = self._get(
            {
                "search_query": query,
                "start": 0,
                "max_results": max_results,
                "sortBy": {"relevance": "relevance", "date": "submittedDate"}[sort],
                "sortOrder": "descending",
            }
        )
        return self._parse_feed(xml)

    def get_by_id(self, arxiv_id: str) -> PaperMeta | None:
        papers = self._parse_feed(self._get({"id_list": arxiv_id, "max_results": 1}))
        return papers[0] if papers else None

    def download_pdf(self, paper: PaperMeta, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", paper.arxiv_id or "paper")
        target = dest_dir / f"{safe}.pdf"
        if target.exists() and target.stat().st_size > 10_000:
            log.info("using cached PDF %s", target)
            return target

        url = paper.pdf_url or f"https://arxiv.org/pdf/{paper.arxiv_id}"
        self._limiter.wait()
        with self._session.get(url, timeout=120, stream=True, allow_redirects=True) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            if "pdf" not in ctype.lower() and "octet-stream" not in ctype.lower():
                raise ArxivError(f"expected a PDF from {url}, got Content-Type={ctype!r}")
            limit = self.s.max_pdf_mb * 1024 * 1024
            written = 0
            with open(target, "wb") as fh:
                for block in resp.iter_content(chunk_size=1 << 16):
                    written += len(block)
                    if written > limit:
                        fh.close()
                        target.unlink(missing_ok=True)
                        raise ArxivError(f"PDF exceeds {self.s.max_pdf_mb} MB cap")
                    fh.write(block)
        if target.stat().st_size < 1000:
            target.unlink(missing_ok=True)
            raise ArxivError("downloaded PDF is suspiciously small")
        return target

    # -- parsing ------------------------------------------------------- #
    @staticmethod
    def _parse_feed(xml_text: str) -> list[PaperMeta]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise ArxivError(f"malformed Atom feed from arXiv: {exc}") from exc

        papers: list[PaperMeta] = []
        for entry in root.findall("atom:entry", NS):
            raw_id = (entry.findtext("atom:id", "", NS) or "").strip()
            if not raw_id:
                continue
            tail = raw_id.rsplit("/abs/", 1)[-1]
            version = ""
            if "v" in tail.split("/")[-1]:
                base, _, ver = tail.rpartition("v")
                if ver.isdigit():
                    tail, version = base, f"v{ver}"
            # An error feed has a single entry with no title/summary of substance.
            title = " ".join((entry.findtext("atom:title", "", NS) or "").split())
            if title.lower().startswith("error"):
                continue
            pdf_url = ""
            for link in entry.findall("atom:link", NS):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href", "")
            papers.append(
                PaperMeta(
                    arxiv_id=tail,
                    version=version,
                    title=title,
                    authors=[
                        (a.findtext("atom:name", "", NS) or "").strip()
                        for a in entry.findall("atom:author", NS)
                    ],
                    abstract=" ".join((entry.findtext("atom:summary", "", NS) or "").split()),
                    categories=[
                        c.get("term", "") for c in entry.findall("atom:category", NS) if c.get("term")
                    ],
                    primary_category=(
                        entry.find("arxiv:primary_category", NS).get("term", "")
                        if entry.find("arxiv:primary_category", NS) is not None
                        else ""
                    ),
                    published=entry.findtext("atom:published", "", NS) or "",
                    updated=entry.findtext("atom:updated", "", NS) or "",
                    abs_url=f"https://arxiv.org/abs/{tail}",
                    pdf_url=pdf_url or f"https://arxiv.org/pdf/{tail}",
                    comment=(entry.findtext("arxiv:comment", "", NS) or "").strip(),
                    doi=(entry.findtext("arxiv:doi", "", NS) or "").strip(),
                    journal_ref=(entry.findtext("arxiv:journal_ref", "", NS) or "").strip(),
                )
            )
        return papers


def quote(value: str) -> str:  # tiny helper used in logs/tests
    return urllib.parse.quote(value, safe="")
