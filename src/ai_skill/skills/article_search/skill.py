"""Academic article search skill.

Searches arXiv and Semantic Scholar for peer-reviewed papers.
Results are normalised into a unified schema regardless of source.

Parameters accepted in SkillInput.parameters:
    query (str): The search query. Required.
    sources (list[str]): Sources to query. Default: ["arxiv", "semantic_scholar"].
    max_results (int): Maximum results per source. Default: 10.
    date_from (str): ISO date string for earliest publication (e.g. "2020-01-01").
    seek_contradictions (bool): If True, append "critique" / "limitations" to
        the query to find contradicting literature. Default: False.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from ai_skill.core.url_safety import UrlSafetyGuard
from ai_skill.skills.base import BaseSkill, SkillInput, SkillMeta, SkillOutput

logger = logging.getLogger(__name__)

_SEMANTIC_SCHOLAR_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search"
_SEMANTIC_SCHOLAR_FIELDS = "title,authors,abstract,year,externalIds,url,venue,citationCount"
_DEFAULT_MAX_RESULTS = 10

# ---------------------------------------------------------------------------
# Semantic Scholar rate limiter — 1 request / second across all endpoints
# ---------------------------------------------------------------------------
_ss_lock = threading.Lock()
_ss_last_call: float = 0.0  # epoch timestamp of the most recent API call
_SS_MIN_INTERVAL = 1.0  # seconds


def _ss_rate_limited_get(url: str, **kwargs: Any) -> httpx.Response:
    """Perform an HTTP GET to a Semantic Scholar endpoint with 1 req/s throttling
    and automatic retry on 429 (rate limit exceeded).

    Args:
        url: The full endpoint URL.
        **kwargs: Passed directly to ``httpx.get``.

    Returns:
        The ``httpx.Response`` object.
    """
    global _ss_last_call  # noqa: PLW0603
    max_retries = 4
    for attempt in range(max_retries):
        with _ss_lock:
            elapsed = time.monotonic() - _ss_last_call
            if elapsed < _SS_MIN_INTERVAL:
                time.sleep(_SS_MIN_INTERVAL - elapsed)
            _ss_last_call = time.monotonic()

        response = httpx.get(url, **kwargs)
        if response.status_code != 429:
            return response

        retry_after = int(response.headers.get("Retry-After", 5 * (2 ** attempt)))
        logger.warning("Semantic Scholar 429 — aguardando %ds antes de tentar novamente.", retry_after)
        time.sleep(retry_after)

    return response  # return last response after exhausting retries


class ArticleSearchSkill(BaseSkill):
    """Search academic databases for peer-reviewed papers.

    Queries arXiv and Semantic Scholar in parallel, normalises results,
    and deduplicates by DOI or arXiv ID where possible.

    All source URLs are checked via Google Safe Browsing v4 before inclusion.
    """

    SKILL_META = SkillMeta(
        name="article_search",
        description=(
            "Search arXiv and Semantic Scholar for peer-reviewed academic papers "
            "matching the research query. Returns normalised metadata including "
            "title, authors, abstract, year, and DOI."
        ),
        version="1.0.0",
        author="ai_skill",
        license="MIT",
        tags=["search", "academic", "arxiv", "semantic-scholar", "papers", "literature"],
        dependencies=[],
    )

    def __init__(self) -> None:
        """Initialise with URL safety guard."""
        self._safety_guard = UrlSafetyGuard()

    def run(self, input: SkillInput) -> SkillOutput:
        """Search academic databases for papers.

        Args:
            input: SkillInput with parameters: query (required), sources,
                max_results, date_from, seek_contradictions.

        Returns:
            SkillOutput with result containing:
                papers (list[dict]): Normalised paper metadata.
                sources_queried (list[str]): Which databases were queried.
                total_found (int): Total papers returned across all sources.
        """
        params = input.parameters
        query: str | None = params.get("query")
        if not query:
            return self._error_output("Parameter 'query' is required.")

        sources: list[str] = params.get("sources", ["arxiv", "semantic_scholar"])
        max_results: int = int(params.get("max_results", _DEFAULT_MAX_RESULTS))
        date_from: str | None = params.get("date_from")
        seek_contradictions: bool = bool(params.get("seek_contradictions", False))

        if seek_contradictions:
            query = f"{query} limitations critique challenges"

        all_papers: list[dict[str, Any]] = []
        sources_queried: list[str] = []
        errors: list[str] = []

        if "arxiv" in sources:
            try:
                papers = self._search_arxiv(query, max_results, date_from)
                all_papers.extend(papers)
                sources_queried.append("arxiv")
            except Exception as exc:
                errors.append(f"arXiv: {exc}")
                logger.warning("arXiv search failed: %s", exc)

        if "semantic_scholar" in sources:
            try:
                papers = self._search_semantic_scholar(query, max_results)
                all_papers.extend(papers)
                sources_queried.append("semantic_scholar")
            except Exception as exc:
                errors.append(f"Semantic Scholar: {exc}")
                logger.warning("Semantic Scholar search failed: %s", exc)

        if not all_papers and errors:
            return self._error_output(
                f"All academic search sources failed: {'; '.join(errors)}"
            )

        # Deduplicate by DOI
        seen_dois: set[str] = set()
        unique_papers: list[dict[str, Any]] = []
        for paper in all_papers:
            doi = paper.get("doi")
            if doi and doi in seen_dois:
                continue
            if doi:
                seen_dois.add(doi)
            unique_papers.append(paper)

        # Safety filter on URLs
        urls = [p["url"] for p in unique_papers if p.get("url")]
        safe_urls = set(self._safety_guard.filter(urls))
        safe_papers = [
            p for p in unique_papers
            if not p.get("url") or p["url"] in safe_urls
        ]

        error_msg: str | None = None
        if errors:
            error_msg = f"Partial results. Errors: {'; '.join(errors)}"

        return SkillOutput(
            skill_name="article_search",
            result={
                "papers": safe_papers,
                "sources_queried": sources_queried,
                "total_found": len(safe_papers),
                "query": query,
            },
            confidence=0.9 if safe_papers else 0.0,
            sources=[p["url"] for p in safe_papers if p.get("url")],
            error=error_msg,
            cached=False,
        )

    def _search_arxiv(
        self,
        query: str,
        max_results: int,
        date_from: str | None,
    ) -> list[dict[str, Any]]:
        """Search arXiv using the arxiv Python library.

        Args:
            query: Search query string.
            max_results: Maximum results to retrieve.
            date_from: ISO date string for earliest publication.

        Returns:
            List of normalised paper dicts.
        """
        try:
            import arxiv
        except ImportError as exc:
            raise ImportError(
                "arxiv package is required. Install with: uv add arxiv"
            ) from exc

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: list[dict[str, Any]] = []
        for result in client.results(search):
            pub_date = result.published.date() if result.published else None
            if date_from and pub_date:
                from datetime import date
                cutoff = date.fromisoformat(date_from)
                if pub_date < cutoff:
                    continue

            arxiv_id = result.entry_id.split("/")[-1] if result.entry_id else None
            papers.append({
                "title": result.title or "",
                "authors": [a.name for a in result.authors],
                "abstract": result.summary or "",
                "year": result.published.year if result.published else None,
                "url": result.entry_id or "",
                "doi": result.doi,
                "arxiv_id": arxiv_id,
                "venue": "arXiv",
                "citation_count": None,
                "source": "arxiv",
            })

        return papers

    def _search_semantic_scholar(
        self,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Query the Semantic Scholar Graph API.

        Args:
            query: Search query string.
            max_results: Maximum results to retrieve.

        Returns:
            List of normalised paper dicts.
        """
        import os

        headers: dict[str, str] = {}
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key

        response = _ss_rate_limited_get(
            _SEMANTIC_SCHOLAR_ENDPOINT,
            params={
                "query": query,
                "limit": min(max_results, 100),
                "fields": _SEMANTIC_SCHOLAR_FIELDS,
            },
            headers=headers,
            timeout=20.0,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        papers: list[dict[str, Any]] = []
        for item in data.get("data", []):
            doi_info: dict[str, Any] = item.get("externalIds", {})
            doi: str | None = doi_info.get("DOI") or doi_info.get("doi")
            arxiv_id: str | None = doi_info.get("ArXiv")
            paper_id: str = item.get("paperId", "")
            url = item.get("url") or (
                f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""
            )

            papers.append({
                "title": item.get("title", ""),
                "authors": [
                    a.get("name", "") for a in item.get("authors", [])
                ],
                "abstract": item.get("abstract", ""),
                "year": item.get("year"),
                "url": url,
                "doi": doi,
                "arxiv_id": arxiv_id,
                "venue": item.get("venue", ""),
                "citation_count": item.get("citationCount"),
                "source": "semantic_scholar",
            })

        return papers


SKILL_CLASS = ArticleSearchSkill
