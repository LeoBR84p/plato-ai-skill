"""Exa semantic search skill implementation.

Uses the Exa neural search API for high-quality semantic retrieval of web pages
and academic papers. Complements Brave/DuckDuckGo (keyword-based) with
embedding-based similarity search, which performs significantly better for
conceptual and cross-lingual academic queries.

Primary use in CP2:
  - Third fallback in web_search (Brave → DuckDuckGo → Exa)
  - Direct skill call by the planner for deep semantic paper/content search
  - Content retrieval: given a URL, extract clean text via Exa's get_contents

Parameters accepted in SkillInput.parameters:
    query (str): The search query. Required unless 'url' is provided.
    url (str): Fetch contents of a specific URL (skips search). Optional.
    num_results (int): Maximum results to return. Default: 10.
    search_type (str): Exa search strategy — "auto" (balanced, default),
        "fast" (real-time, low latency), "deep" (multi-query synthesis,
        best for research enrichment, 4-12 s), or "deep-reasoning"
        (multi-step reasoning, slowest). Default: "auto".
    category (str): Filter by content category — "research paper", "tweet",
        "news", "company", "pdf", etc. Optional.
    include_text (bool): Fetch full page text alongside results. Default: True.
    max_characters (int): Maximum characters per result text. Default: 2000.
    start_published_date (str): ISO date filter, e.g. "2022-01-01". Optional.
    end_published_date (str): ISO date filter, e.g. "2024-12-31". Optional.
    max_age_hours (int): Cache freshness threshold. 0 = always livecrawl,
        -1 = cache only, omit = default (livecrawl if no cache). Optional.

Environment variables:
    EXA_API_KEY    API key for the Exa neural search API.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ai_skill.skills.base import BaseSkill, SkillInput, SkillMeta, SkillOutput

logger = logging.getLogger(__name__)

_DEFAULT_NUM_RESULTS = 10


class ExaSearchSkill(BaseSkill):
    """Semantic web and academic search via the Exa neural search API.

    Exa uses neural embeddings to rank results by conceptual similarity rather
    than keyword overlap, making it significantly better at finding academic
    papers, conceptual content, and cross-lingual material than keyword-based
    engines.

    Supports two modes:
    - Search mode (``query`` parameter): semantic retrieval with optional
      full-text content extraction and category/date filters.
    - Fetch mode (``url`` parameter): retrieve clean markdown text from a
      specific URL — useful as a fallback fetcher for bot-blocked pages.

    Requires EXA_API_KEY to be set in the environment.
    """

    SKILL_META = SkillMeta(
        name="exa_search",
        description=(
            "Semantic web and academic search via the Exa neural API. "
            "Finds conceptually similar content using embeddings — far better "
            "than keyword search for academic queries and cross-lingual topics. "
            "Use search_type='deep' for thorough research synthesis (4-12 s, "
            "multi-query), 'auto' for balanced speed/quality (default), "
            "'fast' for real-time lookups. Supports category='research paper' "
            "for academic-only results. Also fetches clean text from a specific "
            "URL via Exa's /contents endpoint when 'url' is provided instead of 'query'."
        ),
        version="1.1.0",
        author="ai_skill",
        license="MIT",
        tags=[
            "search",
            "semantic",
            "exa",
            "neural",
            "academic",
            "web",
            "papers",
            "deep-research",
            "information-retrieval",
        ],
        dependencies=["EXA_API_KEY"],
    )

    def __init__(self) -> None:
        """Initialise the skill and read the API key from the environment."""
        self._api_key = os.environ.get("EXA_API_KEY", "")

    def run(self, input: SkillInput) -> SkillOutput:
        """Execute a semantic search or URL content fetch via Exa.

        Args:
            input: SkillInput with parameters: query OR url (one required),
                num_results, use_autoprompt, category, include_text,
                start_published_date, end_published_date.

        Returns:
            SkillOutput with result containing:
                results (list[dict]): [{title, url, snippet, score, published_date}].
                query (str): The original query.
                backend (str): "exa".
                total_found (int): Number of results returned.
        """
        if not self._api_key:
            return self._error_output(
                "EXA_API_KEY is not configured. Set the environment variable to enable Exa search."
            )

        params = input.parameters
        query: str | None = params.get("query")
        url: str | None = params.get("url")

        if not query and not url:
            return self._error_output("Parameter 'query' or 'url' is required.")

        # ── URL fetch mode ────────────────────────────────────────────────────
        if url and not query:
            return self._fetch_url(url)

        # ── Search mode ───────────────────────────────────────────────────────
        num_results: int = int(params.get("num_results", _DEFAULT_NUM_RESULTS))
        search_type: str = params.get("search_type", "auto")
        category: str | None = params.get("category")
        include_text: bool = bool(params.get("include_text", True))
        max_characters: int = int(params.get("max_characters", 2000))
        start_date: str | None = params.get("start_published_date")
        end_date: str | None = params.get("end_published_date")
        max_age_hours: int | None = params.get("max_age_hours")

        try:
            from exa_py import Exa  # type: ignore[import-untyped]
        except ImportError:
            return self._error_output(
                "exa-py is not installed. Run: uv add exa-py"
            )

        try:
            exa = Exa(api_key=self._api_key)

            # Build search kwargs — never pass deprecated useAutoprompt
            search_kwargs: dict[str, Any] = {
                "num_results": min(num_results, 25),
                "type": search_type,
            }
            if category:
                search_kwargs["category"] = category
            if start_date:
                search_kwargs["start_published_date"] = start_date
            if end_date:
                search_kwargs["end_published_date"] = end_date

            contents_kwargs: dict[str, Any] = {}
            if include_text:
                text_opts: dict[str, Any] = {"max_characters": max_characters}
                if max_age_hours is not None:
                    text_opts["maxAgeHours"] = max_age_hours
                contents_kwargs["text"] = text_opts

            if include_text:
                response = exa.search_and_contents(
                    query,
                    **contents_kwargs,
                    **search_kwargs,
                )
            else:
                response = exa.search(query, **search_kwargs)

        except Exception as exc:
            return self._error_output(f"Exa search failed: {exc}")

        results: list[dict[str, str]] = []
        sources: list[str] = []

        for item in getattr(response, "results", []):
            snippet = ""
            if hasattr(item, "text") and item.text:
                snippet = item.text[:800]
            elif hasattr(item, "highlights") and item.highlights:
                snippet = " … ".join(item.highlights[:3])

            entry: dict[str, str] = {
                "title": getattr(item, "title", "") or "",
                "url": getattr(item, "url", "") or "",
                "snippet": snippet,
                "score": str(getattr(item, "score", "")),
                "published_date": getattr(item, "published_date", "") or "",
                "source": "exa",
            }
            results.append(entry)
            if entry["url"]:
                sources.append(entry["url"])

        if not results:
            return self._error_output("Exa search returned no results.")

        return SkillOutput(
            skill_name="exa_search",
            result={
                "results": results,
                "query": query,
                "backend": "exa",
                "total_found": len(results),
            },
            confidence=0.85 if results else 0.0,
            sources=sources,
            error=None,
            cached=False,
        )

    def _fetch_url(self, url: str) -> SkillOutput:
        """Fetch clean text content from a URL using Exa's get_contents.

        Args:
            url: The URL to fetch.

        Returns:
            SkillOutput with result containing: text, url, backend.
        """
        try:
            from exa_py import Exa  # type: ignore[import-untyped]
        except ImportError:
            return self._error_output("exa-py is not installed. Run: uv add exa-py")

        try:
            exa = Exa(api_key=self._api_key)
            # /contents endpoint: text is top-level, not nested in contents={}
            response = exa.get_contents([url], text={"max_characters": 5000})
            items = getattr(response, "results", [])
            if not items:
                return self._error_output(f"Exa get_contents returned no content for {url}.")

            item = items[0]
            text = getattr(item, "text", "") or ""
            title = getattr(item, "title", "") or ""

        except Exception as exc:
            return self._error_output(f"Exa get_contents failed for {url}: {exc}")

        return SkillOutput(
            skill_name="exa_search",
            result={
                "text": text,
                "title": title,
                "url": url,
                "backend": "exa_fetch",
            },
            confidence=0.9 if text else 0.0,
            sources=[url],
            error=None if text else f"No content retrieved from {url}.",
            cached=False,
        )


SKILL_CLASS = ExaSearchSkill
