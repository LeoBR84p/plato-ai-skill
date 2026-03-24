"""Tavily search and content-extraction skill.

Tavily is optimised for AI agents: it returns clean, relevance-ranked snippets
and supports an /extract endpoint that retrieves full page text from known URLs.

Two modes:
- Search mode (``query`` param): real-time web search with optional AI answer.
- Extract mode (``url`` param): fetch clean content from a known URL via Tavily's
  /extract endpoint — useful as a fallback fetcher for bot-blocked pages that
  urllib cannot reach but that don't require full JS rendering.

Position in CP2 fallback chains
---------------------------------
Web search:  Brave → Exa → **Tavily** → DuckDuckGo
URL content: HTTP → **Tavily extract** → Firecrawl → Playwright

Parameters accepted in SkillInput.parameters:
    query (str): Search query. Required in search mode.
    url (str): URL to extract content from. Required in extract mode.
    max_results (int): Search results to return. Default: 10.
    search_depth (str): "basic" (fast) or "advanced" (thorough). Default: "basic".
    include_answer (bool): Request an AI-synthesised answer. Default: False.
    include_raw_content (bool): Include full page text in search results. Default: False.
    include_domains (list[str]): Restrict results to these domains. Optional.
    exclude_domains (list[str]): Exclude these domains. Optional.

Environment variables:
    TAVILY_API_KEY    API key for the Tavily search service.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from ai_skill.skills.base import BaseSkill, SkillInput, SkillMeta, SkillOutput

logger = logging.getLogger(__name__)

_DEFAULT_MAX_RESULTS = 10


class TavilySearchSkill(BaseSkill):
    """Real-time web search and URL content extraction via Tavily.

    Tavily is purpose-built for AI agents: results are pre-ranked for relevance,
    snippets are clean, and the optional AI answer synthesises the top results
    into a single paragraph. The /extract endpoint fetches full page text from
    known URLs, making it a lightweight fallback between urllib and Firecrawl.

    Requires TAVILY_API_KEY to be set in the environment.
    """

    SKILL_META = SkillMeta(
        name="tavily_search",
        description=(
            "Real-time web search and URL content extraction via Tavily. "
            "Optimised for AI agents: clean ranked snippets, optional AI answer, "
            "and an /extract endpoint to fetch full page text from known URLs. "
            "In CP2: search fallback after Exa (before DuckDuckGo); "
            "URL fetch fallback after urllib (before Firecrawl)."
        ),
        version="1.0.0",
        author="ai_skill",
        license="MIT",
        tags=[
            "search",
            "tavily",
            "web",
            "extract",
            "content-fetch",
            "ai-optimised",
            "information-retrieval",
        ],
        dependencies=["TAVILY_API_KEY"],
    )

    def __init__(self) -> None:
        """Initialise the skill and read the API key from the environment."""
        self._api_key = os.environ.get("TAVILY_API_KEY", "")

    def run(self, input: SkillInput) -> SkillOutput:
        """Execute a Tavily search or URL content extraction.

        Dispatches to search mode when ``query`` is provided, or extract mode
        when ``url`` is provided (and ``query`` is absent).

        Args:
            input: SkillInput with parameters described in the module docstring.

        Returns:
            SkillOutput with result containing:
                In search mode — results (list[dict]), answer (str|None),
                    query (str), backend (str), total_found (int).
                In extract mode — text (str), url (str), backend (str).
        """
        if not self._api_key:
            return self._error_output(
                "TAVILY_API_KEY is not configured. "
                "Set the environment variable to enable Tavily."
            )

        params = input.parameters
        query: str | None = params.get("query")
        url: str | None = params.get("url")

        if not query and not url:
            return self._error_output("Parameter 'query' or 'url' is required.")

        if url and not query:
            return self._extract_url(url)

        return self._search(params, query)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Search mode
    # ------------------------------------------------------------------

    def _search(self, params: dict[str, Any], query: str) -> SkillOutput:
        max_results: int = int(params.get("max_results", _DEFAULT_MAX_RESULTS))
        search_depth: str = params.get("search_depth", "basic")
        include_answer: bool = bool(params.get("include_answer", False))
        include_raw: bool = bool(params.get("include_raw_content", False))
        include_domains: list[str] = params.get("include_domains", [])
        exclude_domains: list[str] = params.get("exclude_domains", [])

        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]
        except ImportError:
            return self._error_output("tavily-python is not installed. Run: uv add tavily-python")

        try:
            client = TavilyClient(api_key=self._api_key)
            kwargs: dict[str, Any] = {
                "max_results": min(max_results, 20),
                "search_depth": search_depth,
                "include_answer": include_answer,
                "include_raw_content": include_raw,
            }
            if include_domains:
                kwargs["include_domains"] = include_domains
            if exclude_domains:
                kwargs["exclude_domains"] = exclude_domains

            response: dict[str, Any] = client.search(query, **kwargs)
        except Exception as exc:
            return self._error_output(f"Tavily search failed: {exc}")

        raw_results: list[dict[str, Any]] = response.get("results", [])
        answer: str | None = response.get("answer") or None

        results: list[dict[str, str]] = []
        sources: list[str] = []
        for item in raw_results:
            url_val = item.get("url", "") or ""
            results.append({
                "title": item.get("title", "") or "",
                "url": url_val,
                "snippet": item.get("content", "") or "",
                "score": str(item.get("score", "")),
                "published_date": item.get("published_date", "") or "",
                "source": "tavily",
            })
            if url_val:
                sources.append(url_val)

        if not results:
            return self._error_output("Tavily search returned no results.")

        return SkillOutput(
            skill_name="tavily_search",
            result={
                "results": results,
                "answer": answer,
                "query": query,
                "backend": "tavily",
                "total_found": len(results),
            },
            confidence=0.85 if results else 0.0,
            sources=sources,
            error=None,
            cached=False,
        )

    # ------------------------------------------------------------------
    # Extract mode (URL → full text)
    # ------------------------------------------------------------------

    def _extract_url(self, url: str) -> SkillOutput:
        """Fetch clean text content from a URL via Tavily's /extract endpoint.

        Args:
            url: The URL to extract content from.

        Returns:
            SkillOutput with result containing: text, url, backend.
        """
        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]
        except ImportError:
            return self._error_output(
                "tavily-python is not installed. Run: uv add tavily-python"
            )

        try:
            client = TavilyClient(api_key=self._api_key)
            response: dict[str, Any] = client.extract(urls=[url])
        except Exception as exc:
            return self._error_output(f"Tavily extract failed for {url}: {exc}")

        results = response.get("results", [])
        if not results:
            failed = response.get("failed_results", [])
            reason = failed[0].get("error", "unknown") if failed else "no results"
            return self._error_output(f"Tavily could not extract {url}: {reason}")

        text: str = results[0].get("raw_content", "") or ""
        if not text:
            return self._error_output(f"Tavily extract returned empty content for {url}.")

        logger.debug("tavily_search extract: %s — %d chars", url, len(text))

        return SkillOutput(
            skill_name="tavily_search",
            result={
                "text": text,
                "url": url,
                "backend": "tavily_extract",
            },
            confidence=0.8 if text else 0.0,
            sources=[url],
            error=None,
            cached=False,
        )


SKILL_CLASS = TavilySearchSkill
