"""Web search skill implementation.

Primary backend:  Brave Search API (BRAVE_SEARCH key) — real-time web results
    optimised for AI agents, with optional AI-generated answer (BRAVE_ANSWER key).
Fallback chain:
    1. Brave Search (primary, best quality)
    2. Exa neural search (semantic/embedding-based, great for academic queries)
    3. Tavily (AI-optimised, clean snippets, optional answer synthesis)
    4. DuckDuckGo (keyword-based, no API key, always available)

All URLs returned are pre-filtered through UrlSafetyGuard (Google Safe Browsing v4).

Parameters accepted in SkillInput.parameters:
    query (str): The search query. Required.
    max_results (int): Maximum results to return. Default: 10.
    language (str): BCP-47 language code for results. Default: "pt-BR".
    country (str): Two-letter country code for result localisation. Default: "BR".
    freshness (str): Restrict by age — "pd" (day), "pw" (week), "pm" (month),
        "py" (year). Optional.
    include_answer (bool): Request a Brave AI-generated answer alongside results.
        Requires BRAVE_ANSWER key. Default: False.

Environment variables:
    BRAVE_SEARCH     API key for Brave Web Search API.
    BRAVE_ANSWER     API key for Brave AI Answers API (optional enrichment).
    EXA_API_KEY      API key for Exa neural search (fallback 2).
    TAVILY_API_KEY   API key for Tavily search (fallback 3).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ai_skill.core.url_safety import UrlSafetyGuard
from ai_skill.skills.base import BaseSkill, SkillInput, SkillMeta, SkillOutput

logger = logging.getLogger(__name__)

_BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_BRAVE_ANSWER_ENDPOINT = "https://api.search.brave.com/res/v1/summarizer/search"
_DEFAULT_MAX_RESULTS = 10


class WebSearchSkill(BaseSkill):
    """Search the web for information relevant to the research objective.

    Fallback chain (highest to lowest quality):
    1. Brave Search API — real-time, LLM-optimised results.
    2. Exa neural search — embedding-based semantic search; superior for
       academic and conceptual queries when Brave fails or is unconfigured.
    3. DuckDuckGo — keyword-based, no API key, always available as last resort.

    Optionally enriches Brave results with an AI-generated answer when
    BRAVE_ANSWER is configured.

    All returned URLs are checked against Google Safe Browsing v4 before
    being included in the output.
    """

    SKILL_META = SkillMeta(
        name="web_search",
        description=(
            "Search the web using Brave (primary, LLM-optimised), "
            "Exa neural/semantic search (great for academic queries), "
            "Tavily (AI-optimised, clean snippets), "
            "or DuckDuckGo (last resort, no API key). "
            "All URLs are pre-filtered by Google Safe Browsing v4."
        ),
        version="1.3.0",
        author="ai_skill",
        license="MIT",
        tags=["search", "web", "brave", "exa", "tavily", "information-retrieval", "duckduckgo"],
        dependencies=[],
    )

    def __init__(self) -> None:
        """Initialise the skill with Brave/Exa/Tavily credentials and URL safety guard."""
        self._safety_guard = UrlSafetyGuard()
        self._brave_search_key = os.environ.get("BRAVE_SEARCH")
        self._brave_answer_key = os.environ.get("BRAVE_ANSWER")
        self._exa_key = os.environ.get("EXA_API_KEY")
        self._tavily_key = os.environ.get("TAVILY_API_KEY")

    def run(self, input: SkillInput) -> SkillOutput:
        """Execute a web search query.

        Args:
            input: SkillInput with parameters: query (required), max_results,
                language, country, freshness, include_answer.

        Returns:
            SkillOutput with result containing:
                results (list[dict]): [{title, url, snippet, source}].
                answer (str | None): AI-generated answer if requested and available.
                query (str): The original query.
                backend (str): Which backend was used.
                total_found (int): Number of safe results returned.
        """
        params = input.parameters
        query: str | None = params.get("query")
        if not query:
            return self._error_output("Parameter 'query' is required.")

        max_results: int = int(params.get("max_results", _DEFAULT_MAX_RESULTS))
        language: str = params.get("language", "pt-BR")
        country: str = params.get("country", "BR")
        freshness: str = params.get("freshness", "")
        include_answer: bool = bool(params.get("include_answer", False))

        results: list[dict[str, str]] = []
        answer: str | None = None
        backend: str
        brave_exc: Exception | None = None

        # ── Backend 1: Brave ──────────────────────────────────────────────────
        if self._brave_search_key:
            try:
                results = self._brave_search(
                    query, max_results, language, country, freshness
                )
                backend = "brave"

                if include_answer and self._brave_answer_key:
                    try:
                        answer = self._brave_answer(query)
                    except Exception as exc:
                        logger.warning("Brave Answer API failed: %s", exc)

            except Exception as exc:
                brave_exc = exc
                logger.warning("Brave Search failed (%s). Trying Exa.", exc)
                results = []

        # ── Backend 2: Exa (fallback when Brave fails or is unconfigured) ─────
        if not results and self._exa_key:
            try:
                results = self._exa_search(query, max_results)
                backend = "exa_fallback" if brave_exc else "exa"
            except Exception as exa_exc:
                logger.warning("Exa search failed (%s). Trying DuckDuckGo.", exa_exc)
                results = []

        # ── Backend 3: Tavily (AI-optimised, after Exa) ───────────────────────
        if not results and self._tavily_key:
            try:
                results = self._tavily_search(query, max_results)
                backend = "tavily_fallback" if brave_exc else "tavily"
            except Exception as tv_exc:
                logger.warning("Tavily search failed (%s). Trying DuckDuckGo.", tv_exc)
                results = []

        # ── Backend 4: DuckDuckGo (last resort, no API key) ───────────────────
        if not results:
            if not self._brave_search_key and not self._exa_key and not self._tavily_key:
                logger.info("No search API keys configured. Using DuckDuckGo.")
            try:
                results = self._duckduckgo_search(query, max_results)
                backend = "duckduckgo"
            except Exception as ddg_exc:
                all_errors = f"DuckDuckGo: {ddg_exc}"
                if brave_exc:
                    all_errors = f"Brave: {brave_exc}. {all_errors}"
                return self._error_output(f"All search backends failed. {all_errors}")

        # Apply URL safety filter
        urls = [r["url"] for r in results if r.get("url")]
        safe_urls = set(self._safety_guard.filter(urls))
        safe_results = [r for r in results if r.get("url") in safe_urls]

        blocked = len(results) - len(safe_results)
        if blocked:
            logger.info(
                "web_search: %d/%d URLs blocked by Safe Browsing.", blocked, len(results)
            )

        return SkillOutput(
            skill_name="web_search",
            result={
                "results": safe_results,
                "answer": answer,
                "query": query,
                "backend": backend,
                "total_found": len(safe_results),
            },
            confidence=0.9 if safe_results else 0.0,
            sources=[r["url"] for r in safe_results if r.get("url")],
            error=None if safe_results else "No safe results found.",
            cached=False,
        )

    # ------------------------------------------------------------------
    # Brave Search backend
    # ------------------------------------------------------------------

    def _brave_search(
        self,
        query: str,
        max_results: int,
        _language: str,
        _country: str,
        freshness: str,
    ) -> list[dict[str, str]]:
        """Query the Brave Web Search API.

        Args:
            query: Search query string.
            max_results: Maximum results to retrieve (API cap: 20 per request).
            language: BCP-47 search language code (e.g. "pt-BR").
            country: Two-letter country code (e.g. "BR").
            freshness: Age filter string ("pd", "pw", "pm", "py") or empty.

        Returns:
            List of result dicts: [{title, url, snippet, source}].
        """
        # Keep params minimal — Brave returns 422 for unrecognised/invalid values.
        count = min(max_results, 20)
        request_params: dict[str, Any] = {
            "q": query,
            "count": count,
            "safesearch": "moderate",
        }
        if freshness:
            request_params["freshness"] = freshness

        response = httpx.get(
            _BRAVE_SEARCH_ENDPOINT,
            params=request_params,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._brave_search_key or "",
            },
            timeout=15.0,
        )
        if not response.is_success:
            logger.warning(
                "Brave Search %d: %s", response.status_code, response.text[:300]
            )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

        results: list[dict[str, str]] = []
        for item in data.get("web", {}).get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
                "source": "brave",
            })
        return results

    def _brave_answer(self, query: str) -> str | None:
        """Request a Brave AI-generated answer via the two-step summarizer flow.

        Step 1: web search with ``summary=1`` → obtains a ``summarizer.key``.
        Step 2: call summarizer endpoint with that key → extracts answer text.

        Args:
            query: The research query.

        Returns:
            The AI answer string, or None if the plan does not support summarizer
            or no summary was generated.
        """
        # Step 1 — web search to obtain summarizer key
        r1 = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": 5, "summary": 1},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._brave_search_key or self._brave_answer_key or "",
            },
            timeout=15.0,
        )
        r1.raise_for_status()
        summarizer_key: str | None = (r1.json().get("summarizer") or {}).get("key")
        if not summarizer_key:
            logger.debug("Brave summarizer key not returned — plan may not include AI Answers.")
            return None

        # Step 2 — retrieve the generated answer
        r2 = httpx.get(
            _BRAVE_ANSWER_ENDPOINT,
            params={"key": summarizer_key, "entity_info": 1},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": self._brave_answer_key or "",
            },
            timeout=20.0,
        )
        r2.raise_for_status()
        data: dict[str, Any] = r2.json()

        summary = data.get("summary", [])
        if isinstance(summary, list) and summary:
            parts = [s.get("data", "") for s in summary if s.get("type") == "token"]
            return "".join(parts).strip() or None
        return None

    # ------------------------------------------------------------------
    # Exa neural search fallback
    # ------------------------------------------------------------------

    def _exa_search(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Query the Exa neural search API.

        Args:
            query: Search query string.
            max_results: Maximum results to retrieve.

        Returns:
            List of result dicts: [{title, url, snippet, score, source}].
        """
        try:
            from exa_py import Exa  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install the exa-py package: uv add exa-py") from exc

        exa = Exa(api_key=self._exa_key or "")
        # type="auto" gives best balance of speed/quality for fallback queries.
        # Never pass use_autoprompt — it is deprecated and will raise a warning.
        response = exa.search_and_contents(
            query,
            type="auto",
            num_results=min(max_results, 25),
            text={"max_characters": 800},
        )

        results: list[dict[str, str]] = []
        for item in getattr(response, "results", []):
            snippet = ""
            if hasattr(item, "text") and item.text:
                snippet = item.text[:500]
            results.append({
                "title": getattr(item, "title", "") or "",
                "url": getattr(item, "url", "") or "",
                "snippet": snippet,
                "score": str(getattr(item, "score", "")),
                "source": "exa",
            })
        return results

    # ------------------------------------------------------------------
    # Tavily fallback
    # ------------------------------------------------------------------

    def _tavily_search(self, query: str, max_results: int) -> list[dict[str, str]]:
        """Query the Tavily search API.

        Args:
            query: Search query string.
            max_results: Maximum results to retrieve.

        Returns:
            List of result dicts: [{title, url, snippet, score, source}].
        """
        try:
            from tavily import TavilyClient  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError("Install the tavily-python package: uv add tavily-python") from exc

        client = TavilyClient(api_key=self._tavily_key or "")
        response = client.search(
            query,
            max_results=min(max_results, 20),
            search_depth="basic",
        )

        results: list[dict[str, str]] = []
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", "") or "",
                "url": item.get("url", "") or "",
                "snippet": item.get("content", "") or "",
                "score": str(item.get("score", "")),
                "source": "tavily",
            })
        return results

    # ------------------------------------------------------------------
    # DuckDuckGo fallback
    # ------------------------------------------------------------------

    def _duckduckgo_search(
        self, query: str, max_results: int
    ) -> list[dict[str, str]]:
        """Query DuckDuckGo using the duckduckgo-search library.

        Args:
            query: Search query string.
            max_results: Maximum results to retrieve.

        Returns:
            List of result dicts: [{title, url, snippet, source}].
        """
        try:
            from ddgs import DDGS  # package renamed from duckduckgo_search to ddgs
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # type: ignore[no-redef]
            except ImportError as exc:
                raise ImportError(
                    "Install the ddgs package: uv add ddgs"
                ) from exc

        results: list[dict[str, str]] = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                    "source": "duckduckgo",
                })
        return results


SKILL_CLASS = WebSearchSkill
