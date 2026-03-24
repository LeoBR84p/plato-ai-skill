"""Firecrawl web scraper skill implementation.

Firecrawl renders JavaScript-heavy pages (React/Vue SPAs, bot-protected sites)
and returns clean Markdown text. This solves the main class of failures seen
in CP2 source verification:
  - Semantic Scholar paper pages (SPA — blank with urllib)
  - McKinsey/corporate sites (bot detection — timeout with urllib)
  - Journal landing pages requiring JS for full abstract display

Primary use in CP2:
  - Called by the planner explicitly to scrape a specific URL for verification
  - Used as fallback in content_summarizer when urllib returns empty content
  - Used as fallback in _fetch_url_content() when HTTP fetch fails

Parameters accepted in SkillInput.parameters:
    url (str): The URL to scrape. Required.
    formats (list[str]): Output formats — ["markdown"], ["html"], or both.
        Default: ["markdown"].
    only_main_content (bool): Strip navigation/headers/footers. Default: True.
    timeout (int): Scrape timeout in milliseconds. Default: 30000.

Environment variables:
    FIRECRAWL_API_KEY    API key for the Firecrawl service.
"""

from __future__ import annotations

import logging
import os

from ai_skill.skills.base import BaseSkill, SkillInput, SkillMeta, SkillOutput

logger = logging.getLogger(__name__)


class FirecrawlScraperSkill(BaseSkill):
    """Scrape JavaScript-rendered pages via the Firecrawl API.

    Firecrawl uses a headless browser under the hood to fully render the page
    before returning clean Markdown. This makes it effective for:
    - React/Vue/Angular SPAs (e.g. Semantic Scholar, ResearchGate)
    - Corporate sites with bot-detection (e.g. McKinsey, Gartner)
    - Pages that require JS for content delivery

    Returns clean Markdown text stripped of navigation and boilerplate.

    Requires FIRECRAWL_API_KEY to be set in the environment.
    """

    SKILL_META = SkillMeta(
        name="firecrawl_scraper",
        description=(
            "Scrape JavaScript-rendered or bot-protected web pages via Firecrawl. "
            "Uses a headless browser to fully render the page and returns clean "
            "Markdown. Essential for Semantic Scholar, McKinsey, ResearchGate, "
            "and any SPA or bot-blocked site that returns empty content via urllib."
        ),
        version="1.0.0",
        author="ai_skill",
        license="MIT",
        tags=[
            "scraping",
            "firecrawl",
            "web",
            "javascript",
            "spa",
            "bot-bypass",
            "markdown",
            "content-extraction",
        ],
        dependencies=["FIRECRAWL_API_KEY"],
    )

    def __init__(self) -> None:
        """Initialise the skill and read the API key from the environment."""
        self._api_key = os.environ.get("FIRECRAWL_API_KEY", "")

    def run(self, input: SkillInput) -> SkillOutput:
        """Scrape a URL via Firecrawl and return clean Markdown content.

        Args:
            input: SkillInput with parameters: url (required), formats,
                only_main_content, timeout.

        Returns:
            SkillOutput with result containing:
                markdown (str): Extracted Markdown content.
                html (str): Raw HTML (only when "html" in formats).
                url (str): The scraped URL.
                title (str): Page title if extractable.
                char_count (int): Length of extracted text.
        """
        if not self._api_key:
            return self._error_output(
                "FIRECRAWL_API_KEY is not configured. "
                "Set the environment variable to enable Firecrawl scraping."
            )

        params = input.parameters
        url: str | None = params.get("url")
        if not url:
            return self._error_output("Parameter 'url' is required.")

        formats: list[str] = params.get("formats", ["markdown"])
        only_main_content: bool = bool(params.get("only_main_content", True))
        timeout_ms: int = int(params.get("timeout", 30000))

        try:
            from firecrawl import FirecrawlApp  # type: ignore[import-untyped]
        except ImportError:
            return self._error_output(
                "firecrawl-py is not installed. Run: uv add firecrawl-py"
            )

        try:
            app = FirecrawlApp(api_key=self._api_key)
            scrape_params: dict = {
                "formats": formats,
                "onlyMainContent": only_main_content,
                "timeout": timeout_ms,
            }
            # firecrawl-py ≥ 1.0 uses .scrape(); older versions used .scrape_url()
            scrape_fn = getattr(app, "scrape", None) or getattr(app, "scrape_url")
            result = scrape_fn(url, **scrape_params)
        except Exception as exc:
            return self._error_output(f"Firecrawl scrape failed for {url}: {exc}")

        # Normalise result — firecrawl-py may return object or dict
        markdown = ""
        html = ""
        title = ""

        if hasattr(result, "markdown"):
            markdown = result.markdown or ""
        elif isinstance(result, dict):
            markdown = result.get("markdown", "") or ""

        if hasattr(result, "html"):
            html = result.html or ""
        elif isinstance(result, dict):
            html = result.get("html", "") or ""

        if hasattr(result, "metadata"):
            meta = result.metadata or {}
            title = (meta.get("title") or "") if isinstance(meta, dict) else ""
        elif isinstance(result, dict):
            title = (result.get("metadata") or {}).get("title", "") or ""

        text = markdown or html
        if not text:
            return self._error_output(
                f"Firecrawl returned empty content for {url}. "
                "The page may require authentication or be otherwise restricted."
            )

        logger.debug(
            "firecrawl_scraper: scraped %s — %d chars (markdown), title=%r",
            url,
            len(markdown),
            title,
        )

        return SkillOutput(
            skill_name="firecrawl_scraper",
            result={
                "markdown": markdown,
                "html": html,
                "url": url,
                "title": title,
                "char_count": len(text),
            },
            confidence=0.9 if markdown else 0.7,
            sources=[url],
            error=None,
            cached=False,
        )


SKILL_CLASS = FirecrawlScraperSkill
