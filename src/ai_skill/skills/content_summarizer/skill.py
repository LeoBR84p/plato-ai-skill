"""Content summarizer skill implementation.

Uses the LLM to summarise content from any source (text, article, PDF extract).
Returns a structured summary with key points, entities, and a relevance score
relative to the current research objective.

Parameters accepted in SkillInput.parameters:
    content (str): The text content to summarise. Required.
    content_type (str): "text" | "article" | "pdf". Default: "text".
    max_length (int): Target summary length in words. Default: 200.
    focus_areas (list[str]): Specific aspects to emphasise in the summary.
    source_url (str): Origin URL for attribution (optional).
    source_year (int): Publication year for datation of claims (optional).
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ai_skill.core.llm_client import LLMClient, LLMClientError
from ai_skill.skills.base import BaseSkill, SkillInput, SkillMeta, SkillOutput

logger = logging.getLogger(__name__)

_SUMMARISE_SYSTEM = """\
You are an academic research assistant. Summarise the provided content
concisely and extract structured information for the research pipeline.

Guidelines:
- Write the summary in the same language as the content or in the research
  objective language if specified.
- Mark claims from sources older than 5 years as potentially_outdated.
- Be factual and avoid adding information not present in the content.
- Relevance score: 0.0 (irrelevant) to 1.0 (highly relevant to research topic).
"""

_SUMMARISE_USER = """\
Content type: {content_type}
Research topic: {topic}
Focus areas: {focus_areas}
Target length: approximately {max_length} words

Content to summarise:
---
{content_truncated}
---
"""


class SummaryOutput(BaseModel):
    """Structured output from the content summarizer.

    Attributes:
        summary: Concise summary of the content.
        key_points: 3-7 key points extracted from the content.
        entities: Named entities (authors, institutions, methods, datasets).
        relevance_score: How relevant the content is to the research topic (0-1).
        potentially_outdated: True if source year is more than 5 years ago.
        language: Detected language of the content.
    """

    summary: str = Field(description="Concise summary of the content.")
    key_points: list[str] = Field(
        description="3-7 key points from the content.", default_factory=list
    )
    entities: dict[str, list[str]] = Field(
        description="Named entities grouped by type (authors, methods, datasets).",
        default_factory=dict,
    )
    relevance_score: float = Field(
        description="Relevance to research topic, 0.0-1.0.", ge=0.0, le=1.0, default=0.5
    )
    potentially_outdated: bool = Field(
        description="True if the source is more than 5 years old.", default=False
    )
    language: str = Field(description="Detected content language.", default="pt-BR")


_MAX_CONTENT_CHARS = 12000  # ~3000 tokens; stay within context limit


class ContentSummarizerSkill(BaseSkill):
    """Summarise text content from any source using the LLM.

    Produces a structured summary with key points, named entities, a relevance
    score relative to the research topic, and a staleness flag for old sources.

    Requires ANTHROPIC_API_KEY to be set in the environment.
    """

    SKILL_META = SkillMeta(
        name="content_summarizer",
        description=(
            "Summarise text, articles, or PDF extracts using the LLM. "
            "Returns structured output: summary, key points, entities, "
            "relevance score, and staleness flag."
        ),
        version="1.0.0",
        author="ai_skill",
        license="MIT",
        tags=["summarization", "nlp", "llm", "extraction", "academic", "text"],
        dependencies=[],
    )

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        """Initialise the skill with an LLM client.

        Args:
            llm_client: Optional pre-configured LLMClient. When None, a new
                client is created from environment variables. Inject a mock
                client for testing.
        """
        self._llm: LLMClient | None = llm_client
        self._current_year = int(
            __import__("datetime").datetime.now().year
        )

    def _get_llm(self) -> LLMClient:
        """Lazily initialise the LLM client.

        Returns:
            The LLMClient instance.
        """
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    @staticmethod
    def _fetch_text(url: str, timeout: int = 10) -> str:
        """Fetch plain text from *url* using a multi-tier fallback chain.

        Handles both HTTP(S) URLs and local file paths. Local PDF paths are
        extracted via PyMuPDF (or pypdf fallback) rather than being read as
        raw bytes — which would produce unreadable binary content.

        Fallback order for HTTP URLs:
        1. ``urllib.request`` + BeautifulSoup (fast, no API key needed).
        2. Tavily extract (bypasses bot-blocking; no headless browser).
        3. Firecrawl (headless-browser scraper — resolves JS SPAs).
        4. Playwright (self-hosted headless Chromium — last resort).

        Returns empty string only when all methods fail.
        """
        import re
        from pathlib import Path

        # ── Local file path: extract text directly ────────────────────────────
        # Guard against urllib silently reading binary PDF bytes as UTF-8.
        is_local = url and not url.startswith(("http://", "https://"))
        if is_local:
            local_path = Path(url)
            if not local_path.exists():
                logger.debug("_fetch_text: local path not found: %s", url)
                return ""
            if local_path.suffix.lower() == ".pdf":
                try:
                    import pymupdf as _fitz  # PyMuPDF ≥ 1.24
                except ImportError:
                    try:
                        import fitz as _fitz  # type: ignore[no-redef]
                    except ImportError:
                        _fitz = None  # type: ignore[assignment]
                if _fitz is not None:
                    try:
                        doc = _fitz.open(str(local_path))
                        parts = [doc[i].get_text(sort=True) for i in range(min(len(doc), 30))]
                        doc.close()
                        return "\n\n".join(parts)[:_MAX_CONTENT_CHARS]
                    except Exception as exc:
                        logger.debug("_fetch_text: PyMuPDF failed for %s: %s", url, exc)
                # pypdf fallback
                try:
                    import pypdf as _pypdf
                    with open(local_path, "rb") as _f:
                        _reader = _pypdf.PdfReader(_f)
                        parts = [p.extract_text() or "" for p in _reader.pages[:30]]
                    return "\n\n".join(parts)[:_MAX_CONTENT_CHARS]
                except Exception as exc:
                    logger.debug("_fetch_text: pypdf fallback failed for %s: %s", url, exc)
            else:
                try:
                    return local_path.read_text(encoding="utf-8", errors="ignore")[:_MAX_CONTENT_CHARS]
                except Exception as exc:
                    logger.debug("_fetch_text: text file read failed for %s: %s", url, exc)
            return ""

        import urllib.request

        # ── Tier 1: direct HTTP + BeautifulSoup ──────────────────────────────
        html = ""
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ai-skill-summarizer/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(65536)
                html = raw.decode("utf-8", errors="ignore")
        except Exception:
            html = ""

        if html:
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
            except Exception:
                text = re.sub(r"<[^>]+>", " ", html)
                text = re.sub(r"\s{2,}", " ", text).strip()

            # Accept the result only when it contains meaningful content.
            # JS-only SPA shells produce < 300 visible chars — escalate.
            if len(text) >= 300:
                return text
            logger.debug(
                "_fetch_text: near-empty HTTP response for %s (%d chars) — escalating.",
                url, len(text),
            )

        # ── Tier 2: Tavily extract ────────────────────────────────────────────
        import os
        tavily_key = os.environ.get("TAVILY_API_KEY", "")
        if tavily_key:
            try:
                from tavily import TavilyClient  # type: ignore[import-untyped]
                tv_client = TavilyClient(api_key=tavily_key)
                tv_resp = tv_client.extract(urls=[url])
                tv_results = tv_resp.get("results", [])
                if tv_results:
                    tv_text = tv_results[0].get("raw_content", "") or ""
                    if tv_text:
                        logger.debug(
                            "_fetch_text: Tavily extract succeeded for %s (%d chars).",
                            url, len(tv_text),
                        )
                        return tv_text[:_MAX_CONTENT_CHARS]
            except Exception as exc:
                logger.debug("_fetch_text: Tavily extract failed for %s: %s", url, exc)

        # ── Tier 3: Firecrawl ─────────────────────────────────────────────────
        firecrawl_key = os.environ.get("FIRECRAWL_API_KEY", "")
        if firecrawl_key:
            try:
                from firecrawl import FirecrawlApp  # type: ignore[import-untyped]
                app = FirecrawlApp(api_key=firecrawl_key)
                # firecrawl-py ≥ 1.0 uses .scrape(); older versions used .scrape_url()
                scrape_fn = getattr(app, "scrape", None) or getattr(app, "scrape_url")
                result = scrape_fn(url, formats=["markdown"])
                fc_text = ""
                if hasattr(result, "markdown") and result.markdown:
                    fc_text = result.markdown
                elif isinstance(result, dict):
                    fc_text = result.get("markdown", "") or result.get("content", "")
                if fc_text:
                    logger.debug(
                        "_fetch_text: Firecrawl succeeded for %s (%d chars).", url, len(fc_text)
                    )
                    return fc_text[:_MAX_CONTENT_CHARS]
            except Exception as exc:
                logger.debug("_fetch_text: Firecrawl failed for %s: %s", url, exc)

        # ── Tier 4: Playwright ────────────────────────────────────────────────
        try:
            from playwright.sync_api import sync_playwright  # type: ignore[import-untyped]
            with sync_playwright() as pw:
                browser = pw.chromium.launch(headless=True)
                try:
                    context = browser.new_context(
                        user_agent=(
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"
                        )
                    )
                    page = context.new_page()
                    page.goto(url, wait_until="networkidle", timeout=30000)
                    pw_text = page.evaluate(
                        """() => {
                            const remove = ['script','style','nav','footer','header','aside'];
                            remove.forEach(t => document.querySelectorAll(t)
                                .forEach(el => el.remove()));
                            return document.body ? document.body.innerText : '';
                        }"""
                    )
                    pw_text = (pw_text or "").strip()
                finally:
                    browser.close()
            if pw_text:
                logger.debug(
                    "_fetch_text: Playwright succeeded for %s (%d chars).", url, len(pw_text)
                )
                return pw_text[:_MAX_CONTENT_CHARS]
        except ImportError:
            logger.debug("_fetch_text: Playwright not installed — skipping for %s.", url)
        except Exception as exc:
            logger.debug("_fetch_text: Playwright failed for %s: %s", url, exc)

        return ""

    def run(self, input: SkillInput) -> SkillOutput:
        """Summarise the provided content.

        When *content* is absent but *source_url* is supplied, the skill
        fetches the URL automatically and uses the retrieved text as content.

        Args:
            input: SkillInput with parameters: content (text to summarise) or
                source_url (auto-fetched when content is absent), content_type,
                max_length, focus_areas, source_year.

        Returns:
            SkillOutput with result containing the SummaryOutput fields.
        """
        params = input.parameters
        content: str | None = params.get("content")
        source_url: str = params.get("source_url", "")

        if not content or not content.strip():
            if source_url:
                content = self._fetch_text(source_url)
            if not content or not content.strip():
                return self._error_output(
                    "Parameter 'content' is required (or provide 'source_url' for auto-fetch). "
                    f"URL fetch {'returned empty' if source_url else 'not attempted'}."
                )

        content_type: str = params.get("content_type", "text")
        max_length: int = int(params.get("max_length", 200))
        focus_areas: list[str] = params.get("focus_areas", [])
        source_year: int | None = params.get("source_year")

        # Determine research topic for relevance scoring
        topic = ""
        if input.objective:
            topic = input.objective.get("topic", "")

        # Truncate content to avoid exceeding context window
        content_truncated = content[:_MAX_CONTENT_CHARS]
        if len(content) > _MAX_CONTENT_CHARS:
            content_truncated += "\n[... content truncated ...]"

        user_content = _SUMMARISE_USER.format(
            content_type=content_type,
            topic=topic or "not specified",
            focus_areas=", ".join(focus_areas) if focus_areas else "general",
            max_length=max_length,
            content_truncated=content_truncated,
        )

        try:
            summary_obj: SummaryOutput = self._get_llm().complete_structured(
                messages=[{"role": "user", "content": user_content}],
                response_model=SummaryOutput,
                system=_SUMMARISE_SYSTEM,
            )
        except LLMClientError as exc:
            return self._error_output(str(exc))

        # Override staleness flag based on source_year if provided
        if source_year is not None:
            age = self._current_year - source_year
            summary_obj.potentially_outdated = age > 5

        result: dict[str, Any] = {
            "summary": summary_obj.summary,
            "key_points": summary_obj.key_points,
            "entities": summary_obj.entities,
            "relevance_score": summary_obj.relevance_score,
            "potentially_outdated": summary_obj.potentially_outdated,
            "language": summary_obj.language,
            "source_year": source_year,
        }

        sources = [source_url] if source_url else []

        return SkillOutput(
            skill_name="content_summarizer",
            result=result,
            confidence=summary_obj.relevance_score,
            sources=sources,
            error=None,
            cached=False,
        )


SKILL_CLASS = ContentSummarizerSkill
