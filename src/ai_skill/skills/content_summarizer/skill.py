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
import os
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

    def run(self, input: SkillInput) -> SkillOutput:
        """Summarise the provided content.

        Args:
            input: SkillInput with parameters: content (required), content_type,
                max_length, focus_areas, source_url, source_year.

        Returns:
            SkillOutput with result containing the SummaryOutput fields.
        """
        params = input.parameters
        content: str | None = params.get("content")
        if not content or not content.strip():
            return self._error_output("Parameter 'content' is required and must not be empty.")

        content_type: str = params.get("content_type", "text")
        max_length: int = int(params.get("max_length", 200))
        focus_areas: list[str] = params.get("focus_areas", [])
        source_url: str = params.get("source_url", "")
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
