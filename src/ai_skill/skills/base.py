"""Base class and metadata contract for all ai_skill skills.

Every skill — whether built-in or a third-party plugin — must:
1. Subclass BaseSkill.
2. Define a class-level SKILL_META attribute of type SkillMeta.
3. Implement the run() method.
4. Never raise exceptions from run(): return SkillOutput with error set instead.

Plugin discovery relies on the SKILL_CLASS module attribute:
    # in my_skill/skill.py
    SKILL_CLASS = MySkill
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ai_skill.core.state import ResearchObjective, SkillOutput


@dataclass(frozen=True)
class SkillMeta:
    """Immutable metadata for a skill, mirroring the SKILL.md YAML frontmatter.

    Attributes:
        name: Unique kebab-case identifier (e.g. "web_search").
        description: One-sentence description used in planning prompts.
        version: Semantic version string (e.g. "1.0.0").
        author: Author or organisation name.
        license: SPDX license identifier (e.g. "MIT", "AGPL-3.0").
        tags: 5-10 descriptive tags for categorisation.
        dependencies: Names of other skills this skill requires.
    """

    name: str
    description: str
    version: str = "1.0.0"
    author: str = "ai_skill"
    license: str = "MIT"
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise metadata to a plain dict for inclusion in planning prompts.

        Returns:
            Dict with all metadata fields.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "license": self.license,
            "tags": self.tags,
            "dependencies": self.dependencies,
        }


class BaseSkill(ABC):
    """Abstract base class for all skills.

    Subclasses must define SKILL_META at class level and implement run().

    Example:
        >>> class MySkill(BaseSkill):
        ...     SKILL_META = SkillMeta(name="my_skill", description="Does X")
        ...
        ...     def run(self, input: SkillInput) -> SkillOutput:
        ...         return SkillOutput(
        ...             skill_name="my_skill",
        ...             result={"data": "..."},
        ...             confidence=0.9,
        ...             sources=[],
        ...             error=None,
        ...             cached=False,
        ...         )
    """

    SKILL_META: SkillMeta

    @abstractmethod
    def run(self, input: SkillInput) -> SkillOutput:
        """Execute the skill synchronously.

        Args:
            input: Skill input with parameters and research context.

        Returns:
            SkillOutput with result populated on success, or error set on failure.
            MUST never raise an exception.
        """

    async def arun(self, input: "SkillInput") -> SkillOutput:
        """Execute the skill asynchronously (default: runs run() in a thread).

        Skills that are inherently async can override this method directly.

        Args:
            input: Skill input with parameters and research context.

        Returns:
            SkillOutput — same contract as run().
        """
        return await asyncio.to_thread(self.run, input)

    @property
    def meta(self) -> SkillMeta:
        """Convenience accessor for SKILL_META."""
        return self.SKILL_META

    def _error_output(self, error: str | Exception) -> SkillOutput:
        """Build a standardised error SkillOutput.

        Args:
            error: Error message string or exception.

        Returns:
            SkillOutput with confidence=0.0 and error set.
        """
        return SkillOutput(
            skill_name=self.SKILL_META.name,
            result={},
            confidence=0.0,
            sources=[],
            error=str(error),
            cached=False,
        )


# ---------------------------------------------------------------------------
# SkillInput — defined here to avoid circular imports with core.state
# ---------------------------------------------------------------------------

class SkillInput(dict):  # type: ignore[type-arg]
    """Input container passed to every skill.

    Behaves as a plain dict for forward compatibility but provides typed
    property accessors for the most common fields.

    Keys:
        parameters (dict): Keyword arguments specific to the skill.
        objective (ResearchObjective | None): Current research objective.
        stage (str): Current pipeline stage name.
        attempt (int): Current retry attempt (0-based).
    """

    @property
    def parameters(self) -> dict[str, Any]:
        """Skill-specific parameters."""
        return self.get("parameters", {})

    @property
    def objective(self) -> ResearchObjective | None:
        """The current research objective, if available."""
        return self.get("objective")

    @property
    def stage(self) -> str:
        """Current pipeline stage name."""
        return self.get("stage", "unknown")

    @property
    def attempt(self) -> int:
        """Current retry attempt index."""
        return self.get("attempt", 0)
