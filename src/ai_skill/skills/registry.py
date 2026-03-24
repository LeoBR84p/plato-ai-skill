"""SkillRegistry: registration, discovery, and dependency resolution for skills.

Skills are discovered from three sources (in priority order):
1. Built-in skills (auto-registered via register_builtin_skills()).
2. Entry points declared in installed packages:
       [project.entry-points."ai_skill.skills"]
       my_skill = "my_package.skills.my_skill:MySkillClass"
3. Local plugin directories listed in AI_SKILL_PLUGIN_DIRS (colon-separated paths).
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

from ai_skill.skills.base import BaseSkill, SkillMeta

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "ai_skill.skills"


class SkillRegistryError(Exception):
    """Raised for skill registration or resolution errors."""


class SkillRegistry:
    """Central registry for all skills available to the research pipeline.

    Provides discovery, registration, and retrieval of skill instances.
    All registered skills must implement the BaseSkill contract.

    Example:
        >>> registry = SkillRegistry()
        >>> registry.register(WebSearchSkill)
        >>> skill = registry.get("web_search")
        >>> output = skill.run(SkillInput(parameters={"query": "test"}))
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill_class: type[BaseSkill]) -> None:
        """Register a skill class.

        Args:
            skill_class: A class that inherits from BaseSkill and defines SKILL_META.

        Raises:
            SkillRegistryError: If the class does not define SKILL_META or if
                a skill with the same name is already registered.
        """
        if not hasattr(skill_class, "SKILL_META"):
            raise SkillRegistryError(
                f"{skill_class.__name__} must define a SKILL_META class attribute."
            )

        meta: SkillMeta = skill_class.SKILL_META
        if meta.name in self._skills:
            raise SkillRegistryError(
                f"Skill '{meta.name}' is already registered. "
                "Use a unique name or deregister the existing skill first."
            )

        self._skills[meta.name] = skill_class()
        logger.debug("Registered skill: %s v%s", meta.name, meta.version)

    def get(self, name: str) -> BaseSkill:
        """Return the skill instance for the given name.

        Args:
            name: The skill's unique name (e.g. "web_search").

        Returns:
            The registered BaseSkill instance.

        Raises:
            SkillRegistryError: If no skill with this name is registered.
        """
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self._skills.keys()))
            raise SkillRegistryError(
                f"Skill '{name}' is not registered. Available: {available}"
            )
        return skill

    def all(self) -> list[SkillMeta]:
        """Return metadata for all registered skills.

        Used by the PlannerAgent to inform the LLM of available tools.

        Returns:
            List of SkillMeta objects sorted by skill name.
        """
        return [skill.SKILL_META for skill in sorted(
            self._skills.values(), key=lambda s: s.SKILL_META.name
        )]

    def all_as_dicts(self) -> list[dict[str, Any]]:
        """Return all skill metadata as plain dicts for JSON serialisation.

        Returns:
            List of dicts suitable for inclusion in planning prompts.
        """
        return [meta.to_dict() for meta in self.all()]

    def names(self) -> list[str]:
        """Return sorted list of all registered skill names.

        Returns:
            Sorted list of name strings.
        """
        return sorted(self._skills.keys())

    def auto_discover(self) -> None:
        """Discover and register skills from all configured sources.

        Scans:
        - Built-in skills package (ai_skill.skills.*)
        - Entry points under the "ai_skill.skills" group
        - Directories in AI_SKILL_PLUGIN_DIRS environment variable
        """
        self._discover_builtins()
        self._discover_entry_points()
        self._discover_plugin_dirs()
        logger.info("SkillRegistry: %d skills registered.", len(self._skills))

    def _discover_builtins(self) -> None:
        """Register all built-in skills from the skills package."""
        builtin_names = [
            "web_search",
            "article_search",
            "pdf_reader",
            "content_summarizer",
            "google_drive",
            "exa_search",
            "firecrawl_scraper",
            "tavily_search",
        ]
        skills_pkg_path = Path(__file__).parent

        for name in builtin_names:
            skill_module_path = skills_pkg_path / name / "skill.py"
            if not skill_module_path.exists():
                logger.debug("Built-in skill '%s' not yet implemented — skipping.", name)
                continue

            try:
                module = importlib.import_module(f"ai_skill.skills.{name}.skill")
                skill_class: type[BaseSkill] | None = getattr(module, "SKILL_CLASS", None)
                if skill_class is None:
                    logger.warning(
                        "skill.py in '%s' does not define SKILL_CLASS — skipping.", name
                    )
                    continue
                self.register(skill_class)
            except SkillRegistryError:
                pass  # already registered (idempotent on re-discovery)
            except Exception as exc:
                logger.error("Failed to load built-in skill '%s': %s", name, exc)

    def _discover_entry_points(self) -> None:
        """Register skills declared via Python entry points."""
        try:
            eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
        except Exception as exc:
            logger.warning("Entry point discovery failed: %s", exc)
            return

        for ep in eps:
            try:
                skill_class: type[BaseSkill] = ep.load()
                self.register(skill_class)
                logger.debug("Loaded plugin skill via entry point: %s", ep.name)
            except SkillRegistryError:
                pass
            except Exception as exc:
                logger.error(
                    "Failed to load skill from entry point '%s': %s", ep.name, exc
                )

    def _discover_plugin_dirs(self) -> None:
        """Register skills found in directories listed in AI_SKILL_PLUGIN_DIRS."""
        plugin_dirs_env = os.environ.get("AI_SKILL_PLUGIN_DIRS", "")
        if not plugin_dirs_env.strip():
            return

        for dir_str in plugin_dirs_env.split(":"):
            plugin_dir = Path(dir_str.strip())
            if not plugin_dir.is_dir():
                logger.warning("AI_SKILL_PLUGIN_DIRS: '%s' is not a directory.", plugin_dir)
                continue

            for skill_dir in plugin_dir.iterdir():
                if not skill_dir.is_dir():
                    continue
                skill_module_path = skill_dir / "skill.py"
                if not skill_module_path.exists():
                    continue

                try:
                    spec = importlib.util.spec_from_file_location(
                        f"ai_skill_plugin.{skill_dir.name}", skill_module_path
                    )
                    if spec is None or spec.loader is None:
                        continue
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[f"ai_skill_plugin.{skill_dir.name}"] = module
                    spec.loader.exec_module(module)  # type: ignore[union-attr]

                    skill_class = getattr(module, "SKILL_CLASS", None)
                    if skill_class is None:
                        logger.warning(
                            "Plugin '%s' has no SKILL_CLASS — skipping.", skill_dir.name
                        )
                        continue
                    self.register(skill_class)
                    logger.debug("Loaded plugin skill from directory: %s", skill_dir.name)
                except SkillRegistryError:
                    pass
                except Exception as exc:
                    logger.error(
                        "Failed to load plugin from '%s': %s", skill_dir, exc
                    )
