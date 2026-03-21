"""Contract tests: verify every built-in skill honours the BaseSkill contract.

Rules verified:
- skill.run() never raises an exception
- SkillOutput.error is None iff confidence > 0.0
- skill.SKILL_META.name matches the registry key
- arun() returns the same type as run()
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_skill.skills.base import BaseSkill, SkillInput, SkillOutput
from ai_skill.skills.registry import SkillRegistry


def _all_skills() -> list[tuple[str, BaseSkill]]:
    """Return (name, instance) pairs for every registered built-in skill."""
    registry = SkillRegistry()
    registry.auto_discover()
    return [(name, registry.get(name)) for name in registry.names()]


_MINIMAL_INPUT: dict[str, Any] = {}


@pytest.mark.parametrize("name,skill", _all_skills())
def test_run_never_raises(name: str, skill: BaseSkill) -> None:
    """skill.run() must return SkillOutput, never raise."""
    inp = SkillInput(parameters=_MINIMAL_INPUT)
    result = skill.run(inp)
    assert isinstance(result, dict), f"{name}.run() must return a dict (SkillOutput)"
    assert "skill_name" in result


@pytest.mark.parametrize("name,skill", _all_skills())
def test_error_none_implies_positive_confidence(name: str, skill: BaseSkill) -> None:
    """If error is None, confidence must be > 0."""
    inp = SkillInput(parameters=_MINIMAL_INPUT)
    result = skill.run(inp)
    if result.get("error") is None:
        assert result.get("confidence", 0.0) > 0.0, (
            f"{name}: error=None but confidence={result.get('confidence')}"
        )


@pytest.mark.parametrize("name,skill", _all_skills())
def test_meta_name_matches_registry(name: str, skill: BaseSkill) -> None:
    """SKILL_META.name must equal the registry key."""
    assert skill.SKILL_META.name == name


@pytest.mark.parametrize("name,skill", _all_skills())
def test_arun_returns_dict(name: str, skill: BaseSkill) -> None:  # noqa: ARG001
    """arun() must return the same dict structure as run()."""
    inp = SkillInput(parameters=_MINIMAL_INPUT)
    result = asyncio.run(skill.arun(inp))
    assert isinstance(result, dict)
    assert "skill_name" in result
