"""Smoke tests: verify the graph compiles and all skills register correctly.

These tests have no external dependencies and must complete in < 5 seconds.
No LLM calls, no HTTP requests.
"""

from __future__ import annotations

import pytest


def test_graph_compiles_without_error() -> None:
    """The StateGraph compiles without configuration errors."""
    from ai_skill.core.graph import build_research_graph

    graph = build_research_graph()
    assert graph is not None


def test_graph_mermaid_contains_start() -> None:
    """The compiled graph produces a valid Mermaid diagram."""
    from ai_skill.core.graph import get_graph_mermaid

    mermaid = get_graph_mermaid()
    assert "__start__" in mermaid.lower() or "start" in mermaid.lower()


def test_all_builtin_skills_register() -> None:
    """All 4 Phase 1 built-in skills register without errors."""
    from ai_skill.skills.registry import SkillRegistry

    registry = SkillRegistry()
    registry.auto_discover()

    names = registry.names()
    expected = {"web_search", "article_search", "pdf_reader", "content_summarizer"}
    missing = expected - set(names)
    assert not missing, f"Skills missing from registry: {missing}"


def test_skill_registry_get() -> None:
    """Registry.get() returns the correct skill class."""
    from ai_skill.skills.registry import SkillRegistry

    registry = SkillRegistry()
    registry.auto_discover()

    skill = registry.get("web_search")
    assert skill is not None
    assert skill.SKILL_META.name == "web_search"


def test_initial_state_factory() -> None:
    """initial_state() returns a valid ResearchState dict."""
    from pathlib import Path

    from ai_skill.core.state import initial_state

    state = initial_state(Path("/tmp/test-workspace"))
    assert state["attempt"] == 0
    assert state["findings"] == []
    assert state["status"] == "planning"


def test_pipeline_stages_enum() -> None:
    """PipelineStage and ResearchStatus enums are importable and have expected values."""
    from ai_skill.core.pipeline_stages import PipelineStage, ResearchStatus

    assert PipelineStage.INITIATE is not None
    assert ResearchStatus.PLANNING is not None


def test_url_safety_guard_instantiates() -> None:
    """UrlSafetyGuard can be instantiated without a GOOGLE_API_KEY."""
    import os

    os.environ.pop("GOOGLE_API_KEY", None)
    from ai_skill.core.url_safety import UrlSafetyGuard

    guard = UrlSafetyGuard()
    # Fail-open: returns True when API key is absent
    assert guard.is_safe("https://example.com") is True


def test_workspace_initialise(tmp_path: "pytest.TempPathFactory") -> None:  # type: ignore[type-arg]
    """ResearchWorkspace creates directories and stub files."""
    from ai_skill.core.workspace import ResearchWorkspace

    ws = ResearchWorkspace(tmp_path / "test-ws")
    ws.initialise("Test Topic")

    assert (tmp_path / "test-ws" / "research-log.md").exists()
    assert (tmp_path / "test-ws" / "findings.md").exists()
    assert (tmp_path / "test-ws" / ".cache").is_dir()


def test_cp3_graph_compiles_without_error() -> None:
    """The CP3 StateGraph compiles without configuration errors."""
    from ai_skill.core.graph import build_cp3_graph

    graph = build_cp3_graph()
    assert graph is not None


def test_cp3_graph_mermaid_contains_compile_design() -> None:
    """The compiled CP3 graph produces a Mermaid diagram with compile_design node."""
    from ai_skill.core.graph import build_cp3_graph

    mermaid = build_cp3_graph().get_graph().draw_mermaid()
    assert "compile_design" in mermaid
