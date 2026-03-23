"""Unit tests for prompts/design.py — no LLM calls, no I/O."""

from __future__ import annotations

import json


def test_build_compile_design_messages_returns_tuple() -> None:
    """build_compile_design_messages returns (system_str, list_of_dicts)."""
    from ai_skill.prompts.design import build_compile_design_messages

    system, messages = build_compile_design_messages(
        charter_document_text="charter text",
        literature_document_text="lit review text",
        cp3_context={"topic": "test", "goals": []},
        findings=[],
    )

    assert isinstance(system, str)
    assert len(system) > 0
    assert isinstance(messages, list)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], str)


def test_build_compile_design_messages_contains_charter() -> None:
    """User message includes the charter text."""
    from ai_skill.prompts.design import build_compile_design_messages

    _, messages = build_compile_design_messages(
        charter_document_text="UNIQUE_CHARTER_MARKER",
        literature_document_text="lit",
        cp3_context={},
        findings=[],
    )

    assert "UNIQUE_CHARTER_MARKER" in messages[0]["content"]


def test_build_compile_design_messages_contains_literature() -> None:
    """User message includes the literature review text."""
    from ai_skill.prompts.design import build_compile_design_messages

    _, messages = build_compile_design_messages(
        charter_document_text="charter",
        literature_document_text="UNIQUE_LIT_MARKER",
        cp3_context={},
        findings=[],
    )

    assert "UNIQUE_LIT_MARKER" in messages[0]["content"]


def test_build_compile_design_messages_serialises_findings() -> None:
    """Findings list is serialised as JSON in the user message."""
    from ai_skill.prompts.design import build_compile_design_messages

    findings = [{"skill_name": "web_search", "result": {"text": "found"}}]
    _, messages = build_compile_design_messages(
        charter_document_text="charter",
        literature_document_text="lit",
        cp3_context={},
        findings=findings,
    )

    assert "web_search" in messages[0]["content"]


def test_build_compile_design_messages_fallback_when_empty() -> None:
    """Empty charter/literature produce fallback placeholder strings."""
    from ai_skill.prompts.design import build_compile_design_messages

    _, messages = build_compile_design_messages(
        charter_document_text="",
        literature_document_text="",
        cp3_context={},
        findings=[],
    )

    content = messages[0]["content"]
    assert "não disponível" in content


def test_build_refine_design_messages_contains_feedback() -> None:
    """build_refine_design_messages embeds the feedback string."""
    from ai_skill.prompts.design import build_refine_design_messages

    design_doc = {
        "sections": [],
        "hypotheses": ["H1: original"],
        "study_type": "observacional",
    }
    feedback = "UNIQUE_FEEDBACK_MARKER"

    system, messages = build_refine_design_messages(
        design_doc=design_doc,
        feedback=feedback,
    )

    assert isinstance(system, str)
    assert feedback in messages[0]["content"]


def test_build_refine_design_messages_contains_design_json() -> None:
    """build_refine_design_messages serialises the design doc as JSON."""
    from ai_skill.prompts.design import build_refine_design_messages

    design_doc = {"study_type": "UNIQUE_TYPE_MARKER", "sections": []}
    _, messages = build_refine_design_messages(design_doc=design_doc, feedback="fix it")

    assert "UNIQUE_TYPE_MARKER" in messages[0]["content"]
