"""Unit tests for CP3 (Research Design) nodes and routing functions.

No LLM calls — LLM-dependent nodes use a mock client.
No file I/O — docx helpers are not tested here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ai_skill.core.pipeline_stages import PipelineStage, ResearchStatus
from ai_skill.core.state import (
    EvaluationResult,
    ResearchDesignDoc,
    ResearchDesignSection,
    ResearchState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_state(**overrides: object) -> ResearchState:
    """Return a minimal ResearchState for testing."""
    state: ResearchState = ResearchState(
        stage=PipelineStage.RESEARCH_DESIGN,
        status=ResearchStatus.PLANNING,
        attempt=0,
        checkpoint_pending=False,
        charter_approved=True,
        literature_approved=True,
        design_approved=False,
        active_checkpoint=3,
        charter_document_text="",
        user_feedback=None,
        user_guidance=None,
        document_version=0,
        workspace_path="/tmp/test-ws",
        findings=[],
        findings_current=[],
        quality_history=[],
        stage_quality_history={},
        messages=[],
        objective={
            "topic": "test topic",
            "goals": ["goal 1"],
            "success_metrics": ["metric 1"],
            "scope_constraints": [],
            "bibliography_style": "abnt",
            "language": "pt-BR",
            "generated_at": "",
            "stage_guidelines": {},
        },
    )
    state.update(overrides)  # type: ignore[arg-type]
    return state


def _converged_eval() -> EvaluationResult:
    return EvaluationResult(
        per_metric=[],
        total_score=0.9,
        converged=True,
        gaps=[],
        regression=False,
    )


def _non_converged_eval() -> EvaluationResult:
    return EvaluationResult(
        per_metric=[],
        total_score=0.5,
        converged=False,
        gaps=["gap 1"],
        regression=False,
    )


# ---------------------------------------------------------------------------
# route_after_evaluate_cp3
# ---------------------------------------------------------------------------

def test_route_after_evaluate_cp3_converged() -> None:
    """Converged evaluation → compile_design."""
    from ai_skill.core.nodes import route_after_evaluate_cp3

    state = _base_state(evaluation=_converged_eval(), attempt=1)
    assert route_after_evaluate_cp3(state) == "compile_design"


def test_route_after_evaluate_cp3_retry() -> None:
    """Non-converged, attempt < 5 → plan."""
    from ai_skill.core.nodes import route_after_evaluate_cp3

    state = _base_state(evaluation=_non_converged_eval(), attempt=2)
    assert route_after_evaluate_cp3(state) == "plan"


def test_route_after_evaluate_cp3_support() -> None:
    """Non-converged, attempt == 5 → request_support."""
    from ai_skill.core.nodes import route_after_evaluate_cp3

    state = _base_state(evaluation=_non_converged_eval(), attempt=5)
    assert route_after_evaluate_cp3(state) == "request_support"


# ---------------------------------------------------------------------------
# route_cp3_start
# ---------------------------------------------------------------------------

def test_route_cp3_start_fresh() -> None:
    """No design doc and no feedback → plan (fresh start)."""
    from ai_skill.core.nodes import route_cp3_start

    state = _base_state()
    assert route_cp3_start(state) == "plan"


def test_route_cp3_start_correction() -> None:
    """Existing design doc + user feedback → refine_design."""
    from ai_skill.core.nodes import route_cp3_start

    design_doc: ResearchDesignDoc = ResearchDesignDoc(
        sections=[ResearchDesignSection(section_title="Método", content="content")],
        study_type="observacional",
        research_paradigm="quantitativo",
        epistemological_stance="pós-positivista",
        hypotheses=["H1: ..."],
        research_questions=[],
        variables=[],
        instruments=[],
        sampling_strategy="",
        sample_size_justification="",
        ethical_considerations="",
        validity_threats=[],
        mitigation_strategies=[],
        data_management_plan="",
        target_journal_tier="Q1",
        reporting_standard="STROBE",
        metrics_and_kpis=[],
        data_sources=[],
        collection_protocol="",
        methodology_timeline="",
    )
    state = _base_state(research_design_doc=design_doc, user_feedback="fix hypothesis")
    assert route_cp3_start(state) == "refine_design"


def test_route_cp3_start_doc_without_feedback() -> None:
    """Existing design doc but no feedback → plan (no correction to apply)."""
    from ai_skill.core.nodes import route_cp3_start

    design_doc: ResearchDesignDoc = ResearchDesignDoc(
        sections=[],
        study_type="observacional",
        research_paradigm="quantitativo",
        epistemological_stance="pós-positivista",
        hypotheses=[],
        research_questions=[],
        variables=[],
        instruments=[],
        sampling_strategy="",
        sample_size_justification="",
        ethical_considerations="",
        validity_threats=[],
        mitigation_strategies=[],
        data_management_plan="",
        target_journal_tier="Q1",
        reporting_standard="STROBE",
        metrics_and_kpis=[],
        data_sources=[],
        collection_protocol="",
        methodology_timeline="",
    )
    state = _base_state(research_design_doc=design_doc)
    assert route_cp3_start(state) == "plan"


# ---------------------------------------------------------------------------
# route_after_review_design
# ---------------------------------------------------------------------------

def test_route_after_review_design_approved() -> None:
    """design_approved=True → END."""
    from ai_skill.core.nodes import route_after_review_design

    state = _base_state(design_approved=True)
    assert route_after_review_design(state) == "END"


def test_route_after_review_design_rejected() -> None:
    """design_approved=False → refine_design."""
    from ai_skill.core.nodes import route_after_review_design

    state = _base_state(design_approved=False)
    assert route_after_review_design(state) == "refine_design"


# ---------------------------------------------------------------------------
# compile_design (mock LLM)
# ---------------------------------------------------------------------------

def test_compile_design_returns_design_doc() -> None:
    """compile_design populates research_design_doc and sets active_checkpoint=3."""
    from ai_skill.core.nodes import compile_design

    # Simulate a Pydantic _ResearchDesignDocLLM instance with attribute access
    mock_section = MagicMock()
    mock_section.section_title = "Método de Pesquisa"
    mock_section.content = "body"

    mock_llm_response = MagicMock()
    mock_llm_response.sections = [mock_section]
    mock_llm_response.study_type = "observacional"
    mock_llm_response.research_paradigm = "quantitativo"
    mock_llm_response.epistemological_stance = "pós-positivista"
    mock_llm_response.hypotheses = ["H1: teste"]
    mock_llm_response.research_questions = []
    mock_llm_response.variables = []
    mock_llm_response.instruments = []
    mock_llm_response.sampling_strategy = "amostra aleatória"
    mock_llm_response.sample_size_justification = "n=100"
    mock_llm_response.ethical_considerations = "nenhum"
    mock_llm_response.validity_threats = []
    mock_llm_response.mitigation_strategies = []
    mock_llm_response.data_management_plan = "FAIR"
    mock_llm_response.metrics_and_kpis = ["KPI1"]
    mock_llm_response.data_sources = ["fonte1"]
    mock_llm_response.collection_protocol = "passo 1"
    mock_llm_response.methodology_timeline = "CP3→CP4→CP5"
    mock_llm_response.reporting_standard = "STROBE"
    mock_llm_response.target_journal_tier = "Q1"

    mock_llm = MagicMock()
    mock_llm.complete_structured.return_value = mock_llm_response

    state = _base_state(
        cp3_context={"topic": "test", "goals": [], "scope_constraints": [], "methodology_preference": ""},
        charter_document_text="charter text",
    )
    state["findings"] = []

    result = compile_design(state, llm=mock_llm)

    assert "research_design_doc" in result
    doc = result["research_design_doc"]
    assert doc["study_type"] == "observacional"
    assert doc["hypotheses"] == ["H1: teste"]
    assert result.get("active_checkpoint") == 3


# ---------------------------------------------------------------------------
# refine_design (mock LLM)
# ---------------------------------------------------------------------------

def test_refine_design_clears_feedback() -> None:
    """refine_design applies corrections and resets user_feedback to None."""
    from ai_skill.core.nodes import refine_design

    original_doc: ResearchDesignDoc = ResearchDesignDoc(
        sections=[ResearchDesignSection(section_title="Método", content="original body")],
        study_type="observacional",
        research_paradigm="quantitativo",
        epistemological_stance="pós-positivista",
        hypotheses=["H1: original"],
        research_questions=[],
        variables=[],
        instruments=[],
        sampling_strategy="",
        sample_size_justification="",
        ethical_considerations="",
        validity_threats=[],
        mitigation_strategies=[],
        data_management_plan="",
        target_journal_tier="Q1",
        reporting_standard="STROBE",
        metrics_and_kpis=[],
        data_sources=[],
        collection_protocol="",
        methodology_timeline="",
    )

    mock_section = MagicMock()
    mock_section.section_title = "Método"
    mock_section.content = "original body"

    mock_response = MagicMock()
    mock_response.sections = [mock_section]
    mock_response.study_type = "observacional"
    mock_response.research_paradigm = "quantitativo"
    mock_response.epistemological_stance = "pós-positivista"
    mock_response.hypotheses = ["H1: refined"]
    mock_response.research_questions = []
    mock_response.variables = []
    mock_response.instruments = []
    mock_response.sampling_strategy = ""
    mock_response.sample_size_justification = ""
    mock_response.ethical_considerations = ""
    mock_response.validity_threats = []
    mock_response.mitigation_strategies = []
    mock_response.data_management_plan = ""
    mock_response.metrics_and_kpis = []
    mock_response.data_sources = []
    mock_response.collection_protocol = ""
    mock_response.methodology_timeline = ""
    mock_response.reporting_standard = "STROBE"
    mock_response.target_journal_tier = "Q1"

    mock_llm = MagicMock()
    mock_llm.complete_structured.return_value = mock_response

    state = _base_state(
        research_design_doc=original_doc,
        user_feedback="update hypothesis H1",
    )

    result = refine_design(state, llm=mock_llm)

    assert result.get("user_feedback") is None
    assert "research_design_doc" in result
