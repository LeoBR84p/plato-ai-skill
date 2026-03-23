"""Central state definition for the LangGraph research pipeline.

The ResearchState TypedDict is the single source of truth for all data
flowing through the graph. Every node receives and returns a (partial)
ResearchState, following the LangGraph state reducer pattern.
"""

from __future__ import annotations

from typing import Annotated, Any
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ai_skill.core.pipeline_stages import PipelineStage, ResearchStatus


class SourceVerification(TypedDict, total=False):
    """Verification result for a single bibliographic source.

    Attributes:
        reference_number: The [N] index of this source in the review.
        url: The URL that was checked.
        title: Title extracted from the reference entry.
        accessible: True if the URL returned a 2xx response.
        content_matches: True when an independent LLM agent confirmed the
            fetched content is consistent with what the review claims.
        verification_note: Short explanation of the verification outcome.
        access_date: Date of access in ABNT format (e.g. "23 mar. 2026").
        access_method: How content was retrieved — "HTTP" for direct HTML fetch,
            "Semantic_Scholar_API" when the S2 Graph API was used,
            "PDF_direto" for a direct PDF URL, or "PDF_fallback" when a
            PDF/download link was followed from the landing page.
            "PDF_direto" and "PDF_fallback" trigger an "Acesso direto (sem API)"
            badge in the docx; "HTTP" and "Semantic_Scholar_API" do not.
    """

    reference_number: int
    url: str
    title: str
    accessible: bool
    content_matches: bool
    verification_note: str
    access_date: str
    access_method: str


class LiteratureReviewSection(TypedDict):
    """One thematic section of the literature review.

    Attributes:
        section_title: Heading for the section.
        content: Markdown body with inline [N] citations.
    """

    section_title: str
    content: str


class ResearchDesignSection(TypedDict):
    """One section of the research design document.

    Attributes:
        section_title: Heading for the section.
        content: Body text (markdown prose).
    """

    section_title: str
    content: str


class ResearchDesignDoc(TypedDict, total=False):
    """Structured research design document produced by CP3.

    Attributes:
        sections: Ordered list of sections covering paradigm, model, PICOS,
            instruments, sampling, FAIR, ethics, risks, and stakeholders.
        study_type: Research design classification
            (experimental | quasi-experimental | observational | case-study |
            action-research | systematic-review | mixed-methods).
        research_paradigm: Epistemological paradigm
            (quantitative | qualitative | mixed).
        epistemological_stance: Worldview
            (post-positivist | constructivist | pragmatic | transformative).
        hypotheses: Testable hypothesis statements (OR Step 2 format).
        research_questions: Alternative/complementary research questions.
        variables: List of dicts with keys: name, type, description,
            operationalization, measurement_scale.
        instruments: Data collection instruments with validation references.
        sampling_strategy: Description of sampling approach and frame.
        sample_size_justification: Power analysis or saturation rationale.
        ethical_considerations: Consent, privacy, CEP/IRB requirements.
        validity_threats: Internal and external validity threats identified.
        mitigation_strategies: List of dicts with keys: threat, mitigation,
            contingency.
        data_management_plan: FAIR-compliant data management summary.
        target_journal_tier: Target publication venue quartile (Q1/Q2/Q3/Q4).
        reporting_standard: Applicable reporting guideline
            (CONSORT | STROBE | PRISMA | SQUIRE | outro | não aplicável).
        metrics_and_kpis: KPIs, acceptance thresholds, statistical power, and
            minimum sample size with justification (ISO 9001:2015).
        data_sources: Primary and secondary data sources with quality criteria
            (credibility, recency, accessibility) and FAIR compliance statement.
        collection_protocol: Step-by-step data collection protocol with
            instruments, tools, and acceptance criteria for each step.
        methodology_timeline: PMBOK-aligned milestone schedule linking
            CP3 (design) → CP4 (collection) → CP5 (analysis) with time estimates.
    """

    sections: list[ResearchDesignSection]
    study_type: str
    research_paradigm: str
    epistemological_stance: str
    hypotheses: list[str]
    research_questions: list[str]
    variables: list[dict[str, Any]]
    instruments: list[str]
    sampling_strategy: str
    sample_size_justification: str
    ethical_considerations: str
    validity_threats: list[str]
    mitigation_strategies: list[dict[str, Any]]
    data_management_plan: str
    target_journal_tier: str
    reporting_standard: str
    metrics_and_kpis: list[str]
    data_sources: list[str]
    collection_protocol: str
    methodology_timeline: str


class LiteratureReviewDoc(TypedDict, total=False):
    """Structured literature review document produced by CP2.

    Attributes:
        sections: Ordered list of review sections with inline citations.
        references: ABNT-formatted reference entries, one per source.
            Each item is a dict with keys: reference_number, title, authors,
            year, url, abnt_entry, summary.
        verified_sources: Verification results per source URL.
    """

    sections: list[LiteratureReviewSection]
    references: list[dict[str, Any]]
    verified_sources: list[SourceVerification]


class ResearchObjective(TypedDict, total=False):
    """The research objective as defined and confirmed by the user.

    Attributes:
        topic: Free-form description of the research topic.
        goals: List of specific research goals to achieve.
        success_metrics: List of measurable criteria for the OVERALL project success.
        scope_constraints: Optional limitations on scope (time, geography, etc.).
        methodology_preference: Preferred research methodology if known.
        bibliography_style: Citation style (default: "abnt").
        language: Primary language of the paper (default: "pt-BR").
        generated_at: Date of charter generation in dd/mm/yyyy format.
        stage_guidelines: Stage-specific directives keyed by PipelineStage value
            (e.g. "literature_review", "research_design", ...).  Each entry is a
            list of 4-8 actionable items that drive planning and evaluation for
            that stage, independently of the overall success_metrics.
    """

    topic: str
    goals: list[str]
    success_metrics: list[str]
    scope_constraints: list[str]
    methodology_preference: str
    bibliography_style: str
    language: str
    generated_at: str
    stage_guidelines: dict[str, list[str]]


class PlanStep(TypedDict):
    """A single step within an execution plan.

    Attributes:
        step_id: Zero-based index of this step within the plan.
        skill_name: Name of the skill to execute (must be registered).
        parameters: Keyword arguments to pass to skill.run().
        depends_on: Indices of steps that must complete before this one.
        rationale: Why this step is needed for the current objective.
    """

    step_id: int
    skill_name: str
    parameters: dict[str, Any]
    depends_on: list[int]
    rationale: str


class ExecutionPlan(TypedDict):
    """A complete execution plan produced by the PlannerAgent.

    Attributes:
        steps: Ordered list of plan steps (dependency-respecting).
        rationale: Overall rationale for this plan.
        estimated_cost: Rough estimate of API calls / tokens needed.
        attempt: Which retry attempt this plan belongs to (0-based).
    """

    steps: list[PlanStep]
    rationale: str
    estimated_cost: str
    attempt: int


class SkillOutput(TypedDict, total=False):
    """Standardized output from any skill execution.

    Skills NEVER raise exceptions. All errors are captured in the error field.
    A confidence of 0.0 always accompanies a non-None error.

    Attributes:
        skill_name: Name of the skill that produced this output.
        result: The skill's payload (structure varies by skill).
        confidence: Quality/reliability score in [0.0, 1.0].
        sources: URLs or identifiers of sources consulted.
        error: Error message if the skill failed, else None.
        cached: Whether this output was served from ResearchMemory cache.
    """

    skill_name: str
    result: dict[str, Any]
    confidence: float
    sources: list[str]
    error: str | None
    cached: bool


class MetricScore(TypedDict):
    """Score for a single success metric.

    Attributes:
        metric: The success metric text from ResearchObjective.
        score: Value in [0.0, 1.0].
        rationale: Why this score was assigned.
        gaps: Specific shortcomings that prevented a higher score.
    """

    metric: str
    score: float
    rationale: str
    gaps: list[str]


class EvaluationResult(TypedDict):
    """Result of evaluating skill outputs against success metrics.

    Attributes:
        per_metric: Score for each individual success metric.
        total_score: Weighted average of per_metric scores.
        converged: True when total_score >= AI_SKILL_CONVERGENCE_THRESHOLD.
        gaps: Aggregated list of gaps across all metrics.
        regression: True if total_score dropped > 0.05 from previous attempt.
    """

    per_metric: list[MetricScore]
    total_score: float
    converged: bool
    gaps: list[str]
    regression: bool


class QualitySnapshot(TypedDict):
    """Quality metrics snapshot recorded after each inner loop attempt.

    Attributes:
        attempt: Attempt index (0-based).
        stage: Pipeline stage this snapshot belongs to.
        total_score: Overall quality score for this attempt.
        per_metric_scores: Score per success metric.
        skills_used: Names of skills that ran in this attempt.
        cache_hit_rate: Fraction of skill calls served from cache.
    """

    attempt: int
    stage: str
    total_score: float
    per_metric_scores: dict[str, float]
    skills_used: list[str]
    cache_hit_rate: float


class ResearchState(TypedDict, total=False):
    """Central state for the LangGraph research pipeline.

    All fields are optional (total=False) to allow partial updates from nodes.
    LangGraph merges partial state dicts returned by each node.

    The `messages` field uses the add_messages reducer so that message history
    is accumulated rather than replaced on each node return.
    """

    # Core research context
    objective: ResearchObjective
    stage: PipelineStage
    status: ResearchStatus

    # Planning and execution
    plan: ExecutionPlan
    findings: list[SkillOutput]          # all outputs accumulated across ALL attempts
    findings_current: list[SkillOutput]  # outputs of the CURRENT attempt only (used by evaluate)

    # Evaluation and quality
    evaluation: EvaluationResult
    attempt: int
    quality_history: list[QualitySnapshot]  # continuous across all stages
    stage_quality_history: dict[str, list[QualitySnapshot]]  # per-stage isolation

    # Checkpoints and user interaction
    checkpoint_pending: bool
    user_feedback: str | None
    user_guidance: str | None  # guidance provided interactively when convergence fails
    checkpoint_label: str
    charter_approved: bool

    # Literature review (Checkpoint 2)
    literature_approved: bool
    active_checkpoint: int
    literature_review_doc: LiteratureReviewDoc
    charter_document_text: str  # extracted text of CP1 [final].docx — CP2 context

    # Research design (Checkpoint 3)
    research_design_doc: ResearchDesignDoc
    design_approved: bool

    # CP3 handoff — populated by begin_design() before CP3 graph starts.
    # Contains topic, goals, scope_constraints, methodology_preference, literature_summary.
    cp3_context: dict

    # CP2 handoff — populated by begin_literature_review() before CP2 graph starts.
    # Contains ONLY the fields CP2 nodes are allowed to see: topic, goals,
    # scope_constraints.  success_metrics and all other CP1-only fields are absent.
    # When present, plan() and evaluate() use this instead of the full objective.
    cp2_context: dict

    # Document versioning
    document_version: int
    workspace_path: str

    # LangGraph message history (uses add_messages reducer)
    messages: Annotated[list[BaseMessage], add_messages]


def initial_state(workspace_path: str, topic: str = "") -> ResearchState:
    """Return an empty ResearchState for a new research project.

    Args:
        workspace_path: Absolute path to the research workspace directory.
        topic: Optional research topic to pre-populate the objective.

    Returns:
        A ResearchState with default values for a fresh project.
    """
    objective: ResearchObjective = ResearchObjective(
        topic=topic,
        goals=[],
        success_metrics=[],
        scope_constraints=[],
        bibliography_style="abnt",
        language="pt-BR",
        generated_at="",
        stage_guidelines={},
    )
    return ResearchState(
        objective=objective,
        stage=PipelineStage.INITIATE,
        status=ResearchStatus.PLANNING,
        attempt=0,
        checkpoint_pending=False,
        charter_approved=False,
        literature_approved=False,
        design_approved=False,
        active_checkpoint=1,
        charter_document_text="",
        user_feedback=None,
        user_guidance=None,
        document_version=0,
        workspace_path=str(workspace_path),
        findings=[],
        findings_current=[],
        quality_history=[],
        stage_quality_history={},
        messages=[],
    )
