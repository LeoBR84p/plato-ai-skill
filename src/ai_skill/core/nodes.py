"""LangGraph node functions for the Phase 1 research pipeline.

Each function takes a ResearchState and returns a partial ResearchState dict.
LangGraph merges partial updates into the full state automatically.

Nodes in Phase 1:
    initiate         — create workspace directory and initial state
    align_charter    — draft and refine the Research Charter with the user
    plan             — generate an ExecutionPlan using the LLM
    execute          — run the skills defined in the current plan
    evaluate         — score findings against success metrics
    deliver          — write findings.md and mark project as converged
    request_support  — display gaps to user and collect revised guidance
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_skill.core import history as _history
from ai_skill.core.llm_client import LLMClient, LLMClientError, set_current_node
from ai_skill.core.pipeline_stages import PipelineStage, ResearchStatus
from ai_skill.core.project_workspace import ProjectWorkspace
from ai_skill.core.state import (
    EvaluationResult,
    ExecutionPlan,
    LiteratureReviewDoc,
    LiteratureReviewSection,
    MetricScore,
    PlanStep,
    QualitySnapshot,
    ResearchObjective,
    ResearchState,
    SkillOutput,
    SourceVerification,
)
from ai_skill.core.workspace import ResearchWorkspace
from ai_skill.prompts.alignment import (
    build_charter_draft_messages,
    build_charter_refine_messages,
)
from ai_skill.prompts.evaluation import build_evaluation_messages
from ai_skill.prompts.literature import (
    build_compile_messages,
    build_refine_messages,
    build_verify_messages,
)
from ai_skill.prompts.planning import build_planning_messages
from ai_skill.skills.registry import SkillRegistry
from ai_skill.skills.base import SkillInput

logger = logging.getLogger(__name__)

_CONVERGENCE_THRESHOLD = float(
    os.environ.get("AI_SKILL_CONVERGENCE_THRESHOLD", "0.75")
)


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM outputs
# ---------------------------------------------------------------------------

class _ExecutionPlanLLM(BaseModel):
    """LLM-parseable representation of an ExecutionPlan."""

    steps: list[dict[str, Any]] = Field(
        description="List of plan steps with skill_name, parameters, depends_on, rationale."
    )
    rationale: str = Field(description="Overall rationale for this plan.")
    estimated_cost: str = Field(
        description="Rough estimate: 'low' (<10 API calls), 'medium' (10-50), 'high' (>50)."
    )


class _EvaluationResultLLM(BaseModel):
    """LLM-parseable representation of an EvaluationResult.

    Note: total_score and converged are NOT in this model — they are computed
    deterministically in Python from per_metric scores to eliminate LLM arithmetic
    errors and premature convergence declarations.
    """

    per_metric: list[dict[str, Any]] = Field(
        description="List of {metric, score, rationale, gaps} dicts."
    )
    gaps: list[str] = Field(description="Aggregated gaps across all metrics.")


class _CharterLLM(BaseModel):
    """LLM-parseable Research Charter."""

    topic: str
    goals: list[str] = Field(min_length=1)
    success_metrics: list[str] = Field(min_length=1)
    scope_constraints: list[str] = Field(default_factory=list)
    methodology_preference: str = Field(default="")
    bibliography_style: str = Field(default="abnt")
    language: str = Field(default="pt-BR")
    stage_guidelines: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Stage-specific directives keyed by PipelineStage value "
            "(literature_review, research_design, data_collection_guide, "
            "analysis_guide, results_interpretation, paper_composition, publication). "
            "Each entry is a list of 4-8 actionable items driving planning and "
            "evaluation for that stage independently of the overall success_metrics."
        ),
    )


class _LiteratureSourceLLM(BaseModel):
    """One bibliographic entry in the literature review."""

    reference_number: int = Field(description="Sequential citation index starting at 1.")
    title: str = Field(description="Full title of the work.")
    authors: str = Field(description="Author(s) in ABNT order (SOBRENOME, Nome).")
    year: str = Field(description="Publication year as a string.")
    url: str = Field(description="Direct URL to the original document or page.")
    abnt_entry: str = Field(
        description="Complete ABNT NBR 6023:2018 formatted reference entry "
                    "including the URL and the placeholder {ACCESS_DATE}."
    )
    summary: str = Field(
        description="One-sentence summary of what this source contributes to the review."
    )


class _LiteratureReviewSectionLLM(BaseModel):
    """One thematic section of the literature review."""

    section_title: str = Field(description="Heading for this section in pt-BR.")
    content: str = Field(
        description="Body text in pt-BR Markdown with inline [N] citations "
                    "referencing the bibliography."
    )


class _LiteratureReviewLLM(BaseModel):
    """Full structured literature review — enforces consistent output format."""

    sections: list[_LiteratureReviewSectionLLM] = Field(
        min_length=1,
        description="Ordered thematic sections that form the review body.",
    )
    references: list[_LiteratureSourceLLM] = Field(
        min_length=1,
        description="Numbered bibliography (must match all [N] used in sections).",
    )


class _VerifyResultLLM(BaseModel):
    """Independent verification result for a single source URL."""

    content_matches: bool = Field(
        description="True when the fetched content is consistent with the cited claim."
    )
    verification_note: str = Field(
        description="One sentence explaining the verification decision."
    )


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def initiate(state: ResearchState) -> dict[str, Any]:
    """Create the workspace directory and set initial pipeline state.

    Args:
        state: Current ResearchState (may be partially empty on first run).

    Returns:
        Partial state update with stage, status, and workspace_path.
    """
    workspace_dir = os.environ.get("AI_SKILL_WORKSPACE_DIR", "./research-workspace")
    workspace_path = state.get("workspace_path") or workspace_dir
    Path(workspace_path).mkdir(parents=True, exist_ok=True)

    logger.info("Research workspace: %s", workspace_path)

    return {
        "stage": PipelineStage.RESEARCH_CHARTER,
        "status": ResearchStatus.PLANNING,
        "workspace_path": str(Path(workspace_path).resolve()),
        "attempt": 0,
        "findings": [],
        "quality_history": [],
        "document_version": 0,
    }


def align_charter(state: ResearchState, llm: LLMClient | None = None) -> dict[str, Any]:
    """Draft or refine the Research Charter and save it as a .docx checkpoint.

    On the first call (no goals yet), drafts the charter from the topic.
    On subsequent calls (user provided feedback via review_charter), refines it.
    Always saves ``Checkpoint 1 - Research Charter.docx`` to the ProjectWorkspace.

    Args:
        state: Current ResearchState. ``user_feedback`` carries revision
            instructions from the previous review cycle.
        llm: Optional injected LLMClient (for testing).

    Returns:
        Partial state update with the updated ``objective`` and cleared
        ``user_feedback`` (consumed by this node).
    """
    _configure_history(state, "align_charter")
    _llm = llm or LLMClient()
    user_feedback = state.get("user_feedback")
    existing_objective = state.get("objective")
    has_full_charter = bool(existing_objective and existing_objective.get("goals"))

    # Skip if a draft already exists and there's nothing new to refine.
    # This covers resume-to-approve cycles where the graph re-runs from START.
    if has_full_charter and not user_feedback:
        logger.debug("align_charter: draft exists, no feedback — skipping LLM call.")
        return {}

    if has_full_charter and user_feedback:
        # Refinement pass: user reviewed the docx and requested changes
        system, messages = build_charter_refine_messages(existing_objective, user_feedback)
    else:
        # First pass: generate from topic (user_feedback not needed here)
        topic = (existing_objective or {}).get("topic", "")
        if not topic:
            logger.warning("align_charter: no topic in state. Using placeholder.")
            topic = "Research topic not yet specified."
        system, messages = build_charter_draft_messages(topic)

    try:
        charter_obj: _CharterLLM = _llm.complete_structured(
            messages=messages,
            response_model=_CharterLLM,
            system=system,
        )
    except LLMClientError as exc:
        logger.error("Charter generation failed: %s", exc)
        return {"status": ResearchStatus.FAILED}

    # Preserve generated_at from the first draft; never update it on refinement
    existing_generated_at = (existing_objective or {}).get("generated_at", "")
    generated_at = existing_generated_at or _today_dd_mm_yyyy()

    objective = ResearchObjective(
        topic=charter_obj.topic,
        goals=charter_obj.goals,
        success_metrics=charter_obj.success_metrics,
        scope_constraints=charter_obj.scope_constraints,
        methodology_preference=charter_obj.methodology_preference,
        bibliography_style=charter_obj.bibliography_style,
        language=charter_obj.language,
        generated_at=generated_at,
        stage_guidelines=charter_obj.stage_guidelines,
    )

    workspace_path = state.get("workspace_path", "")
    checkpoint_docx_path: str = ""
    if workspace_path:
        ResearchWorkspace(Path(workspace_path)).log(
            f"Research Charter {'refined' if has_full_charter else 'drafted'} "
            f"for topic: {charter_obj.topic}"
        )
        project_ws = _get_project_workspace(workspace_path)
        if project_ws is not None:
            docx_bytes = _charter_to_docx(objective)
            if docx_bytes:
                saved = project_ws.save_checkpoint_preview(1, docx_bytes)
                checkpoint_docx_path = str(saved)
                logger.info("Checkpoint 1 preview saved: %s", saved)

    return {
        "objective": objective,
        "charter_approved": False,
        "checkpoint_label": checkpoint_docx_path,  # path shown at interrupt
        "user_feedback": None,
        "status": ResearchStatus.PLANNING,
    }


def review_charter(state: ResearchState) -> dict[str, Any]:
    """Checkpoint gate executed after the user resumes from a charter review.

    The graph pauses (``interrupt_before``) before this node.  When the user
    resumes with ``--feedback``, that text lands in ``user_feedback`` and this
    node signals that a revision cycle is needed.  When the user resumes
    without feedback, the charter is approved and the pipeline proceeds.

    Args:
        state: Current ResearchState. ``user_feedback`` carries revision
            instructions if the user requested changes.

    Returns:
        Partial state update with ``charter_approved`` set accordingly.
    """
    user_feedback = state.get("user_feedback")
    if user_feedback:
        logger.info("Charter revision requested: %s", user_feedback[:80])
        return {"charter_approved": False, "status": ResearchStatus.PLANNING}
    logger.info("Research Charter approved by user.")
    return {"charter_approved": True, "status": ResearchStatus.PLANNING}


def route_after_review_charter(state: ResearchState) -> str:
    """Conditional edge: decide whether to refine the charter or end CP1.

    Args:
        state: Current ResearchState after ``review_charter`` ran.

    Returns:
        ``"END"`` if charter approved, ``"align_charter"`` if revision needed.
    """
    return "END" if state.get("charter_approved") else "align_charter"


def cp2_router(_state: ResearchState) -> dict[str, Any]:
    """CP2 start router — no-op node; routing logic lives in route_cp2_start.

    Returns:
        Empty dict (no state changes).
    """
    return {}


def route_cp2_start(state: ResearchState) -> str:
    """Conditional edge from cp2_router: decide CP2 entry point.

    Args:
        state: Current ResearchState.

    Returns:
        ``"refine_literature"`` when a compiled review exists and user
        feedback is pending (correction cycle); ``"plan"`` otherwise (fresh
        literature research start).
    """
    if state.get("literature_review_doc") and state.get("user_feedback"):
        return "refine_literature"
    return "plan"


def plan(state: ResearchState, registry: SkillRegistry | None = None, llm: LLMClient | None = None) -> dict[str, Any]:
    """Generate an ExecutionPlan using the LLM.

    Args:
        state: Current ResearchState with objective and previous gaps.
        registry: Optional injected SkillRegistry (for testing).
        llm: Optional injected LLMClient (for testing).

    Returns:
        Partial state update with the new plan.
    """
    _configure_history(state, "plan")
    _registry = registry or _get_default_registry()
    _llm = llm or LLMClient()

    objective = state.get("objective") or ResearchObjective(
        topic="", goals=[], success_metrics=[]
    )
    stage = state.get("stage", PipelineStage.LITERATURE_REVIEW)
    stage_str = str(stage.value if hasattr(stage, "value") else stage)
    attempt = state.get("attempt", 0)
    previous_eval = state.get("evaluation")
    gaps = previous_eval.get("gaps", []) if previous_eval else []
    user_guidance = state.get("user_guidance")

    # Use stage-specific guidelines. For LITERATURE_REVIEW, fall back to
    # bibliographic-only defaults rather than the overall success_metrics,
    # so the planner never confuses CP2 scope with full project criteria.
    raw_guidelines = (objective.get("stage_guidelines") or {}).get(stage_str)
    if raw_guidelines:
        stage_guidelines: list[str] | None = raw_guidelines
    elif stage_str == PipelineStage.LITERATURE_REVIEW.value:
        stage_guidelines = _default_literature_review_guidelines()
    else:
        defaults = _default_stage_guidelines(stage_str)
        stage_guidelines = defaults if defaults else None

    system, messages = build_planning_messages(
        objective=objective,
        stage=stage_str,
        attempt=attempt,
        gaps=gaps,
        skill_registry_summary=_registry.all_as_dicts(),
        user_guidance=user_guidance,
        stage_guidelines=stage_guidelines,
    )

    try:
        plan_obj: _ExecutionPlanLLM = _llm.complete_structured(
            messages=messages,
            response_model=_ExecutionPlanLLM,
            system=system,
        )
    except LLMClientError as exc:
        logger.error("Plan generation failed: %s", exc)
        return {"status": ResearchStatus.FAILED}

    steps: list[PlanStep] = []
    for i, raw_step in enumerate(plan_obj.steps):
        steps.append(PlanStep(
            step_id=i,
            skill_name=raw_step.get("skill_name", ""),
            parameters=raw_step.get("parameters", {}),
            depends_on=raw_step.get("depends_on", []),
            rationale=raw_step.get("rationale", ""),
        ))

    execution_plan = ExecutionPlan(
        steps=steps,
        rationale=plan_obj.rationale,
        estimated_cost=plan_obj.estimated_cost,
        attempt=attempt,
    )

    return {
        "plan": execution_plan,
        "status": ResearchStatus.EXECUTING,
        "user_guidance": None,  # consumed; clear so it doesn't repeat on next attempt
    }


def execute(
    state: ResearchState,
    registry: SkillRegistry | None = None,
) -> dict[str, Any]:
    """Execute the current plan by running each skill step.

    Steps with no dependencies run concurrently via asyncio.gather.
    Steps with dependencies run after their prerequisites complete.

    Args:
        state: Current ResearchState with plan and objective.
        registry: Optional injected SkillRegistry (for testing).

    Returns:
        Partial state update with findings accumulated from skill outputs.
    """
    _registry = registry or _get_default_registry()
    current_plan = state.get("plan")
    if not current_plan or not current_plan.get("steps"):
        logger.warning("execute: no plan or empty steps — skipping.")
        return {"findings": state.get("findings", []), "findings_current": []}

    objective = state.get("objective")
    stage = state.get("stage", PipelineStage.LITERATURE_REVIEW)
    attempt = state.get("attempt", 0)

    steps = current_plan["steps"]
    outputs: dict[int, SkillOutput] = {}

    # Group steps by dependency level for sequential batching
    batches = _topological_batches(steps)

    for batch in batches:
        batch_inputs: list[tuple[int, SkillInput]] = []
        for step in batch:
            skill_input = SkillInput({
                "parameters": {**step["parameters"], "_skill_name": step.get("skill_name", "")},
                "objective": objective,
                "stage": str(stage.value if hasattr(stage, "value") else stage),
                "attempt": attempt,
            })
            batch_inputs.append((step["step_id"], skill_input))

        # Run batch concurrently
        batch_outputs = asyncio.run(
            _run_batch_async(_registry, batch_inputs)
        )
        outputs.update(batch_outputs)

        # Log each skill execution to llm_history.yaml
        _history.configure(state.get("workspace_path", ""))
        stage_str_exec = str(stage.value if hasattr(stage, "value") else stage)
        for step in batch:
            sid = step["step_id"]
            out = batch_outputs.get(sid)
            if out is not None:
                _history.log_skill_call(
                    skill=step.get("skill_name", "unknown"),
                    stage=stage_str_exec,
                    attempt=attempt,
                    request_sent={
                        k: v for k, v in step.get("parameters", {}).items()
                        if k not in ("content",)  # omit large text blobs
                    },
                    response_received={
                        "confidence": out.get("confidence", 0.0),
                        "sources_count": len(out.get("sources", [])),
                        "sources": out.get("sources", [])[:20],
                        "error": out.get("error"),
                        "result_keys": list(out.get("result", {}).keys()),
                    },
                )

    findings_current: list[SkillOutput] = list(outputs.values())
    existing_findings = state.get("findings", [])
    all_findings = existing_findings + findings_current

    return {
        "findings": all_findings,
        "findings_current": findings_current,
        "status": ResearchStatus.EVALUATING,
    }


def evaluate(
    state: ResearchState,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Evaluate findings against success metrics and decide whether to retry.

    Args:
        state: Current ResearchState with findings and objective.
        llm: Optional injected LLMClient (for testing).

    Returns:
        Partial state update with evaluation result and incremented attempt counter.
    """
    _configure_history(state, "evaluate")
    _llm = llm or LLMClient()
    objective = state.get("objective") or ResearchObjective(
        topic="", goals=[], success_metrics=[]
    )
    # Use only the current attempt's findings for evaluation so that poor results
    # from earlier retries do not contaminate scoring of the current attempt.
    findings = state.get("findings_current") or state.get("findings", [])
    attempt = state.get("attempt", 0)
    quality_history = state.get("quality_history", [])
    stage = state.get("stage", PipelineStage.LITERATURE_REVIEW)
    stage_str = str(stage.value if hasattr(stage, "value") else stage)

    # Use stage-specific guidelines. For LITERATURE_REVIEW, fall back to
    # bibliographic-only defaults so evaluation never mixes CP2 scope with
    # full project criteria (implementation, publication, etc.).
    raw_guidelines = (objective.get("stage_guidelines") or {}).get(stage_str)
    if raw_guidelines:
        stage_guidelines: list[str] | None = raw_guidelines
    elif stage_str == PipelineStage.LITERATURE_REVIEW.value:
        stage_guidelines = _default_literature_review_guidelines()
    else:
        defaults = _default_stage_guidelines(stage_str)
        stage_guidelines = defaults if defaults else None

    system, messages = build_evaluation_messages(
        objective=objective,
        findings=findings,
        threshold=_CONVERGENCE_THRESHOLD,
        stage_guidelines=stage_guidelines,
    )

    try:
        eval_obj: _EvaluationResultLLM = _llm.complete_structured(
            messages=messages,
            response_model=_EvaluationResultLLM,
            system=system,
        )
    except LLMClientError as exc:
        logger.error("Evaluation failed: %s", exc)
        # Fail-safe: treat as not converged with empty gaps
        eval_obj = _EvaluationResultLLM(
            per_metric=[],
            gaps=[f"Evaluation error: {exc}"],
        )

    # Compute total_score and converged deterministically in Python.
    # Never trust the LLM to do arithmetic — it introduces non-determinism.
    per_metric: list[MetricScore] = [
        MetricScore(
            metric=m.get("metric", ""),
            score=float(m.get("score", 0.0)),
            rationale=m.get("rationale", ""),
            gaps=m.get("gaps", []),
        )
        for m in eval_obj.per_metric
    ]
    scores = [m["score"] for m in per_metric]
    total_score = sum(scores) / len(scores) if scores else 0.0
    converged = total_score >= _CONVERGENCE_THRESHOLD

    # Detect regression vs previous attempt
    previous_score = quality_history[-1]["total_score"] if quality_history else None
    regression = (
        previous_score is not None
        and (previous_score - total_score) > 0.05
    )

    evaluation = EvaluationResult(
        per_metric=per_metric,
        total_score=total_score,
        converged=converged,
        gaps=eval_obj.gaps,
        regression=regression,
    )

    # Build quality snapshot
    skills_used = list({
        f.get("skill_name", "") for f in findings if f.get("skill_name")
    })
    snapshot = QualitySnapshot(
        attempt=attempt,
        stage=stage_str,
        total_score=total_score,
        per_metric_scores={m["metric"]: float(m["score"]) for m in eval_obj.per_metric},
        skills_used=skills_used,
        cache_hit_rate=0.0,  # ResearchMemory added in Phase 2
    )

    # Update both the continuous history and the per-stage isolated history
    stage_quality_history: dict[str, list[QualitySnapshot]] = dict(
        state.get("stage_quality_history") or {}
    )
    stage_history_for_stage = list(stage_quality_history.get(stage_str, []))
    stage_history_for_stage.append(snapshot)
    stage_quality_history[stage_str] = stage_history_for_stage

    if regression:
        logger.warning(
            "Quality regression detected: %.2f → %.2f (attempt %d)",
            previous_score,
            total_score,
            attempt,
        )

    return {
        "evaluation": evaluation,
        "attempt": attempt + 1,
        "quality_history": quality_history + [snapshot],
        "stage_quality_history": stage_quality_history,
        "status": ResearchStatus.CONVERGED if converged else ResearchStatus.EXECUTING,
    }


def request_support(state: ResearchState) -> dict[str, Any]:
    """Notify the user that convergence failed and collect revised guidance.

    This node sets checkpoint_pending=True so that the graph pauses via
    interrupt_before and waits for user input before replanning.

    Args:
        state: Current ResearchState with evaluation gaps.

    Returns:
        Partial state update requesting user feedback.
    """
    evaluation = state.get("evaluation") or {}
    gaps = evaluation.get("gaps", [])

    logger.warning(
        "Pipeline could not converge after 5 attempts. Requesting user support."
    )

    gaps_display = "\n".join(f"  - {g}" for g in gaps) if gaps else "  (no specific gaps identified)"
    message = (
        f"After 5 attempts, the research pipeline could not meet all success metrics.\n\n"
        f"Unresolved gaps:\n{gaps_display}\n\n"
        f"Please provide guidance to help the agent replan."
    )

    workspace_path = state.get("workspace_path", "")
    if workspace_path:
        ResearchWorkspace(Path(workspace_path)).log(
            f"User support requested. Gaps: {gaps}"
        )

    return {
        "status": ResearchStatus.CHECKPOINT_PENDING,
        "checkpoint_pending": True,
        "checkpoint_label": (
            f"SUPPORT NEEDED: {message}"
        ),
        "attempt": 0,  # Reset attempt counter for replanning
    }


# ---------------------------------------------------------------------------
# Checkpoint 2 — Literature Review nodes
# ---------------------------------------------------------------------------


def compile_literature(
    state: ResearchState,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Compile a structured literature review from the accumulated findings.

    Calls the LLM with a fixed output schema (_LiteratureReviewLLM) to ensure
    a consistent format across multiple runs.

    Args:
        state: Current ResearchState with findings and objective.
        llm: Optional injected LLMClient (for testing).

    Returns:
        Partial state update with ``literature_review_doc`` and
        ``active_checkpoint=2``.
    """
    _configure_history(state, "compile_literature")
    _llm = llm or LLMClient()
    findings = state.get("findings", [])
    charter_document_text = state.get("charter_document_text", "")

    # Sort by confidence (desc) and take top 40 to prevent token overflow.
    # Uses all accumulated findings (not just current attempt) for richer compilation.
    sorted_findings = sorted(findings, key=lambda f: f.get("confidence", 0.0), reverse=True)
    top_findings = sorted_findings[:40]

    findings_summary: list[dict[str, Any]] = []
    for f in top_findings:
        skill_name = f.get("skill_name", "")
        result = f.get("result") or {}
        entry: dict[str, Any] = {
            "skill": skill_name,
            "confidence": f.get("confidence", 0.0),
            "sources": f.get("sources", [])[:5],
        }
        if skill_name == "article_search":
            papers = result.get("papers", [])
            entry["papers"] = [
                {
                    "title": p.get("title", ""),
                    "abstract": (p.get("abstract", "") or "")[:300],
                    "url": p.get("url", ""),
                    "year": p.get("year"),
                    "authors": p.get("authors", [])[:3],
                    "venue": p.get("venue", ""),
                }
                for p in papers[:15]
            ]
        elif skill_name == "content_summarizer":
            entry["summary"] = (result.get("summary", "") or "")[:500]
            entry["key_points"] = result.get("key_points", [])[:5]
        else:
            for k, v in list(result.items())[:5]:
                if isinstance(v, str):
                    entry[k] = v[:400]
                elif isinstance(v, list):
                    entry[k] = v[:10]
                else:
                    entry[k] = v
        findings_summary.append(entry)

    system, messages = build_compile_messages(
        charter_document_text=charter_document_text,
        findings=findings_summary,
    )

    try:
        review_obj: _LiteratureReviewLLM = _llm.complete_structured(
            messages=messages,
            response_model=_LiteratureReviewLLM,
            system=system,
            temperature=0.3,
        )
    except LLMClientError as exc:
        logger.error("Literature review compilation failed: %s", exc)
        return {"status": ResearchStatus.FAILED}

    sections: list[LiteratureReviewSection] = [
        LiteratureReviewSection(section_title=s.section_title, content=s.content)
        for s in review_obj.sections
    ]
    references: list[dict[str, Any]] = [
        {
            "reference_number": r.reference_number,
            "title": r.title,
            "authors": r.authors,
            "year": r.year,
            "url": r.url,
            "abnt_entry": r.abnt_entry,
            "summary": r.summary,
        }
        for r in review_obj.references
    ]

    literature_review_doc = LiteratureReviewDoc(
        sections=sections,
        references=references,
        verified_sources=[],
    )

    workspace_path = state.get("workspace_path", "")
    if workspace_path:
        ResearchWorkspace(Path(workspace_path)).log(
            f"Literature review compiled: {len(sections)} sections, "
            f"{len(references)} references."
        )

    return {
        "literature_review_doc": literature_review_doc,
        "active_checkpoint": 2,
        "status": ResearchStatus.EXECUTING,
    }


def verify_literature(
    state: ResearchState,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Verify each bibliographic source with an independent LLM agent.

    For every reference in ``literature_review_doc``:
    1. Attempt to fetch the URL via HTTP.
    2. Send fetched content + claim to a separate LLM call (temperature=0)
       to check factual consistency.
    3. Attach a :class:`SourceVerification` record to the document.

    This node is intentionally separate from ``compile_literature`` so that
    the verification agent cannot rationalise its own previous claims.

    Args:
        state: Current ResearchState with ``literature_review_doc``.
        llm: Optional injected LLMClient (for testing).

    Returns:
        Partial state update with verified ``literature_review_doc``.
    """
    _configure_history(state, "verify_literature")
    _llm = llm or LLMClient()
    review_doc: dict[str, Any] = dict(state.get("literature_review_doc") or {})
    references: list[dict[str, Any]] = list(review_doc.get("references", []))

    if not references:
        logger.warning("verify_literature: no references to verify.")
        return {"literature_review_doc": LiteratureReviewDoc(**review_doc)}

    access_date = _abnt_access_date()

    # Replace ACCESS_DATE placeholder in all references upfront
    for ref in references:
        abnt_entry = ref.get("abnt_entry", "")
        ref["abnt_entry"] = abnt_entry.replace("{ACCESS_DATE}", access_date)

    def _verify_one(ref: dict[str, Any]) -> SourceVerification:
        """Verify a single reference — runs in a thread pool worker."""
        url = ref.get("url", "")
        ref_num = ref.get("reference_number", 0)
        title = ref.get("title", "")
        summary = ref.get("summary", "")

        accessible, fetched_content = _fetch_url_content(url)

        content_matches = False
        note = "URL não acessível — verificação automática não foi possível."

        if accessible:
            system, messages = build_verify_messages(
                reference_number=ref_num,
                title=title,
                summary=summary,
                url=url,
                fetched_content=fetched_content,
            )
            try:
                v_result: _VerifyResultLLM = _llm.complete_structured(
                    messages=messages,
                    response_model=_VerifyResultLLM,
                    system=system,
                    temperature=0.0,
                )
                content_matches = v_result.content_matches
                note = v_result.verification_note
            except LLMClientError as exc:
                logger.warning(
                    "Verification LLM call failed for ref [%d]: %s", ref_num, exc
                )
                note = f"Verificação automática falhou: {exc}"

        return SourceVerification(
            reference_number=ref_num,
            url=url,
            title=title,
            accessible=accessible,
            content_matches=content_matches,
            verification_note=note,
            access_date=access_date,
        )

    verified: list[SourceVerification] = []
    max_workers = min(5, max(1, len(references)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_verify_one, ref): ref for ref in references}
        for future in as_completed(futures):
            try:
                verified.append(future.result())
            except Exception as exc:
                ref = futures[future]
                logger.warning("Verification failed for ref [%s]: %s", ref.get("url", "?"), exc)

    review_doc["references"] = references
    review_doc["verified_sources"] = verified  # type: ignore[assignment]

    workspace_path = state.get("workspace_path", "")
    if workspace_path:
        ok = sum(1 for v in verified if v.get("content_matches"))
        warn = sum(1 for v in verified if v.get("accessible") and not v.get("content_matches"))
        fail = sum(1 for v in verified if not v.get("accessible"))
        ResearchWorkspace(Path(workspace_path)).log(
            f"Source verification: ✓ {ok} verified, ⚠ {warn} questionable, ✗ {fail} inaccessible."
        )

    return {"literature_review_doc": LiteratureReviewDoc(**review_doc)}


def deliver_literature(state: ResearchState) -> dict[str, Any]:
    """Convert the verified literature review to a .docx and save Checkpoint 2 preview.

    Renders each section with its content, then a bibliography section with
    coloured verification marks (✓ green / ⚠ orange / ✗ red) beside each entry.

    Args:
        state: Current ResearchState with a verified ``literature_review_doc``.

    Returns:
        Partial state update with ``checkpoint_label`` pointing to the saved preview.
    """
    review_doc: dict[str, Any] = state.get("literature_review_doc") or {}
    workspace_path = state.get("workspace_path", "")
    checkpoint_docx_path = ""

    if workspace_path:
        docx_bytes = _literature_review_to_docx(review_doc)
        if docx_bytes:
            project_ws = _get_project_workspace(workspace_path)
            if project_ws is not None:
                saved = project_ws.save_checkpoint_preview(2, docx_bytes)
                checkpoint_docx_path = str(saved)
                logger.info("Checkpoint 2 preview saved: %s", saved)
        ResearchWorkspace(Path(workspace_path)).log("Literature Review preview saved.")

    return {
        "checkpoint_label": checkpoint_docx_path,
        "literature_approved": False,
        "status": ResearchStatus.PLANNING,
    }


def review_literature(state: ResearchState) -> dict[str, Any]:
    """Checkpoint gate executed after the user resumes from the literature review.

    The graph pauses (``interrupt_before``) before this node.  When the user
    resumes with ``--correct``, formatted feedback lands in ``user_feedback``
    and this node signals that a revision cycle is needed.  Without feedback
    the review is approved and the pipeline ends.

    Args:
        state: Current ResearchState. ``user_feedback`` carries revision
            instructions if the user requested changes.

    Returns:
        Partial state update with ``literature_approved`` set accordingly.
    """
    user_feedback = state.get("user_feedback")
    if user_feedback:
        logger.info("Literature review revision requested.")
        return {"literature_approved": False, "status": ResearchStatus.PLANNING}
    logger.info("Literature Review approved by user.")
    return {"literature_approved": True, "status": ResearchStatus.COMPLETED}


def refine_literature(
    state: ResearchState,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Refine the literature review based on researcher corrections from the .docx.

    Args:
        state: Current ResearchState with ``user_feedback`` and
            ``literature_review_doc``.
        llm: Optional injected LLMClient (for testing).

    Returns:
        Partial state update with refined ``literature_review_doc`` and
        cleared ``user_feedback``.
    """
    _configure_history(state, "refine_literature")
    _llm = llm or LLMClient()
    review_doc: dict[str, Any] = state.get("literature_review_doc") or {}
    user_feedback = state.get("user_feedback", "")

    system, messages = build_refine_messages(
        review_doc=review_doc,
        feedback=user_feedback or "",
    )

    try:
        refined_obj: _LiteratureReviewLLM = _llm.complete_structured(
            messages=messages,
            response_model=_LiteratureReviewLLM,
            system=system,
            temperature=0.3,
        )
    except LLMClientError as exc:
        logger.error("Literature review refinement failed: %s", exc)
        return {"status": ResearchStatus.FAILED}

    sections: list[LiteratureReviewSection] = [
        LiteratureReviewSection(section_title=s.section_title, content=s.content)
        for s in refined_obj.sections
    ]
    references: list[dict[str, Any]] = [
        {
            "reference_number": r.reference_number,
            "title": r.title,
            "authors": r.authors,
            "year": r.year,
            "url": r.url,
            "abnt_entry": r.abnt_entry,
            "summary": r.summary,
        }
        for r in refined_obj.references
    ]

    updated_doc = LiteratureReviewDoc(
        sections=sections,
        references=references,
        verified_sources=[],
    )

    workspace_path = state.get("workspace_path", "")
    if workspace_path:
        ResearchWorkspace(Path(workspace_path)).log(
            "Literature review refined based on researcher corrections."
        )

    return {
        "literature_review_doc": updated_doc,
        "user_feedback": None,
        "status": ResearchStatus.EXECUTING,
    }


def route_after_review_literature(state: ResearchState) -> str:
    """Conditional edge: decide next node after literature review gate.

    Args:
        state: Current ResearchState after ``review_literature`` ran.

    Returns:
        ``"END"`` if approved, ``"refine_literature"`` if revision needed.
    """
    return "END" if state.get("literature_approved") else "refine_literature"


# ---------------------------------------------------------------------------
# Edge routing functions
# ---------------------------------------------------------------------------

def route_after_evaluate(state: ResearchState) -> str:
    """Conditional edge: decide next node after evaluation.

    Args:
        state: Current ResearchState with evaluation result.

    Returns:
        Name of the next node: "compile_literature", "plan", or "request_support".
    """
    evaluation = state.get("evaluation")
    attempt = state.get("attempt", 0)
    max_retries = int(os.environ.get("AI_SKILL_MAX_RETRIES", "5"))

    if evaluation and evaluation.get("converged"):
        return "compile_literature"
    if attempt >= max_retries:
        return "request_support"
    return "plan"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_project_workspace(workspace_path: str) -> ProjectWorkspace | None:
    """Return the ProjectWorkspace linked to an internal state path, if any.

    Convention: the ResearchWorkspace lives at ``<project>/.state/``.
    If the parent of *workspace_path* contains ``workspace.yaml``, it is a
    ProjectWorkspace and is returned; otherwise returns None.

    Args:
        workspace_path: Absolute path to the ResearchWorkspace directory.

    Returns:
        The :class:`ProjectWorkspace` instance, or None.
    """
    if not workspace_path:
        return None
    parent = Path(workspace_path).parent
    if (parent / "workspace.yaml").exists():
        return ProjectWorkspace.from_path(parent)
    return None


def _charter_to_docx(objective: ResearchObjective) -> bytes:
    """Serialise a ResearchObjective into a minimal .docx file.

    Args:
        objective: The research objective TypedDict.

    Returns:
        Raw bytes of the generated .docx file.
    """
    try:
        import io
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        logger.warning("python-docx not available — skipping charter .docx export.")
        return b""

    doc = Document()
    doc.add_heading("Research Charter", level=1)

    topic = objective.get("topic", "")
    doc.add_heading("Tópico de Pesquisa", level=2)
    doc.add_paragraph(topic)

    goals = objective.get("goals", [])
    if goals:
        doc.add_heading("Objetivos", level=2)
        for goal in goals:
            doc.add_paragraph(goal, style="List Bullet")

    metrics = objective.get("success_metrics", [])
    if metrics:
        doc.add_heading("Métricas de Sucesso", level=2)
        for m in metrics:
            doc.add_paragraph(m, style="List Bullet")

    constraints = objective.get("scope_constraints", [])
    if constraints:
        doc.add_heading("Restrições de Escopo", level=2)
        for c in constraints:
            doc.add_paragraph(c, style="List Bullet")

    methodology = objective.get("methodology_preference", "")
    if methodology:
        doc.add_heading("Preferência Metodológica", level=2)
        doc.add_paragraph(methodology)

    # Render stage-specific guidelines — one section per CP stage
    _STAGE_DISPLAY_NAMES: dict[str, str] = {
        "literature_review":      "CP2 — Diretrizes da Revisão Bibliográfica",
        "research_design":        "CP3 — Diretrizes do Design de Pesquisa",
        "data_collection_guide":  "CP4 — Diretrizes da Coleta de Dados",
        "analysis_guide":         "CP5 — Diretrizes da Análise",
        "results_interpretation": "CP6/7 — Diretrizes de Interpretação de Resultados",
        "paper_composition":      "CP8 — Diretrizes para Redação do Artigo",
        "publication":            "CP8+ — Diretrizes para Publicação",
    }
    stage_guidelines: dict[str, list[str]] = objective.get("stage_guidelines") or {}
    for stage_key, display_name in _STAGE_DISPLAY_NAMES.items():
        directives = stage_guidelines.get(stage_key, [])
        if directives:
            doc.add_heading(display_name, level=2)
            for directive in directives:
                doc.add_paragraph(directive, style="List Bullet")

    bib_style = objective.get("bibliography_style", "abnt")
    language = objective.get("language", "pt-BR")
    generated_at = objective.get("generated_at", "")
    meta_para = doc.add_paragraph()
    meta_line = f"Estilo bibliográfico: {bib_style}  |  Idioma: {language}"
    if generated_at:
        meta_line += f"  |  Gerado em: {generated_at}"
    meta_para.add_run(meta_line).font.size = Pt(9)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()




def _configure_history(state: ResearchState, node_name: str) -> None:
    """Point the history logger at the current workspace and set node + stage names."""
    workspace_path = state.get("workspace_path", "")
    stage = state.get("stage")
    stage_str = stage.value if hasattr(stage, "value") else str(stage or "")
    if workspace_path:
        _history.configure(workspace_path)
    _history.set_current_stage(stage_str)
    set_current_node(node_name)


def _default_literature_review_guidelines() -> list[str]:
    """Return generic bibliographic-only guidelines for CP2.

    Used as fallback when the charter was created before stage_guidelines
    were introduced, or when the researcher did not customise them.
    These criteria are intentionally restricted to the bibliographic review
    scope — they never reference implementation, experiments, or publication.
    """
    return [
        "Realizar pelo menos 4 buscas distintas com queries diferentes cobrindo aspectos "
        "teóricos, metodológicos e aplicados do tema de pesquisa.",
        "Cada busca deve retornar resultados relevantes (total_found > 0); consultas em "
        "arXiv e Semantic Scholar são obrigatórias.",
        "O conjunto de buscas deve cobrir pelo menos 2 sub-áreas temáticas distintas "
        "derivadas do tópico de pesquisa.",
        "Ao menos 30% dos papers retornados devem ter sido publicados nos últimos 5 anos "
        "(campo year >= ano_atual - 5).",
        "Para ao menos 3 papers de alta relevância, executar summarização completa via "
        "content_summarizer para extrair contribuições detalhadas ao campo.",
        "As queries devem ser específicas o suficiente para retornar papers do domínio "
        "correto — queries genéricas demais (total_found > 200) devem ser refinadas.",
        "Cobrir tanto referências seminais (> 10 anos) quanto referências recentes "
        "(< 3 anos) para mostrar evolução do campo.",
        "Documentar os termos de busca e bases consultadas em cada step do plano.",
    ]


def _default_stage_guidelines(stage_str: str) -> list[str]:
    """Return generic stage-specific guidelines for stages without custom directives.

    Used as fallback when the charter was created without stage_guidelines for
    a given stage. Criteria are scoped to each stage's specific deliverable and
    never reference global project goals (implementation, publication, etc.).

    Args:
        stage_str: PipelineStage value string (e.g. "research_design").

    Returns:
        List of guideline strings, or empty list if stage_str is unrecognised.
    """
    defaults: dict[str, list[str]] = {
        "research_design": [
            "Identificar e justificar o método de pesquisa mais adequado ao objetivo.",
            "Formular pelo menos 2 hipóteses ou questões de pesquisa testáveis.",
            "Definir variáveis dependentes e independentes com precisão operacional.",
            "Especificar os critérios de validade e confiabilidade da abordagem escolhida.",
            "Relacionar a metodologia escolhida com precedentes na literatura revisada.",
        ],
        "data_collection_guide": [
            "Produzir protocolo de coleta com instrumento, passos e critérios de aceitação.",
            "Definir tamanho amostral mínimo com justificativa estatística.",
            "Especificar formato de entrega dos dados (estrutura, unidades, codificação).",
            "Incluir checklist de verificação pré-coleta para o pesquisador.",
        ],
        "analysis_guide": [
            "Descrever passo a passo do método analítico com fórmulas em LaTeX.",
            "Especificar softwares ou bibliotecas recomendados com versões.",
            "Incluir critérios de interpretação dos resultados (ex: p-value, effect size).",
            "Antecipar possíveis limitações ou fontes de erro na análise.",
        ],
        "results_interpretation": [
            "Descrever resultados com base exclusivamente nos dados fornecidos pelo usuário.",
            "Relacionar cada resultado com as hipóteses formuladas no design de pesquisa.",
            "Identificar resultados inesperados e propor explicações plausíveis.",
            "Discutir limitações dos resultados obtidos.",
        ],
        "paper_composition": [
            "Produzir todas as seções: Abstract, Introdução, Metodologia, Resultados, "
            "Discussão e Conclusão.",
            "Garantir coerência narrativa entre seções e consistência das citações inline.",
            "Aplicar formatação ABNT NBR 6023:2018 nas referências.",
            "Revisar que cada afirmação factual está suportada por uma citação.",
        ],
        "publication": [
            "Validar conformidade com TOP Guidelines (transparência, abertura).",
            "Verificar FAIR principles para dados e materiais de pesquisa.",
            "Exportar documento em .docx, Markdown e PDF.",
            "Revisar metadados do documento (título, autores, palavras-chave, resumo).",
        ],
    }
    return defaults.get(stage_str, [])


def _today_dd_mm_yyyy() -> str:
    """Return today's date in dd/mm/yyyy format for charter generation stamps.

    Returns:
        String like "22/03/2026".
    """
    from datetime import datetime
    now = datetime.now()
    return f"{now.day:02d}/{now.month:02d}/{now.year}"


def _abnt_access_date() -> str:
    """Return today's date in ABNT access-date format.

    Returns:
        String like "Acesso em: 22 mar. 2026."
    """
    from datetime import datetime
    _month_abbr = [
        "jan.", "fev.", "mar.", "abr.", "maio", "jun.",
        "jul.", "ago.", "set.", "out.", "nov.", "dez.",
    ]
    now = datetime.now()
    return f"Acesso em: {now.day} {_month_abbr[now.month - 1]} {now.year}."


def _fetch_url_content(url: str, timeout: int = 10) -> tuple[bool, str]:
    """Fetch a URL and return (accessible, text_snippet).

    A best-effort fetch: returns up to 3000 characters of the response body.
    Non-HTML content (PDFs, binaries) is returned as an empty snippet with
    ``accessible=True`` so the caller still marks the URL as reachable.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Tuple of (accessible, text_snippet).  ``accessible`` is True when
        the server returned any 2xx response.  ``text_snippet`` is up to
        3000 chars of decoded body (empty on error or binary content).
    """
    import urllib.error
    import urllib.request

    if not url or not url.startswith(("http://", "https://")):
        return False, ""

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; ai-skill-bot/1.0; "
                    "+https://github.com/leobr/plato)"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(8192)
            try:
                text = raw.decode("utf-8", errors="ignore")
            except Exception:
                text = ""
            return True, text[:3000]
    except urllib.error.HTTPError as exc:
        logger.debug("HTTP error fetching %s: %s", url, exc)
        return False, ""
    except Exception as exc:
        logger.debug("Failed to fetch %s: %s", url, exc)
        return False, ""


def _literature_review_to_docx(review_doc: dict[str, Any]) -> bytes:
    """Convert a LiteratureReviewDoc dict to a formatted .docx file.

    Renders:
    - A title heading "Revisão Bibliográfica"
    - Each section as Heading 2 + body paragraphs (Markdown-lite rendering)
    - A "Referências Bibliográficas" section with colour-coded entries:
        ✓ green  — accessible and content verified
        ⚠ orange — accessible but content questionable
        ✗ red    — inaccessible URL

    Args:
        review_doc: A dict matching the LiteratureReviewDoc shape.

    Returns:
        Raw bytes of the generated .docx file, or empty bytes on error.
    """
    try:
        import io
        from docx import Document
        from docx.shared import Pt, RGBColor
    except ImportError:
        logger.warning("python-docx not available — skipping literature review .docx export.")
        return b""

    doc = Document()
    doc.add_heading("Revisão Bibliográfica", level=1)

    sections = review_doc.get("sections", [])
    for section in sections:
        heading = section.get("section_title", "")
        content = section.get("content", "")
        if heading:
            doc.add_heading(heading, level=2)
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("### "):
                doc.add_heading(stripped[4:], level=3)
            elif stripped.startswith("## "):
                doc.add_heading(stripped[3:], level=2)
            elif stripped.startswith("# "):
                doc.add_heading(stripped[2:], level=1)
            elif stripped.startswith("- "):
                doc.add_paragraph(stripped[2:], style="List Bullet")
            else:
                doc.add_paragraph(stripped)

    # Build verification lookup
    verified_map: dict[int, dict[str, Any]] = {}
    for v in review_doc.get("verified_sources", []):
        ref_num = v.get("reference_number", 0)
        verified_map[ref_num] = v  # type: ignore[assignment]

    references = review_doc.get("references", [])
    if references:
        doc.add_heading("Referências Bibliográficas", level=2)
        for ref in references:
            ref_num = ref.get("reference_number", 0)
            abnt_entry = ref.get("abnt_entry", "")
            v_info = verified_map.get(ref_num, {})
            accessible = v_info.get("accessible", False)
            matches = v_info.get("content_matches", False)
            note = v_info.get("verification_note", "")

            if accessible and matches:
                mark = "✓"
                color = RGBColor(0, 128, 0)  # green
            elif accessible:
                mark = "⚠"
                color = RGBColor(204, 102, 0)  # orange
            else:
                mark = "✗"
                color = RGBColor(180, 0, 0)  # red

            para = doc.add_paragraph()
            mark_run = para.add_run(f"[{mark}] ")
            mark_run.font.color.rgb = color
            mark_run.font.bold = True
            mark_run.font.size = Pt(9)

            entry_run = para.add_run(abnt_entry)
            entry_run.font.size = Pt(9)

            if note:
                note_run = para.add_run(f"\n        [{note}]")
                note_run.font.size = Pt(8)
                note_run.font.italic = True
                note_run.font.color.rgb = RGBColor(100, 100, 100)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _get_default_registry() -> SkillRegistry:
    """Create and auto-discover a SkillRegistry instance.

    Returns:
        A SkillRegistry with all available built-in skills registered.
    """
    from ai_skill.skills.registry import SkillRegistry
    registry = SkillRegistry()
    registry.auto_discover()
    return registry


def _topological_batches(steps: list[PlanStep]) -> list[list[PlanStep]]:
    """Group plan steps into batches that can run concurrently.

    Steps with no unresolved dependencies form a batch. Each batch runs
    after the previous one completes.

    Args:
        steps: List of PlanStep with step_id and depends_on fields.

    Returns:
        List of batches (each batch is a list of steps to run concurrently).
    """
    completed: set[int] = set()
    remaining = list(steps)
    batches: list[list[PlanStep]] = []

    max_iterations = len(steps) + 1  # safety valve against cycles
    iterations = 0

    while remaining and iterations < max_iterations:
        batch = [
            s for s in remaining
            if all(dep in completed for dep in s.get("depends_on", []))
        ]
        if not batch:
            # Remaining steps have unresolvable dependencies — add them as-is
            batches.append(remaining)
            break
        batches.append(batch)
        for step in batch:
            completed.add(step["step_id"])
        remaining = [s for s in remaining if s["step_id"] not in completed]
        iterations += 1

    return batches


async def _run_batch_async(
    registry: SkillRegistry,
    batch_inputs: list[tuple[int, SkillInput]],
) -> dict[int, SkillOutput]:
    """Run a batch of skill steps concurrently using asyncio.gather.

    Args:
        registry: The skill registry to look up skill instances.
        batch_inputs: List of (step_id, SkillInput) pairs.

    Returns:
        Dict mapping step_id → SkillOutput.
    """
    async def run_one(step_id: int, skill_input: SkillInput) -> tuple[int, SkillOutput]:
        skill_name = skill_input.parameters.get("_skill_name", "")
        # The skill_name is stored in the parameters under a special key
        # set by the execute node; extract it before passing to the skill.
        clean_params = {k: v for k, v in skill_input.parameters.items() if k != "_skill_name"}
        skill_input_clean = SkillInput({**dict(skill_input), "parameters": clean_params})

        try:
            skill = registry.get(skill_name)
            output = await skill.arun(skill_input_clean)
        except Exception as exc:
            logger.error("Skill '%s' raised an exception: %s", skill_name, exc)
            output = SkillOutput(
                skill_name=skill_name,
                result={},
                confidence=0.0,
                sources=[],
                error=str(exc),
                cached=False,
            )
        return step_id, output

    tasks = [run_one(sid, sinput) for sid, sinput in batch_inputs]
    results = await asyncio.gather(*tasks)
    return dict(results)


