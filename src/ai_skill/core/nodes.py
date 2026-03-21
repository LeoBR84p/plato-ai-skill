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
import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ai_skill.core.llm_client import LLMClient, LLMClientError
from ai_skill.core.pipeline_stages import PipelineStage, ResearchStatus
from ai_skill.core.state import (
    EvaluationResult,
    ExecutionPlan,
    MetricScore,
    PlanStep,
    QualitySnapshot,
    ResearchObjective,
    ResearchState,
    SkillOutput,
)
from ai_skill.core.workspace import ResearchWorkspace
from ai_skill.prompts.alignment import (
    build_charter_draft_messages,
    build_charter_refine_messages,
)
from ai_skill.prompts.evaluation import build_evaluation_messages
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
    """LLM-parseable representation of an EvaluationResult."""

    per_metric: list[dict[str, Any]] = Field(
        description="List of {metric, score, rationale, gaps} dicts."
    )
    total_score: float = Field(description="Weighted average score 0.0-1.0.", ge=0.0, le=1.0)
    converged: bool = Field(
        description=f"True when total_score >= {_CONVERGENCE_THRESHOLD}."
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
    """Draft a Research Charter and present it to the user for alignment.

    In interactive mode (stdin), the user can edit individual fields or
    confirm the draft. This node is interrupted via interrupt_before in the
    compiled graph — the user input arrives through the checkpoint mechanism.

    Args:
        state: Current ResearchState. Must contain user_feedback if this is
            a refinement pass.
        llm: Optional injected LLMClient (for testing).

    Returns:
        Partial state update with objective and checkpoint_pending=True.
    """
    _llm = llm or LLMClient()
    user_feedback = state.get("user_feedback")
    existing_objective = state.get("objective")

    if existing_objective and user_feedback:
        # Refinement pass: user has reviewed the draft and provided feedback
        system, messages = build_charter_refine_messages(
            existing_objective, user_feedback
        )
    elif not existing_objective:
        # First pass: draft from scratch — we need a topic from user_feedback
        topic = user_feedback or state.get("objective", {}).get("topic", "")
        if not topic:
            logger.warning("align_charter: no topic provided. Using placeholder.")
            topic = "Research topic not yet specified."
        system, messages = build_charter_draft_messages(topic)
    else:
        # Already confirmed — no changes needed
        return {"checkpoint_pending": False, "user_feedback": None}

    try:
        charter_obj: _CharterLLM = _llm.complete_structured(
            messages=messages,
            response_model=_CharterLLM,
            system=system,
        )
    except LLMClientError as exc:
        logger.error("Charter generation failed: %s", exc)
        return {"status": ResearchStatus.FAILED}

    objective = ResearchObjective(
        topic=charter_obj.topic,
        goals=charter_obj.goals,
        success_metrics=charter_obj.success_metrics,
        scope_constraints=charter_obj.scope_constraints,
        methodology_preference=charter_obj.methodology_preference,
        bibliography_style=charter_obj.bibliography_style,
        language=charter_obj.language,
    )

    workspace_path = state.get("workspace_path", "")
    if workspace_path:
        workspace = ResearchWorkspace(Path(workspace_path))
        workspace.append_log(
            f"Research Charter drafted for topic: {charter_obj.topic}"
        )

    return {
        "objective": objective,
        "checkpoint_pending": True,
        "checkpoint_label": "CHECKPOINT 1: Review and approve the Research Charter",
        "user_feedback": None,
    }


def plan(state: ResearchState, registry: SkillRegistry | None = None, llm: LLMClient | None = None) -> dict[str, Any]:
    """Generate an ExecutionPlan using the LLM.

    Args:
        state: Current ResearchState with objective and previous gaps.
        registry: Optional injected SkillRegistry (for testing).
        llm: Optional injected LLMClient (for testing).

    Returns:
        Partial state update with the new plan.
    """
    _registry = registry or _get_default_registry()
    _llm = llm or LLMClient()

    objective = state.get("objective") or ResearchObjective(
        topic="", goals=[], success_metrics=[]
    )
    stage = state.get("stage", PipelineStage.LITERATURE_REVIEW)
    attempt = state.get("attempt", 0)
    previous_eval = state.get("evaluation")
    gaps = previous_eval.get("gaps", []) if previous_eval else []

    system, messages = build_planning_messages(
        objective=objective,
        stage=str(stage.value if hasattr(stage, "value") else stage),
        attempt=attempt,
        gaps=gaps,
        skill_registry_summary=_registry.all_as_dicts(),
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
        return {"findings": state.get("findings", [])}

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
                "parameters": step["parameters"],
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

    findings: list[SkillOutput] = list(outputs.values())
    existing_findings = state.get("findings", [])
    all_findings = existing_findings + findings

    return {
        "findings": all_findings,
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
    _llm = llm or LLMClient()
    objective = state.get("objective") or ResearchObjective(
        topic="", goals=[], success_metrics=[]
    )
    findings = state.get("findings", [])
    attempt = state.get("attempt", 0)
    quality_history = state.get("quality_history", [])

    system, messages = build_evaluation_messages(
        objective=objective,
        findings=findings,
        threshold=_CONVERGENCE_THRESHOLD,
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
            total_score=0.0,
            converged=False,
            gaps=[f"Evaluation error: {exc}"],
        )

    # Detect regression vs previous attempt
    previous_score = quality_history[-1]["total_score"] if quality_history else None
    regression = (
        previous_score is not None
        and (previous_score - eval_obj.total_score) > 0.05
    )

    per_metric: list[MetricScore] = [
        MetricScore(
            metric=m.get("metric", ""),
            score=float(m.get("score", 0.0)),
            rationale=m.get("rationale", ""),
            gaps=m.get("gaps", []),
        )
        for m in eval_obj.per_metric
    ]

    evaluation = EvaluationResult(
        per_metric=per_metric,
        total_score=eval_obj.total_score,
        converged=eval_obj.converged,
        gaps=eval_obj.gaps,
        regression=regression,
    )

    # Build quality snapshot
    stage = state.get("stage", PipelineStage.LITERATURE_REVIEW)
    skills_used = list({
        f.get("skill_name", "") for f in findings if f.get("skill_name")
    })
    snapshot = QualitySnapshot(
        attempt=attempt,
        stage=str(stage.value if hasattr(stage, "value") else stage),
        total_score=eval_obj.total_score,
        per_metric_scores={m["metric"]: float(m["score"]) for m in eval_obj.per_metric},
        skills_used=skills_used,
        cache_hit_rate=0.0,  # ResearchMemory added in Phase 2
    )

    if regression:
        logger.warning(
            "Quality regression detected: %.2f → %.2f (attempt %d)",
            previous_score,
            eval_obj.total_score,
            attempt,
        )

    return {
        "evaluation": evaluation,
        "attempt": attempt + 1,
        "quality_history": quality_history + [snapshot],
        "status": ResearchStatus.CONVERGED if eval_obj.converged else ResearchStatus.EXECUTING,
    }


def deliver(state: ResearchState) -> dict[str, Any]:
    """Write findings.md to the workspace and mark the project as completed.

    Args:
        state: Current ResearchState with findings and quality history.

    Returns:
        Partial state update with status=COMPLETED.
    """
    workspace_path = state.get("workspace_path", "")
    objective = state.get("objective") or {}
    findings = state.get("findings", [])
    quality_history = state.get("quality_history", [])
    evaluation = state.get("evaluation") or {}

    if workspace_path:
        workspace = ResearchWorkspace(Path(workspace_path))
        content = _format_findings_md(objective, findings, quality_history, evaluation)
        workspace.write_findings(content)
        workspace.append_log(
            f"Pipeline converged. Final score: {evaluation.get('total_score', 0):.2f}"
        )

    logger.info(
        "Research pipeline completed. Score: %.2f",
        evaluation.get("total_score", 0.0),
    )

    return {"status": ResearchStatus.COMPLETED}


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
        ResearchWorkspace(Path(workspace_path)).append_log(
            f"User support requested. Gaps: {gaps}"
        )

    return {
        "status": ResearchStatus.FAILED,
        "checkpoint_pending": True,
        "checkpoint_label": (
            f"SUPPORT NEEDED: {message}"
        ),
        "attempt": 0,  # Reset attempt counter for replanning
    }


# ---------------------------------------------------------------------------
# Edge routing functions
# ---------------------------------------------------------------------------

def route_after_evaluate(state: ResearchState) -> str:
    """Conditional edge: decide next node after evaluation.

    Args:
        state: Current ResearchState with evaluation result.

    Returns:
        Name of the next node: "deliver", "plan", or "request_support".
    """
    evaluation = state.get("evaluation")
    attempt = state.get("attempt", 0)
    max_retries = int(os.environ.get("AI_SKILL_MAX_RETRIES", "5"))

    if evaluation and evaluation.get("converged"):
        return "deliver"
    if attempt >= max_retries:
        return "request_support"
    return "plan"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

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


def _format_findings_md(
    objective: dict[str, Any],
    findings: list[SkillOutput],
    quality_history: list[QualitySnapshot],
    evaluation: dict[str, Any],
) -> str:
    """Format a findings.md document from the pipeline outputs.

    Args:
        objective: The research objective dict.
        findings: List of skill outputs.
        quality_history: List of quality snapshots.
        evaluation: Final EvaluationResult dict.

    Returns:
        Markdown string for findings.md.
    """
    lines: list[str] = [
        f"# Findings: {objective.get('topic', 'Research')}",
        "",
        "## Research Objective",
        "",
        f"**Topic**: {objective.get('topic', '')}",
        "",
        "**Goals**:",
    ]
    for goal in objective.get("goals", []):
        lines.append(f"- {goal}")

    lines += [
        "",
        "**Success Metrics**:",
    ]
    for metric in objective.get("success_metrics", []):
        lines.append(f"- {metric}")

    lines += [
        "",
        "## Quality Summary",
        "",
        f"**Final Score**: {evaluation.get('total_score', 0.0):.2f}",
        f"**Converged**: {evaluation.get('converged', False)}",
        f"**Attempts**: {len(quality_history)}",
        "",
    ]

    if evaluation.get("per_metric"):
        lines.append("**Per-metric scores**:")
        for m in evaluation["per_metric"]:
            lines.append(f"- {m.get('metric', '')}: {m.get('score', 0):.2f}")
        lines.append("")

    lines += [
        "## Sources Consulted",
        "",
    ]
    all_sources: list[str] = []
    for finding in findings:
        all_sources.extend(finding.get("sources", []))
    unique_sources = list(dict.fromkeys(s for s in all_sources if s))
    for src in unique_sources:
        lines.append(f"- {src}")

    return "\n".join(lines)
