"""Prompt templates for the PlannerAgent node."""

from __future__ import annotations

import json
from typing import Any

from ai_skill.core.state import ResearchObjective


PLANNING_SYSTEM = """\
You are a research planning expert. Your task is to create an execution plan
for an academic research pipeline using available skills.

Available skills (name → description):
{skill_registry_json}

Rules:
- Use only skills listed above.
- Each step must reference exactly one skill by its name.
- Specify depends_on as a list of step_id integers that must complete first.
- Steps with no dependencies (depends_on: []) can run in parallel.
- Keep the plan focused on the current stage objectives.
- Estimate cost as: low (<10 API calls), medium (10-50), high (>50).

content_summarizer — IMPORTANT:
- Requires EITHER "content" (pre-fetched text) OR "source_url" (auto-fetched).
  NEVER call it without at least one of these — it will fail.
- When chaining after article_search or web_search, pass the article
  abstract/snippet as "content" to avoid an extra HTTP round-trip.
- With URL only:  {{"source_url": "https://...", "content_type": "article", "source_year": 2023}}
- With text:      {{"content": "<fetched text>", "source_url": "https://...", "content_type": "article"}}
"""

PLANNING_USER = """\
Research objective (overall context):
{objective_json}

Current stage: {stage}
Attempt: {attempt} / 5

Stage-specific guidelines (these are the primary driver for this plan):
{stage_guidelines_json}

Previous attempt gaps (empty on first attempt):
{gaps_json}

Produce an ExecutionPlan whose steps directly address the stage-specific guidelines
above. Use the overall objective only for thematic context. Fill the gaps listed if
this is a retry.
"""

PLANNING_USER_WITH_GUIDANCE = """\
Research objective (overall context):
{objective_json}

Current stage: {stage}
Attempt: {attempt} / 5

Stage-specific guidelines (primary driver):
{stage_guidelines_json}

Previous attempt gaps:
{gaps_json}

ORIENTAÇÃO DO PESQUISADOR (considerar com prioridade máxima):
{user_guidance}

Produza um ExecutionPlan que incorpore diretamente a orientação acima,
além de endereçar as lacunas identificadas e seguir as diretrizes da etapa.
"""


def build_planning_messages(
    objective: ResearchObjective,
    stage: str,
    attempt: int,
    gaps: list[str],
    skill_registry_summary: list[dict[str, Any]],
    user_guidance: str | None = None,
    stage_guidelines: list[str] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Build the system prompt and messages list for the planning node.

    When *stage_guidelines* is provided and non-empty those directives become
    the primary driver of the plan.  When absent the planner falls back to the
    overall ``objective.success_metrics`` context.

    Args:
        objective: The confirmed research objective.
        stage: Current pipeline stage name.
        attempt: Current attempt index (0-based).
        gaps: List of gap strings from the previous EvaluationResult.
        skill_registry_summary: List of dicts with 'name' and 'description'
            for each registered skill.
        user_guidance: Optional researcher guidance collected after convergence failure.
        stage_guidelines: Stage-specific directives from
            ``objective["stage_guidelines"][stage]``.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    effective_guidelines: list[str] = stage_guidelines or objective.get("success_metrics", [])

    # When stage_guidelines are present, strip success_metrics from the objective
    # so the planner cannot accidentally design steps towards global project goals
    # (implementation, experiments, publication) that are out of scope for this stage.
    if stage_guidelines:
        objective_for_prompt: dict[str, Any] = {
            k: v for k, v in dict(objective).items()
            if k not in ("success_metrics",)
        }
    else:
        objective_for_prompt = dict(objective)

    objective_json = json.dumps(objective_for_prompt, ensure_ascii=False, indent=2)

    system = PLANNING_SYSTEM.format(
        skill_registry_json=json.dumps(skill_registry_summary, ensure_ascii=False, indent=2)
    )
    if user_guidance:
        user_content = PLANNING_USER_WITH_GUIDANCE.format(
            objective_json=objective_json,
            stage=stage,
            attempt=attempt,
            stage_guidelines_json=json.dumps(effective_guidelines, ensure_ascii=False, indent=2),
            gaps_json=json.dumps(gaps, ensure_ascii=False, indent=2),
            user_guidance=user_guidance,
        )
    else:
        user_content = PLANNING_USER.format(
            objective_json=objective_json,
            stage=stage,
            attempt=attempt,
            stage_guidelines_json=json.dumps(effective_guidelines, ensure_ascii=False, indent=2),
            gaps_json=json.dumps(gaps, ensure_ascii=False, indent=2),
        )
    messages = [{"role": "user", "content": user_content}]
    return system, messages
