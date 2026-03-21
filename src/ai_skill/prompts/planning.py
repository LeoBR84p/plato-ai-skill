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
"""

PLANNING_USER = """\
Research objective:
{objective_json}

Current stage: {stage}
Attempt: {attempt} / 5

Previous attempt gaps (empty on first attempt):
{gaps_json}

Produce an ExecutionPlan that addresses the objective, focusing on filling
the gaps listed above if this is a retry.
"""


def build_planning_messages(
    objective: ResearchObjective,
    stage: str,
    attempt: int,
    gaps: list[str],
    skill_registry_summary: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Build the system prompt and messages list for the planning node.

    Args:
        objective: The confirmed research objective.
        stage: Current pipeline stage name.
        attempt: Current attempt index (0-based).
        gaps: List of gap strings from the previous EvaluationResult.
        skill_registry_summary: List of dicts with 'name' and 'description'
            for each registered skill.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    system = PLANNING_SYSTEM.format(
        skill_registry_json=json.dumps(skill_registry_summary, ensure_ascii=False, indent=2)
    )
    user_content = PLANNING_USER.format(
        objective_json=json.dumps(dict(objective), ensure_ascii=False, indent=2),
        stage=stage,
        attempt=attempt,
        gaps_json=json.dumps(gaps, ensure_ascii=False, indent=2),
    )
    messages = [{"role": "user", "content": user_content}]
    return system, messages
