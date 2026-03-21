"""Prompt templates for the EvaluatorAgent node."""

from __future__ import annotations

import json
from typing import Any

from ai_skill.core.state import ResearchObjective, SkillOutput


EVALUATION_SYSTEM = """\
You are a rigorous academic research evaluator. Your task is to assess how
well the collected research findings satisfy the stated success metrics.

For each metric, assign a score between 0.0 and 1.0:
- 1.0: Metric is fully and convincingly satisfied.
- 0.7-0.9: Metric is mostly satisfied with minor gaps.
- 0.4-0.6: Metric is partially satisfied; significant gaps remain.
- 0.0-0.3: Metric is barely or not satisfied.

Convergence threshold: {threshold}
If the weighted average of all scores >= {threshold}, set converged to true.

Be strict. Academic work demands rigour.
"""

EVALUATION_USER = """\
Research objective:
{objective_json}

Success metrics to evaluate:
{metrics_json}

Skill outputs collected (summaries):
{findings_json}

For each metric, provide score, rationale, and specific gaps.
Then set converged based on whether average score >= {threshold}.
"""


def build_evaluation_messages(
    objective: ResearchObjective,
    findings: list[SkillOutput],
    threshold: float = 0.75,
) -> tuple[str, list[dict[str, str]]]:
    """Build the system prompt and messages for the evaluation node.

    Args:
        objective: The confirmed research objective with success_metrics.
        findings: List of SkillOutput from the current execution attempt.
        threshold: Convergence threshold (default 0.75).

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    # Summarise findings to stay within token limits
    findings_summary: list[dict[str, Any]] = []
    for f in findings:
        findings_summary.append({
            "skill": f.get("skill_name", "unknown"),
            "confidence": f.get("confidence", 0.0),
            "error": f.get("error"),
            "result_keys": list(f.get("result", {}).keys()),
            "sources_count": len(f.get("sources", [])),
        })

    system = EVALUATION_SYSTEM.format(threshold=threshold)
    user_content = EVALUATION_USER.format(
        objective_json=json.dumps(dict(objective), ensure_ascii=False, indent=2),
        metrics_json=json.dumps(
            objective.get("success_metrics", []), ensure_ascii=False, indent=2
        ),
        findings_json=json.dumps(findings_summary, ensure_ascii=False, indent=2),
        threshold=threshold,
    )
    messages = [{"role": "user", "content": user_content}]
    return system, messages
