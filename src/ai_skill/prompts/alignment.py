"""Prompt templates for the user alignment (Research Charter) node."""

from __future__ import annotations

import json

from ai_skill.core.state import ResearchObjective


CHARTER_DRAFT_SYSTEM = """\
You are an academic research advisor. Given a free-form research topic,
draft a structured Research Charter with:
- 3-5 specific, measurable research goals
- 3-5 concrete success metrics (quantifiable where possible)
- Any implicit scope constraints you can infer from the topic
- A suggested research methodology (if it can be inferred)
- The bibliography style (default: abnt)
- The primary language (default: pt-BR)

Be specific and academic in tone. Avoid vague statements.
"""

CHARTER_DRAFT_USER = """\
Research topic: {topic}

Draft a Research Charter for this topic.
"""

CHARTER_REFINE_SYSTEM = """\
You are an academic research advisor helping a researcher refine their
Research Charter. The researcher has provided feedback on the current draft.
Incorporate their changes precisely and return the updated charter.
"""

CHARTER_REFINE_USER = """\
Current charter:
{charter_json}

Researcher feedback:
{feedback}

Return the complete updated Research Charter incorporating the feedback.
"""

METRICS_SUGGEST_SYSTEM = """\
You are an academic research methods expert. Suggest 2-3 additional success
metrics for the research objective that would strengthen the evaluation.
Focus on measurability and academic rigour.
"""

METRICS_SUGGEST_USER = """\
Research topic: {topic}

Current goals:
{goals_json}

Current metrics:
{metrics_json}

Suggest additional metrics that are complementary (not redundant).
"""


def build_charter_draft_messages(topic: str) -> tuple[str, list[dict[str, str]]]:
    """Build messages to draft an initial Research Charter from a topic.

    Args:
        topic: Free-form description of the research topic.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    user_content = CHARTER_DRAFT_USER.format(topic=topic)
    return CHARTER_DRAFT_SYSTEM, [{"role": "user", "content": user_content}]


def build_charter_refine_messages(
    charter: ResearchObjective, feedback: str
) -> tuple[str, list[dict[str, str]]]:
    """Build messages to refine a charter based on user feedback.

    Args:
        charter: The current Research Charter draft.
        feedback: User's free-form feedback or edit instructions.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    user_content = CHARTER_REFINE_USER.format(
        charter_json=json.dumps(dict(charter), ensure_ascii=False, indent=2),
        feedback=feedback,
    )
    return CHARTER_REFINE_SYSTEM, [{"role": "user", "content": user_content}]


def build_metrics_suggest_messages(
    objective: ResearchObjective,
) -> tuple[str, list[dict[str, str]]]:
    """Build messages to suggest additional success metrics.

    Args:
        objective: The current Research Charter with topic, goals, and metrics.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    user_content = METRICS_SUGGEST_USER.format(
        topic=objective.get("topic", ""),
        goals_json=json.dumps(
            objective.get("goals", []), ensure_ascii=False, indent=2
        ),
        metrics_json=json.dumps(
            objective.get("success_metrics", []), ensure_ascii=False, indent=2
        ),
    )
    return METRICS_SUGGEST_SYSTEM, [{"role": "user", "content": user_content}]
