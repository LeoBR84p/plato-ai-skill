"""Prompt templates for the PlannerAgent node."""

from __future__ import annotations

import json
from typing import Any

from ai_skill.core.state import ResearchObjective


def _stage_context(objective: ResearchObjective) -> dict[str, Any]:
    """Return the strict subset of objective fields exposed to stage-scoped LLMs.

    This is an explicit allowlist — only the three named fields are included.
    Any field added to ResearchObjective in the future is withheld by default.
    There is no dict iteration, no key-filter expression, and no catch-all that
    could accidentally forward a field that should stay hidden.

    Allowed:
        topic           — needed so the LLM knows the research subject.
        goals           — needed so search steps are thematically grounded.
        scope_constraints — needed so the planner respects declared boundaries.

    Withheld (with reason):
        success_metrics      — global project deliverables; causes the planner to
                               design steps toward implementation/publication/etc.
        methodology_preference — irrelevant to search/collection work.
        bibliography_style   — irrelevant at planning time.
        language             — irrelevant at planning time.
        generated_at         — metadata; not useful to the LLM.
        stage_guidelines     — passed separately as stage_guidelines_json.
    """
    return {
        "topic": objective.get("topic") or "",
        "goals": list(objective.get("goals") or []),
        "scope_constraints": list(objective.get("scope_constraints") or []),
    }


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
- MAXIMUM 15 steps per plan. Be concise — depth over breadth.
- DO NOT pre-write research content inside parameters. Parameters should be
  search queries, URLs, or short instructions — never multi-paragraph prose.
  The skills will fetch and summarize the actual content at runtime.
- String parameter values must be ≤ 300 characters each.

content_summarizer — IMPORTANT:
- Requires EITHER "content" (pre-fetched text, ≤ 300 chars) OR "source_url".
  NEVER call it without at least one of these — it will fail.
- "source_url" MUST be an HTTP/HTTPS URL. NEVER pass a local file path as
  "source_url" — for local PDF files use pdf_reader with "file_path" instead.
- Prefer "source_url" so the skill fetches live content. Use "content" only
  for short abstracts passed from a preceding article_search step.
- With URL only:  {{"source_url": "https://...", "content_type": "article", "source_year": 2023}}
- With text:      {{"content": "<abstract ≤ 300 chars>", "source_url": "https://...", "content_type": "article"}}

pdf_reader — for local PDF files in attachments/:
- Use "file_path" with the absolute path shown in the available files list.
- DO NOT use content_summarizer with a local path — use pdf_reader instead.
"""

PLANNING_USER = """\
Tópico, objetivos e restrições de escopo da pesquisa:
{objective_json}

Etapa atual: {stage}
Tentativa: {attempt} / 5

Diretrizes desta etapa (driver principal do plano):
{stage_guidelines_json}

Lacunas da tentativa anterior (vazio na primeira tentativa):
{gaps_json}
{available_files_section}
Produza um ExecutionPlan cujos steps endereçam diretamente as diretrizes acima.
"""

PLANNING_USER_WITH_GUIDANCE = """\
Tópico, objetivos e restrições de escopo da pesquisa:
{objective_json}

Etapa atual: {stage}
Tentativa: {attempt} / 5

Diretrizes desta etapa (driver principal do plano):
{stage_guidelines_json}

Lacunas da tentativa anterior:
{gaps_json}
{available_files_section}
ORIENTAÇÃO DO PESQUISADOR (prioridade máxima):
{user_guidance}

Produza um ExecutionPlan que incorpore diretamente a orientação acima,
endereçando as lacunas e seguindo as diretrizes da etapa.
"""


def build_planning_messages(
    objective: ResearchObjective,
    stage: str,
    attempt: int,
    gaps: list[str],
    skill_registry_summary: list[dict[str, Any]],
    user_guidance: str | None = None,
    stage_guidelines: list[str] | None = None,
    available_files: list[str] | None = None,
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
        available_files: Absolute paths to files the planner may read (e.g. PDFs
            in attachments/). When provided, injected into the prompt so the LLM
            can generate pdf_reader steps with the correct file_path parameters.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    effective_guidelines: list[str] = stage_guidelines or objective.get("success_metrics", [])

    if stage_guidelines:
        objective_for_prompt: dict[str, Any] = _stage_context(objective)
    else:
        objective_for_prompt = dict(objective)

    objective_json = json.dumps(objective_for_prompt, ensure_ascii=False, indent=2)

    if available_files:
        available_files_section = (
            "\nArquivos disponíveis em attachments/ "
            "(use pdf_reader com o parâmetro file_path para lê-los — NÃO use article_search nem web_search):\n"
            + json.dumps(available_files, ensure_ascii=False, indent=2)
            + "\n"
        )
    else:
        available_files_section = ""

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
            available_files_section=available_files_section,
            user_guidance=user_guidance,
        )
    else:
        user_content = PLANNING_USER.format(
            objective_json=objective_json,
            stage=stage,
            attempt=attempt,
            stage_guidelines_json=json.dumps(effective_guidelines, ensure_ascii=False, indent=2),
            gaps_json=json.dumps(gaps, ensure_ascii=False, indent=2),
            available_files_section=available_files_section,
        )
    messages = [{"role": "user", "content": user_content}]
    return system, messages
