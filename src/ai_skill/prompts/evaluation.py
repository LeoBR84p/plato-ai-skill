"""Prompt templates for the EvaluatorAgent node."""

from __future__ import annotations

import json
from typing import Any

from ai_skill.core.state import ResearchObjective, SkillOutput


# Used when no stage_guidelines are available (fallback to overall project metrics).
EVALUATION_SYSTEM_GLOBAL = """\
Você é um avaliador rigoroso de pesquisa acadêmica. Sua tarefa é avaliar o quão
bem os resultados coletados satisfazem os critérios de sucesso definidos.

Para cada critério, atribua uma pontuação entre 0.0 e 1.0:
- 1.0: Critério totalmente satisfeito de forma convincente.
- 0.7-0.9: Critério majoritariamente satisfeito com lacunas menores.
- 0.4-0.6: Critério parcialmente satisfeito; lacunas significativas permanecem.
- 0.0-0.3: Critério mal satisfeito ou não satisfeito.

Seja estrito. A pesquisa acadêmica exige rigor.
"""

# Used when stage_guidelines are present.
# The objective_json is stripped of success_metrics so the LLM cannot
# accidentally score against the global project criteria.
EVALUATION_SYSTEM_STAGE = """\
Você é um avaliador rigoroso de pesquisa acadêmica avaliando os resultados de
UMA ÚNICA etapa do pipeline — não o projeto de pesquisa completo.

SEU ÚNICO TRABALHO: pontuar o quão bem os resultados coletados satisfazem as
diretrizes da etapa listadas na mensagem do usuário em "CRITÉRIOS DESTA ETAPA".

REGRAS ESTRITAS:
1. Avalie SOMENTE os critérios listados em "CRITÉRIOS DESTA ETAPA". Não invente
   critérios adicionais.
2. Trate o tópico de pesquisa apenas como contexto para entender relevância.
   O tópico NÃO define os critérios de avaliação.
3. Cada critério recebe uma pontuação independente de 0.0 a 1.0:
   1.0 = totalmente satisfeito | 0.7-0.9 = majoritariamente | 0.4-0.6 = parcial | 0.0-0.3 = ausente
4. Seja estrito e baseado em evidências. Cite resultados específicos para cada pontuação.
5. Forneça gaps concretos e acionáveis para cada critério não satisfeito.

CRITÉRIOS PROIBIDOS — NUNCA avalie contra estes, independentemente do tópico:
- Implementação de código, software, repositórios, GitHub
- Execução de experimentos, datasets, resultados numéricos
- Redação de artigo, submissão, publicação, revisão por pares
- Execução de coleta ou análise de dados
- Qualquer critério de sucesso global do projeto não listado em "CRITÉRIOS DESTA ETAPA"
"""

EVALUATION_USER_GLOBAL = """\
Objetivo de pesquisa:
{objective_json}

Critérios de sucesso:
{criteria_json}

Outputs das skills coletados (resumo):
{findings_json}

Para cada critério, forneça: pontuação (0.0-1.0), justificativa e lacunas específicas.
"""

EVALUATION_USER_STAGE = """\
Tópico de pesquisa (contexto apenas — NÃO define os critérios de avaliação):
{topic}

CRITÉRIOS DESTA ETAPA (estes são os ÚNICOS critérios a avaliar):
{criteria_json}

Outputs coletados pelas skills (resumo):
{findings_json}

Para cada critério acima, informe: pontuação (0.0-1.0), justificativa e lacunas.

LEMBRETE: avalie APENAS os critérios listados acima. Não mencione implementação
de código, publicação de artigos, repositórios GitHub, execução de experimentos
ou qualquer outro critério que não esteja explicitamente listado.
"""


def build_evaluation_messages(
    objective: ResearchObjective,
    findings: list[SkillOutput],
    threshold: float = 0.75,  # noqa: ARG001
    stage_guidelines: list[str] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Build the system + user messages for the evaluation node.

    When *stage_guidelines* is provided the evaluation is driven exclusively
    by those stage-specific criteria.  The objective's ``success_metrics`` are
    physically absent from the prompt so the LLM cannot score against them.

    When *stage_guidelines* is absent (e.g. CP1 self-check) the overall
    ``success_metrics`` are used as a fallback.

    NOTE: The LLM is asked only for per-metric scores and gaps. ``total_score``
    and ``converged`` are computed deterministically in nodes.py from the
    per-metric scores to eliminate LLM arithmetic errors and premature
    convergence declarations.

    Args:
        objective: The confirmed research objective.
        findings: List of SkillOutput from the current execution attempt.
        threshold: Convergence threshold (used only for reference in logs, not in prompt).
        stage_guidelines: Stage-specific directives.  When supplied these are
            the ONLY scoring criteria; overall success_metrics are suppressed.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    # Cap findings to the 80 most recent to avoid exceeding the 1M token limit.
    # When findings_current is used (recommended), this is typically 5-15 items.
    capped_findings = findings[-80:] if len(findings) > 80 else findings

    findings_summary: list[dict[str, Any]] = []
    for f in capped_findings:
        skill_name = f.get("skill_name", "unknown")
        entry: dict[str, Any] = {
            "skill": skill_name,
            "confidence": f.get("confidence", 0.0),
            "error": f.get("error"),
            "sources_count": len(f.get("sources", [])),
            "sources": f.get("sources", [])[:5],
        }
        result = f.get("result") or {}
        if result:
            if skill_name == "article_search":
                # Include only counts + top-5 titles — never full abstract lists
                papers = result.get("papers", [])
                entry["result_summary"] = {
                    "total_found": result.get("total_found", len(papers)),
                    "sources_queried": result.get("sources_queried", []),
                    "query": result.get("query", ""),
                    "top_titles": [p.get("title", "") for p in papers[:5]],
                }
            elif skill_name == "content_summarizer":
                summary = result.get("summary", "") or ""
                entry["result_summary"] = {
                    "summary": summary[:400],
                    "source_url": result.get("source_url", ""),
                }
            else:
                # Generic: keep first 5 keys, truncate any string values > 300 chars
                compact: dict[str, Any] = {}
                for k, v in list(result.items())[:5]:
                    if isinstance(v, str) and len(v) > 300:
                        compact[k] = v[:300] + "…"
                    elif isinstance(v, list) and len(v) > 10:
                        compact[k] = v[:10]
                    else:
                        compact[k] = v
                entry["result_summary"] = compact
        findings_summary.append(entry)

    findings_json = json.dumps(findings_summary, ensure_ascii=False, indent=2)
    criteria_json = json.dumps(
        stage_guidelines if stage_guidelines else objective.get("success_metrics", []),
        ensure_ascii=False,
        indent=2,
    )

    if stage_guidelines:
        # Stage-scoped evaluation: suppress success_metrics from the prompt
        system = EVALUATION_SYSTEM_STAGE
        user_content = EVALUATION_USER_STAGE.format(
            topic=objective.get("topic", ""),
            criteria_json=criteria_json,
            findings_json=findings_json,
        )
    else:
        # Global fallback (CP1 or stages without specific guidelines)
        system = EVALUATION_SYSTEM_GLOBAL
        user_content = EVALUATION_USER_GLOBAL.format(
            objective_json=json.dumps(dict(objective), ensure_ascii=False, indent=2),
            criteria_json=criteria_json,
            findings_json=findings_json,
        )

    return system, [{"role": "user", "content": user_content}]
