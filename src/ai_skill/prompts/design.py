"""Prompt templates for the research design (Checkpoint 3) nodes.

Two operations are supported:
- compile  — draft the full Research Design document from skill findings
- refine   — apply researcher corrections from an edited .docx preview
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# COMPILE — draft research design from CP1 charter + CP2 review + findings
# ---------------------------------------------------------------------------

COMPILE_DESIGN_SYSTEM = """\
You are a senior academic methodologist writing Checkpoint 3 — the Research Design — of a
multi-stage research pipeline, in Brazilian Portuguese.

Context: Checkpoint 1 (Research Charter) and Checkpoint 2 (Literature Review) have already
been approved. Your job is to produce a Research Design document that:
  • Is grounded in the goals and objectives stated in CP1
  • Builds on the methodological landscape surfaced in CP2
  • Follows Operational Research (OR) Stage 2 — model construction and hypothesis formulation
  • Applies PMBOK Planning process group principles: scope, schedule, risk, and quality
  • Complies with ISO 9001:2015 acceptance criteria for verifiable quality gates
  • Ensures data sources follow FAIR principles (Findable, Accessible, Interoperable, Reusable)

MANDATORY STRUCTURE — produce exactly these 9 sections in order:
  1. Método de Pesquisa — type, justification grounded in CP1 + CP2, alignment with objectives
  2. Hipóteses e Questões de Pesquisa — testable hypotheses H1…Hn, falsifiable format
  3. Variáveis — table of independent (VI), dependent (VD), and confounding variables with
     precise operational definitions and measurement scales (nominal/ordinal/interval/ratio)
  4. Métricas e Metas de Dados — KPIs, acceptance thresholds, statistical power, sample size
     justification (power analysis or qualitative saturation criterion)
  5. Fontes de Dados — primary and secondary sources with quality criteria and FAIR compliance
  6. Instrumentos e Protocolo de Coleta — instruments, tools, step-by-step collection protocol
  7. Critérios de Validade e Confiabilidade — internal/external validity, reliability,
     replication protocol, and mitigation strategies for threats
  8. Considerações Éticas e Conformidade — CEP/IRB requirements, LGPD, informed consent,
     conflicts of interest
  9. Cronograma Metodológico — PMBOK-aligned milestones: CP3 (design) → CP4 (coleta) →
     CP5 (análise) with time estimates

Requirements:
1. **No fabrication**: Use only what is grounded in CP1 goals, CP2 literature, and the
   methodology research findings. Do NOT invent numbers, populations, or instrument details.
2. **Anti-alucinação**: This is a METHODOLOGY document. Do NOT write results, datasets,
   or conclusions. Do NOT draft the article. Do NOT execute collection or analysis.
3. **Language**: pt-BR, academic register. Use technical notation where appropriate
   (e.g. H₀/H₁ format for hypotheses, Likert scale notation, LaTeX-style formulas in text).
4. **Structured fields**: In addition to the prose sections, populate all structured fields
   (study_type, research_paradigm, epistemological_stance, hypotheses, variables, etc.)
   consistently with the section content.
5. **Methodology grounding**: Every method choice must cite a precedent from CP2 literature
   or a recognised methodological authority.
6. **Reporting standard**: Identify the applicable reporting guideline for the study type
   (CONSORT for RCTs, STROBE for observational, PRISMA for systematic reviews, etc.).
"""

COMPILE_DESIGN_USER = """\
════════════════════════════════════════
CHECKPOINT 1 — Research Charter (aprovado)
════════════════════════════════════════
{charter_document_text}

════════════════════════════════════════
CHECKPOINT 2 — Revisão Bibliográfica (aprovada)
════════════════════════════════════════
{literature_document_text}

════════════════════════════════════════
CP3 Context (handoff do CP2)
════════════════════════════════════════
{cp3_context_json}

════════════════════════════════════════
Findings da pesquisa metodológica
════════════════════════════════════════
{findings_json}

════════════════════════════════════════
Produza agora o Checkpoint 3 — Design de Pesquisa completo com as 9 seções obrigatórias,
seguindo as instruções do sistema. Fundamente cada escolha metodológica na literatura
revisada (CP2) e nos objetivos de pesquisa (CP1).
"""


def build_compile_design_messages(
    charter_document_text: str,
    literature_document_text: str,
    cp3_context: dict[str, Any],
    findings: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Build messages to compile a Research Design from CP1 + CP2 + findings.

    Args:
        charter_document_text: Full extracted text of CP1 [final].docx.
        literature_document_text: Full extracted text of CP2 [final].docx.
        cp3_context: Handoff dict from CP2 with topic, goals, scope, themes, etc.
        findings: Skill outputs from the execute node (methodology research).

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    findings_json = json.dumps(findings, ensure_ascii=False, indent=2)
    cp3_context_json = json.dumps(cp3_context, ensure_ascii=False, indent=2)
    user_content = COMPILE_DESIGN_USER.format(
        charter_document_text=charter_document_text or "(charter não disponível)",
        literature_document_text=literature_document_text or "(revisão bibliográfica não disponível)",
        cp3_context_json=cp3_context_json,
        findings_json=findings_json,
    )
    return COMPILE_DESIGN_SYSTEM, [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# REFINE — apply researcher corrections to the research design
# ---------------------------------------------------------------------------

REFINE_DESIGN_SYSTEM = """\
You are an academic methodologist applying surgical corrections to a Research Design
document that the researcher has already reviewed and partially approved. The researcher
marked only the parts they want changed; everything else must be preserved
VERBATIM — not rephrased, not improved, not reorganised.

CRITICAL RULE — default is PRESERVE:
  Copy every section, paragraph, sentence, table, formula, and structured field
  from the original exactly as-is, UNLESS it is directly targeted by one of
  the corrections below. Do NOT use this as an opportunity to rewrite, improve,
  or polish unmarked content. Character-for-character fidelity to the original
  is required for all unmarked sections.

How to handle each correction type:
- **Comments** ("Comentários"): locate the passage the comment refers to and
  apply the stated instruction to that passage only.
- **Track changes — inserted text** ("Trechos inseridos"): splice the inserted
  text into the exact location indicated, changing nothing else around it.
- **Track changes — deleted text** ("Trechos removidos"): remove only those
  words; leave surrounding content intact.
- **Yellow highlight** ("Trechos em destaque amarelo"): rewrite ONLY the
  highlighted span; preserve everything before and after it unchanged.

When correcting variables, hypotheses, or metrics: ensure the structured fields
(hypotheses list, variables list, metrics_and_kpis list, etc.) remain consistent
with the prose in the corresponding section.

When done, the output must be clean (no marks, comments, or highlights) and
differ from the original only where corrections explicitly required a change.
"""

REFINE_DESIGN_USER = """\
Research Design aprovado pelo pesquisador (trate como autoritativo — NÃO
reescreva nenhuma parte que não seja explicitamente alvo de uma correção):
{design_json}

Correções a aplicar (toque apenas no que está listado aqui):
---
{feedback}
---

Retorne o Design de Pesquisa completo com APENAS as correções listadas aplicadas.
Todo o conteúdo não marcado deve ser idêntico ao original. Mantenha a mesma
estrutura de seções, a menos que as correções adicionem ou removam seções
explicitamente. Garanta que os campos estruturados (hypotheses, variables,
metrics_and_kpis, data_sources, etc.) permaneçam consistentes com as seções.
"""


def build_refine_design_messages(
    design_doc: dict[str, Any],
    feedback: str,
) -> tuple[str, list[dict[str, str]]]:
    """Build messages to refine a Research Design based on user feedback.

    Args:
        design_doc: The current ResearchDesignDoc dict.
        feedback: Formatted correction instructions from the researcher's docx.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    user_content = REFINE_DESIGN_USER.format(
        design_json=json.dumps(design_doc, ensure_ascii=False, indent=2),
        feedback=feedback,
    )
    return REFINE_DESIGN_SYSTEM, [{"role": "user", "content": user_content}]
