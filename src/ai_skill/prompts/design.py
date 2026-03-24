"""Prompt templates for the research design (Checkpoint 3) nodes.

Four-phase CP3 pipeline:
- ideate   — Phase 2: propose a full research design from CP1 + CP2 + PDF context
- review_frameworks — Phase 3: critique and adjust the proposal against OR/PMBOK/PRISMA/FAIR/ABNT/ISO
- evaluate_objectives — Phase 4: score the design against CP1 research objectives
- refine   — apply researcher corrections from an edited .docx preview

Legacy compile prompt kept for reference but superseded by the ideate prompt.
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


# ---------------------------------------------------------------------------
# IDEATE — Phase 2: propose the research design (ideation, not retrieval)
# ---------------------------------------------------------------------------

IDEATE_DESIGN_SYSTEM = """\
Você é um metodólogo acadêmico sênior responsável pelo Checkpoint 3 — Design de Pesquisa —
de um pipeline de pesquisa em Pesquisa Operacional, redigido em português brasileiro.

Os Checkpoints 1 (Research Charter) e 2 (Revisão Bibliográfica) já foram aprovados.
Os PDFs de referência foram lidos na Fase 1 e os resumos estão disponíveis como contexto.

SUA TAREFA É PROPOR — não resumir.
Você deve tomar DECISÕES metodológicas fundamentadas no CP1 + CP2 e justificá-las.
O design pode e deve ir além do que está literalmente escrito nos PDFs:
  • Formule hipóteses originais adequadas ao problema de pesquisa do CP1.
  • Escolha o método mais adequado mesmo que nenhum PDF o mencione explicitamente.
  • Defina variáveis, escalas e KPIs com precisão operacional.
  • Projete um protocolo de coleta de dados viável para o contexto descrito.

ESTRUTURA OBRIGATÓRIA — produza exatamente estas 9 seções na ordem:
  1. Método de Pesquisa — tipo, justificativa fundamentada em CP1+CP2, alinhamento com objetivos
  2. Hipóteses e Questões de Pesquisa — ≥2 hipóteses testáveis H1, H2 … em formato falsificável (H₀/H₁)
  3. Variáveis — tabela de VI, VD e confundidoras: nome, tipo, operacionalização, escala de medição
  4. Métricas e Metas de Dados — KPIs, limiares de aceitação, poder estatístico, tamanho amostral mínimo
  5. Fontes de Dados — primárias e secundárias com critérios FAIR (Findable, Accessible, Interoperable, Reusable)
  6. Instrumentos e Protocolo de Coleta — instrumentos, ferramentas, protocolo passo a passo
  7. Critérios de Validade e Confiabilidade — ameaças à validade interna/externa, estratégias de mitigação
  8. Considerações Éticas e Conformidade — CEP/IRB, LGPD, consentimento informado
  9. Cronograma Metodológico — marcos PMBOK: CP3 (design) → CP4 (coleta) → CP5 (análise)

REGRAS:
- Linguagem: pt-BR, registro acadêmico. Use notação técnica (H₀/H₁, escala Likert, fórmulas).
- Fundamentação: cada escolha metodológica deve citar ao menos um precedente da revisão (CP2)
  ou autoridade metodológica reconhecida (ex.: Yin 2018 para estudo de caso).
- Anti-alucinação: NÃO escreva resultados, datasets ou conclusões. NÃO execute coleta nem análise.
- Seções preservadas: seções listadas em PRESERVED_SECTIONS devem ser copiadas VERBATIM — sem
  reformulação, sem melhoria, sem reordenação.
- Lacunas: enderece diretamente cada item em GAPS_FROM_PREVIOUS_EVALUATION.
"""

IDEATE_DESIGN_USER = """\
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
Contexto metodológico extraído dos PDFs de referência (Fase 1)
════════════════════════════════════════
{pdf_context_json}

════════════════════════════════════════
PRESERVED_SECTIONS (copie verbatim — NÃO altere)
════════════════════════════════════════
{preserved_sections_json}

════════════════════════════════════════
GAPS_FROM_PREVIOUS_EVALUATION (endereça diretamente)
════════════════════════════════════════
{gaps_json}

════════════════════════════════════════
Produza agora o Checkpoint 3 — Design de Pesquisa completo com as 9 seções obrigatórias.
Proponha decisões metodológicas originais fundamentadas em CP1+CP2. Preserve as seções
listadas em PRESERVED_SECTIONS sem qualquer alteração. Endereça cada lacuna em GAPS.
"""


def build_ideate_design_messages(
    charter_document_text: str,
    literature_document_text: str,
    cp3_context: dict[str, Any],
    pdf_context: list[dict[str, Any]],
    preserved_sections: dict[str, Any],
    gaps: list[str],
) -> tuple[str, list[dict[str, str]]]:
    """Build messages to ideate (propose) a Research Design.

    Args:
        charter_document_text: Full text of approved CP1 .docx.
        literature_document_text: Full text of approved CP2 .docx.
        cp3_context: Handoff dict from CP2 (topic, goals, scope, themes).
        pdf_context: Compact summaries from Phase 1 pdf_reader findings.
        preserved_sections: Sections that scored >= 0.85 in previous evaluation;
            must be copied verbatim. Empty on first iteration.
        gaps: Gap strings from previous evaluate_objectives call. Empty on first call.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    user_content = IDEATE_DESIGN_USER.format(
        charter_document_text=charter_document_text or "(charter não disponível)",
        literature_document_text=literature_document_text or "(revisão bibliográfica não disponível)",
        cp3_context_json=json.dumps(cp3_context, ensure_ascii=False, indent=2),
        pdf_context_json=json.dumps(pdf_context, ensure_ascii=False, indent=2),
        preserved_sections_json=(
            json.dumps(preserved_sections, ensure_ascii=False, indent=2)
            if preserved_sections
            else "(primeira iteração — nenhuma seção preservada)"
        ),
        gaps_json=(
            json.dumps(gaps, ensure_ascii=False, indent=2)
            if gaps
            else "(primeira iteração — nenhuma lacuna anterior)"
        ),
    )
    return IDEATE_DESIGN_SYSTEM, [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# REVIEW_FRAMEWORKS — Phase 3: critique against OR/PMBOK/PRISMA/FAIR/ABNT/ISO
# ---------------------------------------------------------------------------

REVIEW_FRAMEWORKS_SYSTEM = """\
Você é um especialista em qualidade metodológica de pesquisa acadêmica. Sua função é revisar
criticamente um Design de Pesquisa redigido em pt-BR contra os seguintes referenciais formais:

1. PESQUISA OPERACIONAL (OR) — Etapa 2 (model construction): o design deve incluir formulação
   de hipóteses falsificáveis, identificação de variáveis com operacionalização precisa, e
   justificativa do método em relação ao problema modelado.

2. PMBOK (7ª edição) — Domínio de Planejamento: escopo, cronograma, qualidade e risco devem
   ser endereçados. O cronograma metodológico deve ter marcos mensuráveis.

3. PRISMA 2020 — aplicável quando o método inclui revisão sistemática ou meta-análise:
   protocolo de busca, critérios PICOS, fluxograma de seleção, e avaliação de risco de viés.

4. PRISMA-trAIce 2025 — extensão do PRISMA para estudos envolvendo IA/ML:
   {prisma_traice_context}
   Verifique: documentação de modelos de IA usados, rastreabilidade de decisões algorítmicas,
   viés em datasets sintéticos ou aumentados com IA, reprodutibilidade computacional.

5. FAIR Data Principles — dados devem ser: Findable (identificador persistente), Accessible
   (protocolo aberto), Interoperable (formato padrão), Reusable (licença clara, metadados ricos).

6. SJR / Scimago — para publicação em periódico Q1/Q2: o design deve justificar originalidade
   e contribuição incremental ao estado da arte; instrumentos devem ter validade e confiabilidade
   comprovadas em literatura.

7. ABNT NBR — normas brasileiras aplicáveis: NBR 14724 (trabalhos acadêmicos), NBR 6023
   (referências), NBR 10520 (citações). Verificar consistência de citações e formatação.

8. ISO 9001:2015 — critérios de aceitação verificáveis: KPIs com limiares numéricos,
   protocolo de coleta com critérios de qualidade por etapa, rastreabilidade de dados.

PROCESSO DE REVISÃO:
Para cada referencial, avalie se o design atende, atende parcialmente, ou não atende.
Para cada não-conformidade, produza uma correção cirúrgica (altere APENAS o trecho afetado).
Não reescreva seções inteiras quando uma correção pontual é suficiente.
Não adicione conteúdo que contradiga os objetivos de pesquisa do CP1.
"""

REVIEW_FRAMEWORKS_USER = """\
════════════════════════════════════════
Design de Pesquisa a revisar (produzido na Fase 2)
════════════════════════════════════════
{design_json}

════════════════════════════════════════
Objetivos de pesquisa (CP1 — para não contradizer)
════════════════════════════════════════
{charter_summary}

════════════════════════════════════════
Informações adicionais obtidas por busca web sobre os referenciais
════════════════════════════════════════
{framework_search_results}

════════════════════════════════════════
Revise o design contra os 8 referenciais listados no sistema.
Retorne o design COMPLETO com apenas as correções necessárias aplicadas.
Seções que já atendem aos referenciais devem ser copiadas verbatim.
"""


def build_review_frameworks_messages(
    design_doc: dict[str, Any],
    charter_summary: str,
    framework_search_results: str,
    prisma_traice_context: str,
) -> tuple[str, list[dict[str, str]]]:
    """Build messages to review the design against formal methodology frameworks.

    Args:
        design_doc: Current ResearchDesignDoc dict from Phase 2.
        charter_summary: Brief CP1 objectives summary (topic + goals).
        framework_search_results: Text gathered from web searches about the standards.
        prisma_traice_context: Web-fetched content about PRISMA-trAIce 2025 specifics.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    system = REVIEW_FRAMEWORKS_SYSTEM.format(
        prisma_traice_context=prisma_traice_context or
        "extensão do PRISMA 2020 para estudos com IA generativa e LLMs publicada em 2025; "
        "requer transparência sobre uso de modelos de linguagem em qualquer fase do estudo."
    )
    user_content = REVIEW_FRAMEWORKS_USER.format(
        design_json=json.dumps(design_doc, ensure_ascii=False, indent=2),
        charter_summary=charter_summary or "(resumo do charter não disponível)",
        framework_search_results=framework_search_results or "(busca web não retornou resultados)",
    )
    return system, [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# EVALUATE_OBJECTIVES — Phase 4: score design against CP1 research objectives
# ---------------------------------------------------------------------------

EVALUATE_OBJECTIVES_SYSTEM = """\
Você é um avaliador de qualidade de pesquisa acadêmica. Avalie um Design de Pesquisa
(Checkpoint 3) contra os objetivos de pesquisa definidos no Checkpoint 1 (Research Charter).

CRITÉRIO DE AVALIAÇÃO:
Para cada uma das 9 seções obrigatórias do design, atribua um score de 0.0 a 1.0:
  0.0–0.3 — seção ausente, vazia ou totalmente desconectada dos objetivos do CP1
  0.4–0.6 — seção presente mas incompleta ou apenas parcialmente alinhada
  0.7–0.84 — seção boa, atende à maioria dos critérios, lacunas menores
  0.85–1.0 — seção excelente, totalmente alinhada com CP1, sem lacunas relevantes

SEÇÕES A AVALIAR (9 seções obrigatórias):
  1. Método de Pesquisa
  2. Hipóteses e Questões de Pesquisa
  3. Variáveis
  4. Métricas e Metas de Dados
  5. Fontes de Dados
  6. Instrumentos e Protocolo de Coleta
  7. Critérios de Validade e Confiabilidade
  8. Considerações Éticas e Conformidade
  9. Cronograma Metodológico

Para cada seção com score < 0.85, liste as lacunas específicas em relação aos objetivos do CP1.
O score total é a média aritmética dos 9 scores de seção.
Convergência: score total >= {threshold}.
"""

EVALUATE_OBJECTIVES_USER = """\
════════════════════════════════════════
CHECKPOINT 1 — Objetivos de pesquisa (autoritativo)
════════════════════════════════════════
Tópico: {topic}
Objetivos: {goals_json}
Restrições de escopo: {scope_json}

════════════════════════════════════════
Design de Pesquisa a avaliar (Checkpoint 3)
════════════════════════════════════════
{design_json}

════════════════════════════════════════
Avalie cada uma das 9 seções do design contra os objetivos de pesquisa acima.
Para cada seção com score < 0.85, explique as lacunas específicas.
Compute o score total como média das 9 seções.
"""


def build_evaluate_objectives_messages(
    objective: dict[str, Any],
    design_doc: dict[str, Any],
    threshold: float,
) -> tuple[str, list[dict[str, str]]]:
    """Build messages to evaluate the design document against CP1 research objectives.

    Args:
        objective: ResearchObjective (or cp3_context) with topic, goals, scope_constraints.
        design_doc: Current ResearchDesignDoc dict to evaluate.
        threshold: Convergence threshold (e.g. 0.75).

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    system = EVALUATE_OBJECTIVES_SYSTEM.format(threshold=threshold)
    user_content = EVALUATE_OBJECTIVES_USER.format(
        topic=objective.get("topic") or "(não especificado)",
        goals_json=json.dumps(objective.get("goals") or [], ensure_ascii=False, indent=2),
        scope_json=json.dumps(objective.get("scope_constraints") or [], ensure_ascii=False, indent=2),
        design_json=json.dumps(design_doc, ensure_ascii=False, indent=2),
    )
    return system, [{"role": "user", "content": user_content}]
