"""Prompt templates for the literature review (Checkpoint 2) nodes.

Three operations are supported:
- compile  — draft the full literature review from skill findings
- verify   — independent agent checks a single source URL against its claim
- refine   — apply researcher corrections from an edited .docx preview
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# COMPILE — draft literature review from findings
# ---------------------------------------------------------------------------

COMPILE_SYSTEM = """\
Você é um pesquisador acadêmico sênior redigindo o Checkpoint 2 — a Revisão Bibliográfica —
de um pipeline de pesquisa multi-estágio, em português brasileiro.

Context: Checkpoint 1 (Research Charter) has already been approved and defines the
research topic, goals, success metrics, and scope. You have access to its full text.
Your job is to produce a Literature Review that:
  • Directly supports and deepens the goals stated in CP1
  • Does NOT repeat or summarise the charter content — assume the reader already has CP1
  • Synthesises the academic sources found during the research phase
  • Organises knowledge thematically, showing how each body of literature relates to the goals

Requirements:
1. **Continuity with CP1**: Every section must explicitly connect to at least one goal or
   success metric from the Research Charter. Use phrases like "Em relação ao objetivo X…",
   "Para atender à métrica Y…". Do not restate the topic or goals — reference them briefly.
2. **Structure**: 3–6 thematic sections, each with a clear heading and a coherent narrative
   synthesising multiple sources.
3. **Inline citations**: Every factual claim must end with [N] where N is the reference index.
   Multiple citations allowed: [1][3][5].
4. **References list**: Number all sources sequentially [1], [2], … Include only sources
   cited in the text. Every entry must have a direct URL and the placeholder {ACCESS_DATE}.
   **URL priority** (use the first available for each paper):
   (a) ``doi`` field → format as ``https://doi.org/{doi}``
   (b) ``arxiv_id`` field → format as ``https://arxiv.org/abs/{arxiv_id}``
   (c) ``url`` field from the finding (may be a Semantic Scholar page)
   Never invent a URL. If none of (a)–(c) is available, omit the URL.
5. **ABNT NBR 6023:2018 format**:
   - Article: SOBRENOME, Nome. Título. **Periódico**, v. X, n. Y, p. ZZ–ZZ, ano. DOI/URL. {ACCESS_DATE}.
   - Webpage: ORGANIZAÇÃO. **Título**. Local: Editor, ano. Disponível em: <URL>. {ACCESS_DATE}.
6. **Language**: pt-BR, academic register throughout. No colloquialisms.
7. **No fabrication**: Use only information from the provided findings. Do not invent
   authors, years, or claims not supported by the sources.
8. **Consistency**: same citation style throughout; no mixed numbering.
9. **Reference summaries**: For every entry in the references list, write a `summary`
   of 300 to 500 words describing the article's main argument, methodology, key findings,
   conclusions, and relevance to the research objectives — based only on the provided
   findings. The conclusions of the article are mandatory and must be explicitly addressed.
"""

COMPILE_USER = """\
════════════════════════════════════════
CHECKPOINT 1 — Research Charter (aprovado)
════════════════════════════════════════
{charter_document_text}

════════════════════════════════════════
Findings da fase de pesquisa bibliográfica
════════════════════════════════════════
{findings_json}

════════════════════════════════════════
Produza agora o Checkpoint 2 — Revisão Bibliográfica completa, seguindo as instruções
do sistema. Não repita o conteúdo do charter. Conecte cada seção aos objetivos do CP1.
"""


def build_compile_messages(
    charter_document_text: str,
    findings: list[dict[str, Any]],
) -> tuple[str, list[dict[str, str]]]:
    """Build messages to compile a literature review from CP1 charter + findings.

    Args:
        charter_document_text: Full extracted text of CP1 [final].docx.
        findings: Skill outputs from the execute node.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    findings_json = json.dumps(findings, ensure_ascii=False, indent=2)
    user_content = COMPILE_USER.format(
        charter_document_text=charter_document_text or "(charter não disponível)",
        findings_json=findings_json,
    )
    return COMPILE_SYSTEM, [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# VERIFY — independent agent checks a source URL
# ---------------------------------------------------------------------------

VERIFY_SYSTEM = """\
Você é um agente independente de verificação de fatos. Sua única tarefa é avaliar
se o conteúdo disponível em uma URL é consistente com a afirmação feita sobre
essa fonte em uma revisão bibliográfica.

Seja objetivo e conservador. Você NÃO verifica gramática ou estilo — apenas
consistência factual entre o conteúdo obtido e a afirmação.

Responda SOMENTE com JSON válido no seguinte formato (sem texto adicional):
{
  "content_matches": true,
  "verification_note": "Uma frase explicando sua decisão."
}
"""

VERIFY_USER = """\
Reference number: [{reference_number}]
Reference title: {title}
Claim made in the review: {summary}

URL: {url}
Fetched content (first 3000 characters):
---
{fetched_content}
---

Evaluate:
- content_matches: true if the fetched content is consistent with the claim, false otherwise.
- verification_note: One sentence explaining your decision.
"""


def build_verify_messages(
    reference_number: int,
    title: str,
    summary: str,
    url: str,
    fetched_content: str,
) -> tuple[str, list[dict[str, str]]]:
    """Build messages for the verification agent to check one source.

    Args:
        reference_number: The [N] index of this source.
        title: Title of the reference.
        summary: The claim made about this source in the review.
        url: The URL to verify.
        fetched_content: Raw text fetched from the URL.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    user_content = VERIFY_USER.format(
        reference_number=reference_number,
        title=title,
        summary=summary,
        url=url,
        fetched_content=fetched_content[:3000] if fetched_content else "(não acessível)",
    )
    return VERIFY_SYSTEM, [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# REFINE — apply researcher corrections to the review
# ---------------------------------------------------------------------------

REFINE_SYSTEM = """\
Você é um editor acadêmico aplicando correções cirúrgicas a uma Revisão Bibliográfica
que o pesquisador já revisou e parcialmente aprovou. O pesquisador marcou apenas as
partes que deseja alterar; todo o restante deve ser preservado LITERALMENTE — sem
reformular, sem melhorar, sem reorganizar.

REGRA CRÍTICA — padrão é PRESERVAR:
  Copie cada seção, parágrafo, frase, citação [N] e entrada de referência do original
  exatamente como está, A MENOS QUE seja diretamente alvo de uma das correções abaixo.
  NÃO use isto como oportunidade para reescrever, melhorar ou polir conteúdo não marcado.
  Fidelidade caractere-por-caractere ao original é exigida para todas as seções não marcadas.

  Campos estruturados (referências, números de citação [N]) devem manter consistência
  com a lista de referências original.

Como lidar com cada tipo de correção:
- **Comentários** ("Comentários"): localize a passagem que o comentário referencia
  e aplique a instrução declarada apenas àquela passagem.
- **Track changes — texto inserido** ("Trechos inseridos"): insira o texto no
  local exato indicado, sem alterar nada ao redor.
- **Track changes — texto removido** ("Trechos removidos"): remova apenas essas
  palavras; mantenha o conteúdo ao redor intacto.
- **Destaque amarelo** ("Trechos em destaque amarelo"): reescreva APENAS o
  trecho destacado; preserve tudo antes e depois sem alteração.

Todas as citações inline [N] devem permanecer consistentes com a lista de referências.
Ao finalizar, o output deve estar limpo (sem marcas, comentários ou destaques) e
diferir do original apenas onde as correções explicitamente exigiram uma mudança.
"""

REFINE_USER = """\
Literature Review approved by the researcher (treat as authoritative — do NOT
rewrite any part that is not explicitly targeted by a correction):
{review_json}

Corrections to apply (touch only what is listed here):
---
{feedback}
---

Return the complete Literature Review with ONLY the listed corrections applied.
All unmarked content must be identical to the original. Keep the same section
structure unless corrections explicitly add or remove sections. Ensure all [N]
inline citations remain consistent with the references list.
"""


def build_refine_messages(
    review_doc: dict[str, Any],
    feedback: str,
) -> tuple[str, list[dict[str, str]]]:
    """Build messages to refine a literature review based on user feedback.

    Args:
        review_doc: The current LiteratureReviewDoc dict.
        feedback: Formatted correction instructions from the researcher's docx.

    Returns:
        Tuple of (system_prompt, messages_list).
    """
    user_content = REFINE_USER.format(
        review_json=json.dumps(review_doc, ensure_ascii=False, indent=2),
        feedback=feedback,
    )
    return REFINE_SYSTEM, [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# OUTLINE — plan section groups for chunked compilation
# ---------------------------------------------------------------------------

OUTLINE_SYSTEM = """\
Você é um pesquisador acadêmico sênior planejando a estrutura de uma revisão bibliográfica
em português brasileiro. Com base no research charter e nos findings disponíveis,
projete uma estrutura temática coerente dividida em grupos de 2–3 seções cada.

REGRAS:
- Total de seções: 5–8 (nunca menos, nunca mais)
- Cada grupo cobre 2–3 seções tematicamente coesas
- Títulos das seções em pt-BR, concisos (4–8 palavras), sem sobreposição
- group_themes: 3–6 palavras-chave resumindo o que buscar nos findings para aquele grupo
- As seções devem cobrir coletivamente toda a paisagem intelectual dos objetivos do charter
- Se os findings não suportam 5 seções distintas, agrupe sub-temas relacionados

Responda SOMENTE com JSON válido. Exemplo de estrutura:
{
  "groups": [
    {
      "titles": ["Fundamentos Teóricos da Área X", "Abordagens Computacionais Recentes"],
      "themes": ["teoria", "modelos", "framework", "computacional"]
    }
  ]
}
"""

OUTLINE_USER = """\
════════════════════════════════════════
CHECKPOINT 1 — Research Charter (resumo)
════════════════════════════════════════
{charter_document_text}

════════════════════════════════════════
Findings disponíveis (títulos para contexto)
════════════════════════════════════════
{finding_titles}

════════════════════════════════════════
Design a estrutura da revisão bibliográfica em {num_chunks} grupo(s) de seções,
onde cada grupo será compilado independentemente. Grupos devem ser temáticamente
distintos e não sobrepostos.
"""


def build_outline_messages(
    charter_document_text: str,
    finding_titles: list[str],
    num_chunks: int,
) -> tuple[str, list[dict[str, str]]]:
    """Build messages for the outline phase of chunked compilation."""
    titles_text = "\n".join(f"- {t}" for t in finding_titles[:60])
    user_content = OUTLINE_USER.format(
        charter_document_text=(charter_document_text or "")[:4000],
        finding_titles=titles_text,
        num_chunks=num_chunks,
    )
    return OUTLINE_SYSTEM, [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# COMPILE_CHUNK — generate 2–3 sections from a subset of findings
# ---------------------------------------------------------------------------

COMPILE_CHUNK_SYSTEM = """\
Você é um pesquisador acadêmico sênior redigindo parte de uma revisão bibliográfica em
português brasileiro. Você receberá:
  1. Um subconjunto de findings de pesquisa para incorporar
  2. Uma lista de títulos de seção que SEU CHUNK deve produzir (2–3 seções)
  3. Referências já citadas em chunks ANTERIORES (com seus números [N] globais)

REGRAS:
- Escreva APENAS as seções listadas em SECTION_TITLES_TO_GENERATE — nem mais, nem menos
- Toda afirmação factual deve terminar com citação inline [N]
- Para referências JÁ CITADAS em chunks anteriores: use o número [N] existente diretamente
- Para referências NOVAS (não na lista anterior): inclua em new_references numeradas a partir de ref_offset
- NÃO repita em new_references nenhuma referência que apareça em ALREADY_CITED_REFERENCES
- Limite máximo de 15 novas referências por chunk para manter a coesão
- Summaries em new_references: 300–500 palavras cobrindo argumento principal, metodologia,
  achados-chave, conclusões e relevância — baseado exclusivamente nos findings fornecidos
- Idioma: pt-BR, registro acadêmico, ABNT NBR 6023:2018 para entradas de referência
- Sem fabricação: use apenas informações presentes nos findings fornecidos

Responda SOMENTE com JSON válido conforme o schema fornecido.
"""

COMPILE_CHUNK_USER = """\
════════════════════════════════════════
CHECKPOINT 1 — Research Charter (resumo)
════════════════════════════════════════
{charter_document_text}

════════════════════════════════════════
SECTION_TITLES_TO_GENERATE (escreva EXATAMENTE estas seções, nesta ordem)
════════════════════════════════════════
{section_titles}

════════════════════════════════════════
ALREADY_CITED_REFERENCES (use estes números ao citar — NÃO inclua em new_references)
════════════════════════════════════════
{already_cited}

════════════════════════════════════════
Findings para incorporar neste chunk (suas novas referências começam em [{ref_offset}])
════════════════════════════════════════
{findings_json}

Gere as seções listadas acima. Cite fontes já incluídas pelos seus [N] existentes.
Novas fontes começam em [{ref_offset}]. Inclua em new_references SOMENTE referências novas.
"""


def build_chunk_messages(
    charter_document_text: str,
    section_titles: list[str],
    findings: list[dict],
    already_cited: list[dict],
    ref_offset: int,
) -> tuple[str, list[dict[str, str]]]:
    """Build messages for one section chunk in chunked compilation."""
    import json as _json

    titles_text = "\n".join(f"- {t}" for t in section_titles)
    cited_text = "\n".join(
        f"[{r.get('reference_number', '?')}] {r.get('authors', '')} ({r.get('year', '')}). {r.get('title', '')}."
        for r in already_cited
    ) or "(nenhuma — este é o primeiro chunk)"

    user_content = COMPILE_CHUNK_USER.format(
        charter_document_text=(charter_document_text or "")[:4000],
        section_titles=titles_text,
        already_cited=cited_text,
        ref_offset=ref_offset,
        findings_json=_json.dumps(findings, ensure_ascii=False, indent=2),
    )
    return COMPILE_CHUNK_SYSTEM, [{"role": "user", "content": user_content}]


# ---------------------------------------------------------------------------
# REFERENCE_CONTRIBUTION — LLM estimates marginal contribution of new references
# ---------------------------------------------------------------------------

REFERENCE_CONTRIBUTION_SYSTEM = """\
Você é um avaliador de qualidade bibliográfica. Sua tarefa é estimar quanto cada NOVA
referência contribuiu para a melhoria de qualidade de uma revisão bibliográfica, em uma
escala de 0.0–1.0.

Contexto: Uma revisão bibliográfica foi revisada e seu score de qualidade melhorou de
{prev_score:.2f} para {new_score:.2f} (escala 0.0–1.0). A melhoria veio da adição de
novas referências.

Para cada nova referência, estime sua contribuição marginal:
  1.0 — esta referência sozinha provavelmente causou a maior parte da melhoria (insight central e único)
  0.7 — contribuição significativa (evidência empírica importante ou conceito teórico chave)
  0.4 — contribuição moderada (evidência de suporte, corroborativa)
  0.1 — contribuição menor (periférica, tangencial aos objetivos da pesquisa)
  0.0 — sem contribuição visível (não citada ou irrelevante)

Baseie sua avaliação em:
- Quão bem a referência se alinha com os objetivos do research charter
- Se ela preenche uma lacuna mencionada na avaliação anterior
- Se o conteúdo da seção que a cita está em uma área de alto score

Responda SOMENTE com JSON válido conforme o schema fornecido.
"""

REFERENCE_CONTRIBUTION_USER = """\
════════════════════════════════════════
Research Charter goals (for relevance assessment)
════════════════════════════════════════
{charter_goals}

════════════════════════════════════════
Evaluation gaps from PREVIOUS iteration (what was missing)
════════════════════════════════════════
{prev_gaps}

════════════════════════════════════════
NEW references added in this iteration (estimate each one's marginal contribution)
════════════════════════════════════════
{new_references_json}

For each reference, return its URL and an estimated contribution score (0.0–1.0).
"""


def build_contribution_messages(
    charter_goals: list[str],
    prev_gaps: list[str],
    new_references: list[dict],
    prev_score: float,
    new_score: float,
) -> tuple[str, list[dict[str, str]]]:
    """Build messages for the reference contribution estimation call."""
    import json as _json

    system = REFERENCE_CONTRIBUTION_SYSTEM.format(
        prev_score=prev_score,
        new_score=new_score,
    )
    user_content = REFERENCE_CONTRIBUTION_USER.format(
        charter_goals="\n".join(f"- {g}" for g in charter_goals),
        prev_gaps="\n".join(f"- {g}" for g in prev_gaps) or "(none recorded)",
        new_references_json=_json.dumps(
            [{"url": r.get("url", ""), "title": r.get("title", ""), "summary": (r.get("summary") or "")[:200]}
             for r in new_references],
            ensure_ascii=False, indent=2
        ),
    )
    return system, [{"role": "user", "content": user_content}]
