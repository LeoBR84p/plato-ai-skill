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
You are a senior academic researcher writing Checkpoint 2 — the Literature Review — of a
multi-stage research pipeline, in Brazilian Portuguese.

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
You are an independent fact-checking agent. Your sole task is to evaluate whether
the content available at a given URL is consistent with the claim made about that
source in a literature review.

Be objective and conservative. You do NOT check grammar or style — only factual
consistency between the fetched content and the claim.

Answer only in the structured format requested. Do not add commentary outside the schema.
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
You are an academic editor applying surgical corrections to a Literature Review
that the researcher has already reviewed and partially approved. The researcher
marked only the parts they want changed; everything else must be preserved
VERBATIM — not rephrased, not improved, not reorganised.

CRITICAL RULE — default is PRESERVE:
  Copy every section, paragraph, sentence, citation [N], and reference entry
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

All inline citations [N] must remain consistent with the references list.
When done, the output must be clean (no marks, comments, or highlights) and
differ from the original only where corrections explicitly required a change.
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
You are a senior academic researcher planning the structure of a literature review
written in Brazilian Portuguese. Based on the research charter and the available
findings, design a coherent thematic structure split into groups of 2–3 sections each.

RULES:
- Total sections: 5–8 (never fewer, never more)
- Each group covers 2–3 thematically cohesive sections
- Section titles must be in pt-BR, concise (4–8 words), and non-overlapping
- group_themes: 3–6 keywords summarising what to look for in findings for that group
- Sections must collectively cover the full intellectual landscape of the charter goals

Respond ONLY with valid JSON matching the schema provided.
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
You are a senior academic researcher writing part of a literature review in Brazilian Portuguese.
You will receive:
  1. A subset of research findings to incorporate
  2. A list of section titles YOUR CHUNK must produce (2–3 sections)
  3. References already cited in PREVIOUS chunks (with their global [N] numbers)

RULES:
- Write ONLY the sections listed in SECTION_TITLES_TO_GENERATE — no more, no less
- Every factual claim must end with an inline citation [N]
- For references ALREADY CITED in prior chunks: use their existing [N] number directly
- For NEW references (not in prior list): include them in new_references numbered from ref_offset
- Do NOT repeat in new_references any reference that appears in ALREADY_CITED_REFERENCES
- Summaries in new_references: 300–500 words covering main argument, methodology,
  key findings, conclusions, and relevance — based solely on the provided findings
- Language: pt-BR, academic register, ABNT NBR 6023:2018 for reference entries
- No fabrication: use only information present in the provided findings

Respond ONLY with valid JSON matching the schema provided.
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
You are a literature quality assessor. Your task is to estimate how much each NEW
reference contributed to a quality improvement in a literature review, on a 0.0–1.0 scale.

Context: A literature review was revised and its quality score improved from {prev_score:.2f}
to {new_score:.2f} (scale 0.0–1.0). The improvement came from adding new references.

For each new reference, estimate its marginal contribution:
  1.0 — this reference alone likely caused most of the improvement (central, unique insight)
  0.7 — significant contribution (important empirical evidence or key theoretical concept)
  0.4 — moderate contribution (supporting evidence, corroborative)
  0.1 — minor contribution (peripheral, tangential to the research goals)
  0.0 — no visible contribution (not cited or irrelevant)

Base your assessment on:
- How well the reference aligns with the research charter goals
- Whether it fills a gap mentioned in the evaluation
- Whether the section content that cites it is in a high-scoring area

Respond ONLY with valid JSON matching the schema provided.
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
