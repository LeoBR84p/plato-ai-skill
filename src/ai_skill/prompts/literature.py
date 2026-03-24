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
