"""Prompt templates for the user alignment (Research Charter) node."""

from __future__ import annotations

import json

from ai_skill.core.state import ResearchObjective


CHARTER_DRAFT_SYSTEM = """\
You are an academic research advisor. Given a free-form research topic, draft a
structured Research Charter that will guide an 8-stage AI-assisted research pipeline.

The charter must contain:
1. **3-5 specific, measurable research goals** for the overall project.
2. **3-5 concrete success metrics** for the OVERALL project (quantifiable where possible).
3. **Scope constraints** implicit in the topic (time window, geography, domain, etc.).
4. **Methodology preference** (if inferable from the topic).
5. **Bibliography style**: default "abnt".
6. **Language**: default "pt-BR".
7. **Stage-specific guidelines** (stage_guidelines): a dict keyed by stage name with 4–8
   actionable directives each. These drive planning and evaluation for THAT stage only —
   independently of the overall success_metrics. Be specific to the research topic.

Stage keys and their focus areas:

  "literature_review"
    — Directives for the bibliographic research phase (CP2). Examples:
      thematic sub-areas to cover, minimum number of primary references (aim ≥ 40),
      target time window (e.g. last 10 years, with ≥ 30 % from the last 5),
      priority databases (arXiv, Semantic Scholar, Scopus, Web of Science, etc.),
      core search terms and Boolean combinations, PRISMA screening criteria,
      expected thematic structure of the review.

  "research_design"
    — Directives for the methodology and design phase (CP3). Examples:
      study type (experimental/observational/mixed), required instruments or datasets,
      hypotheses to test, validation criteria, ethical considerations.

  "data_collection_guide"
    — Directives for the data collection protocol (CP4). Examples:
      target datasets or populations, sample size requirements, data quality criteria,
      collection tools, reproducibility standards.

  "analysis_guide"
    — Directives for the analysis phase (CP5). Examples:
      statistical or computational techniques, software/libraries, significance thresholds,
      ablation study requirements, baseline comparisons.

  "results_interpretation"
    — Directives for interpreting and reporting results (CP6/7). Examples:
      comparison with prior literature, effect-size reporting standards,
      confidence-interval requirements, failure-mode analysis.

  "paper_composition"
    — Directives for drafting the final paper (CP8). Examples:
      target journal or conference, word-count limits, required sections,
      figures and tables specifications, co-authorship policy.

  "publication"
    — Directives for the publication phase (CP8+). Examples:
      target venue Qualis/Scopus level, open-access requirements, data availability
      statement, code/reproducibility checklist, pre-print policy.

Be specific and academic in tone. Avoid vague statements.
"""

CHARTER_DRAFT_USER = """\
Research topic: {topic}

Draft a Research Charter for this topic.
"""

CHARTER_REFINE_SYSTEM = """\
You are an academic research advisor applying surgical corrections to a
Research Charter that the researcher has already reviewed and partially
approved. The researcher marked only the parts they want changed; everything
else must be preserved VERBATIM — not rephrased, not improved, not
reorganised.

CRITICAL RULE — default is PRESERVE:
  Copy every field, sentence, and list item from the original exactly as-is,
  UNLESS it is directly targeted by one of the corrections below.
  Do NOT use this as an opportunity to rewrite, improve, or clean up unmarked
  content. Character-for-character fidelity to the original is required for
  all unmarked sections.

How to handle each correction type:
- **Comments** ("Comentários"): locate the passage the comment refers to and
  apply the stated instruction to that passage only.
- **Track changes — inserted text** ("Trechos inseridos"): splice the inserted
  text into the exact location indicated, changing nothing else around it.
- **Track changes — deleted text** ("Trechos removidos"): remove only those
  words; leave surrounding content intact.
- **Yellow highlight** ("Trechos em destaque amarelo"): rewrite ONLY the
  highlighted span; preserve everything before and after it unchanged.

When done, the output must be clean (no marks, comments, or highlights) and
differ from the original only where corrections explicitly required a change.
"""

CHARTER_REFINE_USER = """\
Research Charter approved by the researcher (treat as authoritative — do NOT
rewrite any part that is not explicitly targeted by a correction):
{charter_json}

Corrections to apply (touch only what is listed here):
---
{feedback}
---

Return the complete Research Charter with ONLY the listed corrections applied.
All unmarked content must be identical to the original.
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
