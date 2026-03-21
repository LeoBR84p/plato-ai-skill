"""Pipeline stage definitions mapping to Operational Research and PMBOK frameworks.

References:
    - Churchman, Ackoff & Arnoff (1957): 7-step OR methodology
    - PMBOK 8th Edition (PMI, 2025): 5 process groups
    - ISO 21001:2018: Educational organizations management systems
"""

from enum import Enum


class PipelineStage(str, Enum):
    """The 8 stages of the academic research pipeline.

    Each stage maps to an Operational Research step and a PMBOK process group.
    Stages with a HANDOFF require the user to execute work before the agent
    can continue. Stages with a CHECKPOINT pause the graph via interrupt_before.
    """

    INITIATE = "initiate"
    """Stage 0: Project initialization and workspace setup."""

    RESEARCH_CHARTER = "research_charter"
    """Stage 1 (OR-1: Problem Definition / PMBOK Initiating).

    Agent drafts Research Charter with objectives, goals, and success metrics.
    CHECKPOINT 1: user reviews and approves before proceeding.
    """

    LITERATURE_REVIEW = "literature_review"
    """Stage 2 (OR-3: Data Collection / PMBOK Planning+Executing).

    Multi-layer search across Exa.ai, Google CSA, Firecrawl, arXiv,
    Semantic Scholar. Fact-checking and cross-reference of findings.
    CHECKPOINT 2: user curates bibliography.
    """

    RESEARCH_DESIGN = "research_design"
    """Stage 3 (OR-2: Model Construction / PMBOK Planning).

    Identifies the research method, hypotheses, variables, and instruments.
    CHECKPOINT 3: user approves methodology.
    """

    DATA_COLLECTION_GUIDE = "data_collection_guide"
    """Stage 4 (OR-3: Data Collection — guidance only / PMBOK Executing).

    Agent produces collection protocol, instruments, and checklist.
    HANDOFF 4: user executes collection. Agent NEVER fabricates data.
    CHECKPOINT 4: user returns collected data.
    """

    ANALYSIS_GUIDE = "analysis_guide"
    """Stage 5 (OR-4: Solution Formulation / PMBOK Executing).

    Agent produces step-by-step method, formulas (LaTeX), software guide.
    CHECKPOINT 5: user reviews formulas before executing.
    HANDOFF 6: user executes analysis and returns results.
    """

    RESULTS_INTERPRETATION = "results_interpretation"
    """Stage 6 (OR-5: Model Validation / PMBOK Monitoring & Controlling).

    Agent drafts Results section using data provided by the user.
    CHECKPOINT 7: user reviews via cowork (track changes + comments).
    """

    PAPER_COMPOSITION = "paper_composition"
    """Stage 7 (OR-6: Controls / PMBOK Executing+Monitoring).

    Agent drafts all paper sections. ReviewAgent evaluates quality.
    CHECKPOINT 8: section-by-section review via cowork skill.
    """

    PUBLICATION = "publication"
    """Stage 8 (OR-7: Implementation / PMBOK Closing).

    ABNT formatting, open science compliance check, export (.docx, PDF),
    git research(paper) commit.
    """


class ResearchStatus(str, Enum):
    """Status of the research project at any point in time."""

    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    CHECKPOINT_PENDING = "checkpoint_pending"
    HANDOFF_PENDING = "handoff_pending"
    CONVERGED = "converged"
    FAILED = "failed"
    COMPLETED = "completed"


# Mapping: PipelineStage → OR step number (1-7)
OR_STEP_MAP: dict[PipelineStage, int] = {
    PipelineStage.INITIATE: 1,
    PipelineStage.RESEARCH_CHARTER: 1,
    PipelineStage.LITERATURE_REVIEW: 3,
    PipelineStage.RESEARCH_DESIGN: 2,
    PipelineStage.DATA_COLLECTION_GUIDE: 3,
    PipelineStage.ANALYSIS_GUIDE: 4,
    PipelineStage.RESULTS_INTERPRETATION: 5,
    PipelineStage.PAPER_COMPOSITION: 6,
    PipelineStage.PUBLICATION: 7,
}

# Mapping: PipelineStage → PMBOK process group name
PMBOK_PROCESS_MAP: dict[PipelineStage, str] = {
    PipelineStage.INITIATE: "Initiating",
    PipelineStage.RESEARCH_CHARTER: "Initiating",
    PipelineStage.LITERATURE_REVIEW: "Planning / Executing",
    PipelineStage.RESEARCH_DESIGN: "Planning",
    PipelineStage.DATA_COLLECTION_GUIDE: "Executing",
    PipelineStage.ANALYSIS_GUIDE: "Executing",
    PipelineStage.RESULTS_INTERPRETATION: "Monitoring & Controlling",
    PipelineStage.PAPER_COMPOSITION: "Executing / Monitoring",
    PipelineStage.PUBLICATION: "Closing",
}
