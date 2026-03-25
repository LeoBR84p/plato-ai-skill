"""LangGraph StateGraph definitions — one per Checkpoint.

CP1 Graph topology (Research Charter):

    START → initiate → align_charter → [interrupt_before review_charter]
          → review_charter ──┬── charter_approved=True  → END
                             └── charter_approved=False → align_charter (loop)

CP2 Graph topology (Literature Review):

    START → cp2_router ──┬── (fresh)       → plan → execute → evaluate
                         └── (correction)  → refine_literature

    evaluate ──┬── converged   → compile_literature → verify_literature
               ├── attempt<5   → plan
               └── attempt==5  → request_support → plan

    verify_literature → deliver_literature
                      → [interrupt_before review_literature — CHECKPOINT 2]

    review_literature ──┬── literature_approved=True  → END
                        └── literature_approved=False → recheck_sources
                                                        → refine_literature
                                                        → verify_literature
                                                        → deliver_literature
                                                        → [interrupt again]

CP3 Graph topology (Research Design — 4-phase pipeline):

    START → cp3_router ──┬── (fresh)       → read_attachments (Phase 1, once)
                         └── (correction)  → refine_design

    Phase 1 — read_attachments: reads all PDFs in attachments/ (no planner, no retry)
    Phase 2 — ideate_design:    proposes the full research design (CP1+CP2+PDF context)
    Phase 3 — review_frameworks: critiques against OR/PMBOK/PRISMA/PRISMA-trAIce/FAIR/ABNT/ISO
    Phase 4 — evaluate_objectives: scores each section against CP1 objectives

    evaluate_objectives ──┬── converged   → deliver_design
                          ├── attempt<5   → ideate_design  (retry; preserved sections carried over)
                          └── attempt==5  → request_support → ideate_design

    deliver_design → [interrupt_before review_design — CHECKPOINT 3]

    review_design ──┬── design_approved=True  → END
                    └── design_approved=False → refine_design
                                               → deliver_design → [interrupt again]

Checkpoints (interrupt_before) pause the graph, persist state, and wait
for a human response before the next node executes.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from ai_skill.core.nodes import (
    align_charter,
    compile_literature,
    cp2_router,
    cp3_router,
    cp4_router,
    deliver_collection_guide,
    deliver_design,
    deliver_literature,
    draft_collection_guide,
    evaluate,
    evaluate_collection_objectives,
    evaluate_objectives,
    execute,
    ideate_design,
    initiate,
    plan,
    read_attachments,
    recheck_sources,
    refine_collection_guide,
    refine_design,
    refine_literature,
    request_support,
    review_charter,
    review_collection_guide,
    review_collection_standards,
    review_design,
    review_frameworks,
    review_literature,
    route_after_evaluate,
    route_after_evaluate_collection,
    route_after_evaluate_objectives,
    route_after_review_charter,
    route_after_review_collection,
    route_after_review_design,
    route_after_review_literature,
    route_cp2_start,
    route_cp3_start,
    route_cp4_start,
    verify_literature,
)
from ai_skill.core.state import ResearchState

logger = logging.getLogger(__name__)


def build_cp1_graph() -> "CompiledGraph":  # type: ignore[name-defined]
    """Build and compile the Checkpoint 1 (Research Charter) graph.

    Flow:
        START → initiate → align_charter → [interrupt] → review_charter
              → align_charter (reject loop) | END (approve)

    Returns:
        Compiled LangGraph StateGraph for CP1.
    """
    builder: StateGraph = StateGraph(ResearchState)

    builder.add_node("initiate", initiate)
    builder.add_node("align_charter", align_charter)
    builder.add_node("review_charter", review_charter)

    builder.add_edge(START, "initiate")
    builder.add_edge("initiate", "align_charter")
    builder.add_edge("align_charter", "review_charter")

    builder.add_conditional_edges(
        "review_charter",
        route_after_review_charter,
        {"align_charter": "align_charter", "END": END},
    )

    return builder.compile(interrupt_before=["review_charter"])


def build_cp2_graph() -> "CompiledGraph":  # type: ignore[name-defined]
    """Build and compile the Checkpoint 2 (Literature Review) graph.

    Flow:
        START → cp2_router ──┬── plan (fresh start)
                             └── refine_literature (correction cycle)
        plan → execute → evaluate → compile_literature → verify_literature
             → deliver_literature → [interrupt] → review_literature
             → END (approve) | refine_literature (corrections loop)

    Returns:
        Compiled LangGraph StateGraph for CP2.
    """
    builder: StateGraph = StateGraph(ResearchState)

    builder.add_node("cp2_router", cp2_router)
    builder.add_node("plan", plan)
    builder.add_node("execute", execute)
    builder.add_node("evaluate", evaluate)
    builder.add_node("request_support", request_support)
    builder.add_node("compile_literature", compile_literature)
    builder.add_node("verify_literature", verify_literature)
    builder.add_node("deliver_literature", deliver_literature)
    builder.add_node("review_literature", review_literature)
    builder.add_node("recheck_sources", recheck_sources)
    builder.add_node("refine_literature", refine_literature)

    builder.add_edge(START, "cp2_router")
    builder.add_conditional_edges(
        "cp2_router",
        route_cp2_start,
        {"plan": "plan", "refine_literature": "refine_literature"},
    )

    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "evaluate")
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "compile_literature": "compile_literature",
            "plan": "plan",
            "request_support": "request_support",
        },
    )
    builder.add_edge("request_support", "plan")

    builder.add_edge("compile_literature", "verify_literature")
    builder.add_edge("verify_literature", "deliver_literature")
    builder.add_edge("deliver_literature", "review_literature")

    builder.add_conditional_edges(
        "review_literature",
        route_after_review_literature,
        {"END": END, "recheck_sources": "recheck_sources"},
    )
    builder.add_edge("recheck_sources", "refine_literature")
    builder.add_edge("refine_literature", "verify_literature")

    return builder.compile(interrupt_before=["review_literature", "request_support"])


def build_cp3_graph() -> "CompiledGraph":  # type: ignore[name-defined]
    """Build and compile the Checkpoint 3 (Research Design) graph.

    4-phase pipeline:
        Phase 1 — read_attachments: reads all PDFs in attachments/ (once, no retry)
        Phase 2 — ideate_design:    proposes the full design from CP1+CP2+PDF context
        Phase 3 — review_frameworks: critiques against OR/PMBOK/PRISMA/FAIR/ABNT/ISO
        Phase 4 — evaluate_objectives: scores each section against CP1 objectives

    Loop: on non-convergence, evaluate_objectives → ideate_design (preserving
    high-scoring sections). After max retries: request_support → ideate_design.
    Researcher correction cycle: cp3_router → refine_design.

    Returns:
        Compiled LangGraph StateGraph for CP3.
    """
    builder: StateGraph = StateGraph(ResearchState)

    builder.add_node("cp3_router",           cp3_router)
    builder.add_node("read_attachments",     read_attachments)
    builder.add_node("ideate_design",        ideate_design)
    builder.add_node("review_frameworks",    review_frameworks)
    builder.add_node("evaluate_objectives",  evaluate_objectives)
    builder.add_node("request_support",      request_support)
    builder.add_node("deliver_design",       deliver_design)
    builder.add_node("review_design",        review_design)
    builder.add_node("refine_design",        refine_design)

    builder.add_edge(START, "cp3_router")
    builder.add_conditional_edges(
        "cp3_router",
        route_cp3_start,
        {"read_attachments": "read_attachments", "refine_design": "refine_design"},
    )

    # Phase 1 → 2 → 3 → 4 (linear, no retry within phases)
    builder.add_edge("read_attachments",  "ideate_design")
    builder.add_edge("ideate_design",     "review_frameworks")
    builder.add_edge("review_frameworks", "evaluate_objectives")

    # Phase 4 routing: converged → deliver | retry → ideate | exhausted → support
    builder.add_conditional_edges(
        "evaluate_objectives",
        route_after_evaluate_objectives,
        {
            "deliver_design":  "deliver_design",
            "ideate_design":   "ideate_design",
            "request_support": "request_support",
        },
    )
    builder.add_edge("request_support", "ideate_design")

    # Delivery and researcher review
    builder.add_edge("deliver_design", "review_design")
    builder.add_conditional_edges(
        "review_design",
        route_after_review_design,
        {"END": END, "refine_design": "refine_design"},
    )
    builder.add_edge("refine_design", "deliver_design")

    return builder.compile(interrupt_before=["review_design", "request_support"])


def build_cp4_graph() -> "CompiledGraph":  # type: ignore[name-defined]
    """Build and compile the Checkpoint 4 (Data Collection Guide) graph.

    3-phase pipeline:
        Phase 1 — draft_collection_guide:      operationalises CP3 into 8-section guide
        Phase 2 — review_collection_standards: web critique: FAIR/ISO/PMBOK/PRISMA/ABNT
        Phase 3 — evaluate_collection_objectives: scores sections vs CP1 + CP3

    Loop: on non-convergence, evaluate_collection_objectives → draft_collection_guide
    (preserving high-scoring sections).  After max retries: request_support.
    Researcher correction cycle: cp4_router → refine_collection_guide.

    Returns:
        Compiled LangGraph StateGraph for CP4.
    """
    builder: StateGraph = StateGraph(ResearchState)

    builder.add_node("cp4_router",                    cp4_router)
    builder.add_node("draft_collection_guide",        draft_collection_guide)
    builder.add_node("review_collection_standards",   review_collection_standards)
    builder.add_node("evaluate_collection_objectives", evaluate_collection_objectives)
    builder.add_node("request_support",               request_support)
    builder.add_node("deliver_collection_guide",      deliver_collection_guide)
    builder.add_node("review_collection_guide",       review_collection_guide)
    builder.add_node("refine_collection_guide",       refine_collection_guide)

    builder.add_edge(START, "cp4_router")
    builder.add_conditional_edges(
        "cp4_router",
        route_cp4_start,
        {
            "draft_collection_guide":  "draft_collection_guide",
            "refine_collection_guide": "refine_collection_guide",
        },
    )

    # Phase 1 → 2 → 3 (linear, no retry within phases)
    builder.add_edge("draft_collection_guide",      "review_collection_standards")
    builder.add_edge("review_collection_standards", "evaluate_collection_objectives")

    # Phase 3 routing: converged → deliver | retry → draft | exhausted → support
    builder.add_conditional_edges(
        "evaluate_collection_objectives",
        route_after_evaluate_collection,
        {
            "deliver_collection_guide": "deliver_collection_guide",
            "draft_collection_guide":   "draft_collection_guide",
            "request_support":          "request_support",
        },
    )
    builder.add_edge("request_support", "draft_collection_guide")

    # Delivery and researcher review
    builder.add_edge("deliver_collection_guide", "review_collection_guide")
    builder.add_conditional_edges(
        "review_collection_guide",
        route_after_review_collection,
        {"END": END, "refine_collection_guide": "refine_collection_guide"},
    )
    builder.add_edge("refine_collection_guide", "deliver_collection_guide")

    return builder.compile(
        interrupt_before=["review_collection_guide", "request_support"]
    )


def get_graph_mermaid() -> str:
    """Return Mermaid diagrams for all CP graphs.

    Returns:
        Combined Mermaid-format string for CP1, CP2, and CP3 graphs.
    """
    cp1 = build_cp1_graph().get_graph().draw_mermaid()
    cp2 = build_cp2_graph().get_graph().draw_mermaid()
    cp3 = build_cp3_graph().get_graph().draw_mermaid()
    return (
        f"--- CP1 (Research Charter) ---\n{cp1}\n"
        f"--- CP2 (Literature Review) ---\n{cp2}\n"
        f"--- CP3 (Research Design) ---\n{cp3}"
    )
