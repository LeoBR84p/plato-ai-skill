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
                        └── literature_approved=False → refine_literature
                                                        → verify_literature
                                                        → deliver_literature
                                                        → [interrupt again]

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
    deliver_literature,
    evaluate,
    execute,
    initiate,
    plan,
    refine_literature,
    request_support,
    review_charter,
    review_literature,
    route_after_evaluate,
    route_after_review_charter,
    route_after_review_literature,
    route_cp2_start,
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
        {"END": END, "refine_literature": "refine_literature"},
    )
    builder.add_edge("refine_literature", "verify_literature")

    return builder.compile(interrupt_before=["review_literature"])


def get_graph_mermaid() -> str:
    """Return Mermaid diagrams for both CP graphs.

    Returns:
        Combined Mermaid-format string for CP1 and CP2 graphs.
    """
    cp1 = build_cp1_graph().get_graph().draw_mermaid()
    cp2 = build_cp2_graph().get_graph().draw_mermaid()
    return f"--- CP1 (Research Charter) ---\n{cp1}\n--- CP2 (Literature Review) ---\n{cp2}"
