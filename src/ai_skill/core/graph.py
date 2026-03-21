"""LangGraph StateGraph definition for the Phase 1 research pipeline.

Graph topology (Phase 1):

    START → initiate → align_charter →[interrupt_before CHECKPOINT 1]→
    literature_search → literature_review → evaluate →
        ├─ converged  → deliver → END
        ├─ attempt<5  → plan   → execute → evaluate  (loop)
        └─ attempt==5 → request_support → plan

Checkpoints (interrupt_before) pause the graph, persist state, and wait
for a human response before the next node executes. In LangGraph, these
are configured via ``interrupt_before`` on ``compile()``.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph

from ai_skill.core.nodes import (
    align_charter,
    deliver,
    evaluate,
    execute,
    initiate,
    plan,
    request_support,
    route_after_evaluate,
)
from ai_skill.core.state import ResearchState

logger = logging.getLogger(__name__)


def build_research_graph() -> "CompiledGraph":  # type: ignore[name-defined]
    """Build and compile the Phase 1 LangGraph research pipeline.

    Returns:
        A compiled LangGraph ``CompiledStateGraph`` ready for invocation.

    Notes:
        ``interrupt_before=["align_charter"]`` causes the graph to pause after
        ``initiate`` and before ``align_charter`` so the user can review and
        approve the Research Charter before the pipeline proceeds.

        In Phase 2+ additional ``interrupt_before`` nodes will be added for
        each checkpoint (literature review, methodology, etc.).
    """
    builder: StateGraph = StateGraph(ResearchState)

    # --- Nodes -----------------------------------------------------------
    builder.add_node("initiate", initiate)
    builder.add_node("align_charter", align_charter)
    builder.add_node("plan", plan)
    builder.add_node("execute", execute)
    builder.add_node("evaluate", evaluate)
    builder.add_node("deliver", deliver)
    builder.add_node("request_support", request_support)

    # --- Edges -----------------------------------------------------------
    builder.add_edge(START, "initiate")
    builder.add_edge("initiate", "align_charter")
    builder.add_edge("align_charter", "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "evaluate")

    # Conditional routing after evaluate
    builder.add_conditional_edges(
        "evaluate",
        route_after_evaluate,
        {
            "deliver": "deliver",
            "plan": "plan",
            "request_support": "request_support",
        },
    )

    builder.add_edge("request_support", "plan")
    builder.add_edge("deliver", END)

    # --- Compile ---------------------------------------------------------
    # interrupt_before="align_charter" pauses for CHECKPOINT 1 (Research Charter review)
    compiled = builder.compile(interrupt_before=["align_charter"])
    return compiled


def get_graph_mermaid() -> str:
    """Return the Mermaid diagram string for the Phase 1 research graph.

    Returns:
        A Mermaid-format string describing the graph topology.
    """
    graph = build_research_graph()
    return graph.get_graph().draw_mermaid()
