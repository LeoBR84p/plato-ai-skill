"""Audit history logger — writes llm_history.yaml to the project workspace root.

Every LLM call and every skill execution is appended here so the researcher
can audit exactly what was sent/received at each step of the pipeline.

The file uses YAML multi-document format (entries separated by ``---``).
Read back with: ``list(yaml.safe_load_all(open("llm_history.yaml")))``

Every document starts with a ``_task`` key that identifies the entry in the
VS Code YAML outline and enables targeted search/grep.  The key follows the
pattern ``stage / group / detail``, for example:

  _task: literature_review / plan_start / tentativa_1
  _task: literature_review / plan / tentativa_1
  _task: literature_review / execute / tentativa_1 / step_3 / article_search
  _task: literature_review / evaluate / tentativa_1
  _task: research_charter / align_charter

Entry types:

  plan_start  — one per attempt; marks the beginning of a new execution plan
  llm_call    — one per LLM API call (planner, evaluator, charter, verify, etc.)
  skill_call  — one per skill execution within a plan step

Example plan_start:

  _task: literature_review / plan_start / tentativa_1
  timestamp: "2026-03-23T10:00:00"
  type: plan_start
  stage: "literature_review"
  attempt: 1
  step_count: 8
  estimated_cost: medium
  plan_rationale: "Realizar 4 buscas distintas..."

Example skill_call:

  _task: literature_review / execute / tentativa_1 / step_3 / article_search
  timestamp: "2026-03-23T10:00:05"
  type: skill_call
  stage: "literature_review"
  attempt: 1
  step_id: 3
  step_rationale: "Buscar artigos de sistemas multiagente..."
  skill: "article_search"
  request_sent:
    query: "multi-agent systems LLM"
  response_received:
    confidence: 0.9
    sources_count: 20
    error: null
    result_keys: ["papers", "total_found", "query"]
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Module-level path and stage — set before any pipeline node runs.
# Using simple globals is safe because LangGraph runs nodes sequentially.
_log_path: Path | None = None
_current_stage: str = ""
_current_node: str = ""


def configure(workspace_path: str) -> None:
    """Point the history logger at the correct project workspace.

    Must be called once before the first LLM/skill call in a node.  The
    ``workspace_path`` should be the ResearchWorkspace (.state/) directory;
    the log file is written one level up (the ProjectWorkspace root) so it
    is visible to the researcher alongside the checkpoint .docx files.

    Args:
        workspace_path: Absolute path to the .state/ directory.
    """
    global _log_path
    if not workspace_path:
        _log_path = None
        return
    project_root = Path(workspace_path).parent
    if (project_root / "workspace.yaml").exists():
        # Inside a ProjectWorkspace — log at the project root
        _log_path = project_root / "llm_history.yaml"
    else:
        # Standalone workspace — log inside it
        _log_path = Path(workspace_path) / "llm_history.yaml"


def set_current_stage(stage: str) -> None:
    """Set the current pipeline stage for history attribution.

    Called by _configure_history() in nodes.py before each LLM/skill call.

    Args:
        stage: PipelineStage value string (e.g. "literature_review").
    """
    global _current_stage
    _current_stage = stage


def set_current_node(name: str) -> None:
    """Set the graph node name for history attribution.

    Args:
        name: Node function name (e.g. "plan", "evaluate").
    """
    global _current_node
    _current_node = name


def _task_label(
    *,
    node: str = "",
    skill: str = "",
    attempt: int | None = None,
    step_id: int | None = None,
) -> str:
    """Compose the _task identifier string for a history entry.

    The label follows the pattern ``stage / group / detail`` so that VS Code
    YAML outline and plain-text search can distinguish entry types at a glance.

    Args:
        node:     LangGraph node name (used for llm_call and plan_start entries).
        skill:    Skill name (used for skill_call entries).
        attempt:  Zero-based attempt index (converted to 1-based for display).
        step_id:  Plan step index (only meaningful for skill_call entries).

    Returns:
        A slash-separated label string, e.g.
        ``"literature_review / execute / tentativa_1 / step_3 / article_search"``.
    """
    stage = _current_stage or "unknown"
    parts: list[str] = [stage]

    if node:
        parts.append(node)
    elif skill:
        parts.append("execute")

    if attempt is not None:
        parts.append(f"tentativa_{attempt + 1}")

    if step_id is not None:
        parts.append(f"step_{step_id}")

    if skill:
        parts.append(skill)

    return " / ".join(parts)


def _append(entry: dict[str, Any]) -> None:
    """Stream-append *entry* to the YAML log file as a new document.

    Uses YAML multi-document format (``---`` separator) for O(1) append
    instead of read-entire-file-and-rewrite which is O(n²).
    """
    if _log_path is None:
        return
    try:
        # Render just this entry as a single-item YAML document
        text = yaml.dump(entry, allow_unicode=True, sort_keys=False, default_flow_style=False)
        with open(_log_path, "a", encoding="utf-8") as fh:
            fh.write("---\n" + text)
    except Exception as exc:
        logger.debug("history._append failed: %s", exc)


def log_plan_start(
    *,
    attempt: int,
    step_count: int,
    estimated_cost: str,
    plan_rationale: str,
) -> None:
    """Record the start of a new execution plan (one per attempt).

    Acts as a chapter header in llm_history.yaml — search ``plan_start`` to
    jump between attempts.  Logged by plan() in nodes.py after generating
    the ExecutionPlan.

    Args:
        attempt:        Zero-based attempt index.
        step_count:     Total number of steps in the plan.
        estimated_cost: Cost estimate string from the planner ("low"/"medium"/"high").
        plan_rationale: Overall rationale for this plan.
    """
    _append({
        "_task": _task_label(node="plan_start", attempt=attempt),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": "plan_start",
        "stage": _current_stage,
        "attempt": attempt + 1,
        "step_count": step_count,
        "estimated_cost": estimated_cost,
        "plan_rationale": plan_rationale,
    })


def log_llm_call(
    *,
    system: str,
    messages: list[dict[str, str]],
    response: str,
    model: str,
    node: str = "",
) -> None:
    """Record one LLM interaction.

    Args:
        system: The system prompt sent.
        messages: The full messages list sent to the API.
        response: The assistant response text (or JSON string for structured calls).
        model: The model ID used.
        node: The graph node that triggered the call (e.g. "plan", "evaluate").
            When empty, falls back to the module-level _current_node.
    """
    effective_node = node or _current_node
    user_content = messages[-1]["content"] if messages else ""
    _append({
        "_task": _task_label(node=effective_node),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": "llm_call",
        "stage": _current_stage,
        "node": effective_node,
        "model": model,
        "prompt_sent": {
            "system": system,
            "user": user_content,
        },
        "response_received": response,
    })


def log_skill_call(
    *,
    skill: str,
    stage: str,
    attempt: int,
    step_id: int = -1,
    step_rationale: str = "",
    request_sent: dict[str, Any],
    response_received: dict[str, Any],
) -> None:
    """Record one skill execution.

    Args:
        skill: The skill name (e.g. "article_search").
        stage: Current pipeline stage name.
        attempt: The retry attempt index (0-based).
        step_id: Plan step index (0-based).  -1 when not available.
        step_rationale: The rationale for this step from the ExecutionPlan.
        request_sent: The parameters dict sent to the skill (sensitive keys
            like API keys should never appear here — they live in env vars).
        response_received: Summary of the skill output (confidence, sources,
            error, result keys).
    """
    entry: dict[str, Any] = {
        "_task": _task_label(skill=skill, attempt=attempt, step_id=step_id if step_id >= 0 else None),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": "skill_call",
        "stage": stage,
        "attempt": attempt + 1,
        "step_id": step_id if step_id >= 0 else None,
        "skill": skill,
        "request_sent": request_sent,
        "response_received": response_received,
    }
    if step_rationale:
        # Insert after step_id so rationale is visible near the top of the entry
        entry["step_rationale"] = step_rationale
    _append(entry)
