"""Audit history logger — writes llm_history.yaml to the project workspace root.

Every LLM call and every skill execution is appended here so the researcher
can audit exactly what was sent/received at each step of the pipeline.

The file uses YAML multi-document format (entries separated by ``---``).
Read back with: ``list(yaml.safe_load_all(open("llm_history.yaml")))``

Entry structure:

  timestamp: "2026-03-22T10:30:00"
  type: llm_call
  model: "claude-sonnet-4-6"
  stage: "literature_review"
  node: "plan"
  prompt_sent:
    system: "..."
    user: "..."
  response_received: "..."

  ---

  timestamp: "2026-03-22T10:31:00"
  type: skill_call
  skill: "article_search"
  stage: "literature_review"
  attempt: 1
  request_sent:
    query: "anomaly detection HBOS"
    max_results: 20
  response_received:
    confidence: 0.85
    sources_count: 10
    sources:
      - "https://arxiv.org/abs/1234"
    error: null
    result_keys: ["papers", "total"]
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
    user_content = messages[-1]["content"] if messages else ""
    _append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": "llm_call",
        "model": model,
        "stage": _current_stage,
        "node": node or _current_node,
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
    request_sent: dict[str, Any],
    response_received: dict[str, Any],
) -> None:
    """Record one skill execution.

    Args:
        skill: The skill name (e.g. "article_search").
        stage: Current pipeline stage name.
        attempt: The retry attempt index.
        request_sent: The parameters dict sent to the skill (sensitive keys
            like API keys should never appear here — they live in env vars).
        response_received: Summary of the skill output (confidence, sources,
            error, result keys).
    """
    _append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "type": "skill_call",
        "skill": skill,
        "stage": stage,
        "attempt": attempt,
        "request_sent": request_sent,
        "response_received": response_received,
    })
