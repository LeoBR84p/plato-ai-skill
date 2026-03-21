"""Research workspace management.

Handles the filesystem layout for a single research project:
  <workspace_dir>/
    research-state.yaml   — serialised ResearchState snapshot
    research-log.md       — append-only audit log
    findings.md           — current literature findings report
    .cache/               — skill output cache (sha256-keyed JSON files)

Atomic writes (write-to-temp + rename) prevent corruption on crash.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ResearchWorkspace:
    """Manages a single research project workspace on the filesystem.

    Args:
        path: Directory for this research project. Created if it does not exist.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._cache_dir = self.path / ".cache"
        self._state_file = self.path / "research-state.yaml"
        self._log_file = self.path / "research-log.md"
        self._findings_file = self.path / "findings.md"
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Directory / initialisation
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        """Create workspace and cache directories if they do not exist."""
        self.path.mkdir(parents=True, exist_ok=True)
        self._cache_dir.mkdir(exist_ok=True)

    def initialise(self, topic: str) -> None:
        """Write the initial log header and findings stub.

        Args:
            topic: The research topic for this workspace.
        """
        if not self._log_file.exists():
            header = (
                f"# Research Log\n\n"
                f"**Topic**: {topic}  \n"
                f"**Started**: {_utcnow()}  \n\n"
                f"---\n\n"
            )
            self._atomic_write(self._log_file, header)

        if not self._findings_file.exists():
            self._atomic_write(
                self._findings_file,
                f"# Findings\n\n*Research in progress — topic: {topic}*\n",
            )

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def save_state(self, state: dict[str, Any]) -> None:
        """Persist ResearchState to research-state.yaml atomically.

        Args:
            state: The research state dict to serialise.
        """
        serialisable = _make_serialisable(state)
        content = yaml.dump(serialisable, allow_unicode=True, sort_keys=False)
        self._atomic_write(self._state_file, content)

    def load_state(self) -> dict[str, Any] | None:
        """Load a previously persisted ResearchState.

        Returns:
            The state dict, or None if no state file exists.
        """
        if not self._state_file.exists():
            return None
        try:
            text = self._state_file.read_text(encoding="utf-8")
            return yaml.safe_load(text) or {}
        except Exception as exc:
            logger.error("Failed to load state from %s: %s", self._state_file, exc)
            return None

    # ------------------------------------------------------------------
    # Findings
    # ------------------------------------------------------------------

    def write_findings(self, content: str) -> None:
        """Overwrite findings.md with the provided Markdown content.

        Args:
            content: Full Markdown text of the findings report.
        """
        self._atomic_write(self._findings_file, content)

    def read_findings(self) -> str:
        """Read current findings.md.

        Returns:
            The findings Markdown text, or empty string if file absent.
        """
        if not self._findings_file.exists():
            return ""
        return self._findings_file.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def log(self, message: str, level: str = "INFO") -> None:
        """Append a timestamped entry to research-log.md.

        Args:
            message: The log message to append.
            level: Log level label (INFO, WARNING, ERROR).
        """
        entry = f"**{_utcnow()}** [{level}] {message}\n\n"
        with self._log_file.open("a", encoding="utf-8") as fh:
            fh.write(entry)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def cache_get(self, key: str) -> dict[str, Any] | None:
        """Retrieve a cached skill output by its cache key.

        Args:
            key: The sha256-derived cache key.

        Returns:
            The cached dict, or None if not present.
        """
        cache_file = self._cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        try:
            text = cache_file.read_text(encoding="utf-8")
            return json.loads(text)
        except Exception as exc:
            logger.debug("Cache miss (read error) for key %s: %s", key, exc)
            return None

    def cache_set(self, key: str, value: dict[str, Any]) -> None:
        """Store a skill output in the cache.

        Args:
            key: The sha256-derived cache key.
            value: The value to store (must be JSON-serialisable).
        """
        cache_file = self._cache_dir / f"{key}.json"
        try:
            content = json.dumps(value, ensure_ascii=False, indent=2)
            self._atomic_write(cache_file, content)
        except Exception as exc:
            logger.warning("Failed to write cache for key %s: %s", key, exc)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def findings_path(self) -> Path:
        """Absolute path to findings.md."""
        return self._findings_file

    @property
    def log_path(self) -> Path:
        """Absolute path to research-log.md."""
        return self._log_file

    @property
    def state_path(self) -> Path:
        """Absolute path to research-state.yaml."""
        return self._state_file

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _atomic_write(self, target: Path, content: str) -> None:
        """Write content to target path atomically using a temp file + rename.

        Args:
            target: Destination file path.
            content: Text content to write.
        """
        dir_ = target.parent
        fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_serialisable(obj: Any) -> Any:
    """Recursively convert an object to a YAML/JSON-safe structure.

    Args:
        obj: Any Python value.

    Returns:
        A structure composed only of dicts, lists, strings, ints, floats, bools,
        and None.
    """
    if isinstance(obj, dict):
        return {str(k): _make_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Pydantic models, dataclasses, enums, BaseMessage, etc.
    if hasattr(obj, "model_dump"):
        return _make_serialisable(obj.model_dump())
    if hasattr(obj, "__dict__"):
        return _make_serialisable(obj.__dict__)
    return str(obj)
