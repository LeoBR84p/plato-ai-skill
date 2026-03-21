"""Load API keys from the plato/keys/ directory into environment variables.

Key files live outside the ai_skill package (in /root/leobr/plato/keys/) so
they are never included in the package distribution or accidentally committed.

Convention:
    <VARIABLE_NAME>.key  →  content is the key value (stripped of whitespace)

Example files:
    ANTHROPIC_API_KEY.key
    GOOGLE_API_KEY.key
    GOOGLE_CSE_ID.key
    FIRECRAWL_API_KEY.key
    LANGCHAIN_API_KEY.key

Usage:
    from ai_skill.core.key_loader import load_keys
    load_keys()          # idempotent; skips vars already set in the environment

The loader is called automatically at package import time (__init__.py) so
every module benefits without explicit setup.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolved at import time; robust even when the CWD changes at runtime.
_KEYS_DIR = Path(__file__).parents[4] / "keys"


def load_keys(keys_dir: Path | None = None, *, overwrite: bool = False) -> dict[str, str]:
    """Read *.key files and inject their values into os.environ.

    Only sets a variable when it is not already present in the environment
    (unless ``overwrite=True``). This lets Docker / CI inject keys via the
    normal environment mechanism without being silently overridden.

    Args:
        keys_dir: Directory containing the .key files. Defaults to
            ``<plato_root>/keys/``.
        overwrite: If True, overwrite existing environment variables.

    Returns:
        Dict mapping variable name → value for every key that was loaded
        (already-set variables that were skipped are not included).
    """
    target_dir = keys_dir or _KEYS_DIR

    if not target_dir.is_dir():
        logger.debug("keys_dir '%s' not found — skipping key loading.", target_dir)
        return {}

    loaded: dict[str, str] = {}

    for key_file in sorted(target_dir.glob("*.key")):
        var_name = key_file.stem  # ANTHROPIC_API_KEY.key → ANTHROPIC_API_KEY
        if not overwrite and var_name in os.environ:
            logger.debug("Env var '%s' already set — skipping file.", var_name)
            continue

        try:
            value = key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Could not read key file '%s': %s", key_file, exc)
            continue

        if not value:
            logger.warning("Key file '%s' is empty — skipping.", key_file)
            continue

        os.environ[var_name] = value
        loaded[var_name] = value
        logger.debug("Loaded env var '%s' from file.", var_name)

    # Auto-detect Google service account JSON in the keys directory.
    # Sets GOOGLE_APPLICATION_CREDENTIALS to its path if the file exists
    # and the variable is not already configured.
    _auto_set_google_credentials(target_dir, overwrite=overwrite)

    if loaded:
        logger.info("key_loader: loaded %d key(s): %s", len(loaded), ", ".join(loaded))

    return loaded


def _auto_set_google_credentials(keys_dir: Path, *, overwrite: bool = False) -> None:
    """Set GOOGLE_APPLICATION_CREDENTIALS to the first service account JSON found.

    Looks for any ``*.json`` file in keys_dir whose name contains common
    service account naming patterns (credentials, service_account, or any
    ``<project>-<hash>.json`` GCP download format).

    Args:
        keys_dir: Directory to scan for credential JSON files.
        overwrite: If True, overwrite existing env var.
    """
    if not overwrite and os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return

    if not keys_dir.is_dir():
        return

    candidates = sorted(keys_dir.glob("*.json"))
    if not candidates:
        return

    # Prefer files with recognisable patterns; fall back to any JSON
    preferred = [
        f for f in candidates
        if any(kw in f.name.lower() for kw in ("credential", "service_account", "service-account"))
    ] or candidates

    chosen = preferred[0]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(chosen)
    logger.debug("Auto-set GOOGLE_APPLICATION_CREDENTIALS from '%s'.", chosen.name)


def get_key(name: str, keys_dir: Path | None = None) -> str | None:
    """Return the value of a key by name, checking env first then .key file.

    Args:
        name: Variable name, e.g. ``"ANTHROPIC_API_KEY"``.
        keys_dir: Directory containing .key files. Defaults to plato/keys/.

    Returns:
        The key value, or None if not found anywhere.
    """
    if name in os.environ:
        return os.environ[name]

    target_dir = keys_dir or _KEYS_DIR
    key_file = target_dir / f"{name}.key"

    if key_file.is_file():
        try:
            value = key_file.read_text(encoding="utf-8").strip()
            if value:
                return value
        except OSError:
            pass

    return None
