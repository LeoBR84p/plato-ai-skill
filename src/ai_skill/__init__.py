"""ai_skill — AI agent for the complete academic research pipeline."""

__version__ = "0.1.0"

# Load API keys from plato/keys/*.key files into os.environ at package import.
# Idempotent: variables already set in the environment are never overwritten.
from ai_skill.core.key_loader import load_keys as _load_keys
import os as _os

_load_keys()

# LangSmith observability — set project name if not already configured.
if not _os.environ.get("LANGCHAIN_PROJECT"):
    _os.environ["LANGCHAIN_PROJECT"] = "plato-ai-skill"
if _os.environ.get("LANGCHAIN_API_KEY") and not _os.environ.get("LANGCHAIN_TRACING_V2"):
    _os.environ["LANGCHAIN_TRACING_V2"] = "true"
