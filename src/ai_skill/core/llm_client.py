"""Thin wrapper around the Anthropic SDK with structured output via instructor.

All LLM calls in the project go through this module. This keeps the
orchestrator and skills decoupled from the specific SDK version and
makes the client injectable for testing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, TypeVar

import anthropic
import instructor
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Node name injected by nodes.py before each LLM call so history entries
# carry human-readable context (e.g. "plan", "evaluate", "align_charter").
_current_node: str = ""


def set_current_node(name: str) -> None:
    """Set the graph node name for history attribution.

    Called by node functions in nodes.py immediately before using LLMClient.

    Args:
        name: Node function name (e.g. "plan", "evaluate", "align_charter").
    """
    global _current_node
    _current_node = name


_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_MAX_TOKENS = 8192
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds; doubles on each retry

T = TypeVar("T", bound=BaseModel)


class LLMClientError(Exception):
    """Raised when the LLM client fails after all retries."""


class LLMClient:
    """Wrapper around the Anthropic SDK supporting both plain and structured calls.

    Structured calls use the instructor library to guarantee that the LLM
    response parses into a Pydantic model.

    Example:
        >>> client = LLMClient()
        >>> text = client.complete([{"role": "user", "content": "Hello"}])
        >>> model = client.complete_structured([...], MyModel)
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Initialize the LLM client.

        Args:
            api_key: Anthropic API key. Defaults to ANTHROPIC_API_KEY env var.
            model: Model identifier. Defaults to AI_SKILL_MODEL env var or
                claude-sonnet-4-6.
            max_tokens: Maximum tokens per completion. Defaults to
                AI_SKILL_MAX_TOKENS env var or 8192.

        Raises:
            LLMClientError: If ANTHROPIC_API_KEY is not set.
        """
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise LLMClientError(
                "ANTHROPIC_API_KEY is not set. "
                "Export the environment variable before running ai-skill."
            )

        self._model = (
            model
            or os.environ.get("AI_SKILL_MODEL")
            or _DEFAULT_MODEL
        )
        self._max_tokens = int(
            max_tokens
            or os.environ.get("AI_SKILL_MAX_TOKENS", _DEFAULT_MAX_TOKENS)
        )

        raw_client = anthropic.Anthropic(api_key=resolved_key)
        self._instructor_client = instructor.from_anthropic(raw_client)
        self._raw_client = raw_client

    @property
    def model(self) -> str:
        """The model identifier currently in use."""
        return self._model

    def complete(
        self,
        messages: list[dict[str, str]],
        system: str = "",
        max_tokens: int | None = None,
    ) -> str:
        """Send a completion request and return the response as plain text.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            system: Optional system prompt.
            max_tokens: Override the default max_tokens for this call.

        Returns:
            The assistant's response text.

        Raises:
            LLMClientError: If the call fails after _MAX_RETRIES attempts.
        """
        tokens = max_tokens or self._max_tokens

        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "max_tokens": tokens,
                    "messages": messages,
                }
                if system:
                    kwargs["system"] = system

                response = self._raw_client.messages.create(**kwargs)
                content = response.content[0]
                text = content.text if hasattr(content, "text") else str(content)
                try:
                    from ai_skill.core import history as _hist
                    _hist.log_llm_call(
                        system=system,
                        messages=messages,
                        response=text,
                        model=self._model,
                        node=_current_node,
                    )
                except Exception:
                    pass
                return text

            except anthropic.RateLimitError:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Retrying in %.1fs.",
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
            except anthropic.APIError as exc:
                raise LLMClientError(f"Anthropic API error: {exc}") from exc

        raise LLMClientError(
            f"LLM call failed after {_MAX_RETRIES} attempts (rate limit)."
        )

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> T:
        """Send a completion request and parse the response into a Pydantic model.

        Uses the instructor library to enforce structured output. The LLM is
        instructed to return JSON conforming to the model's schema.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            response_model: Pydantic model class to parse the response into.
            system: Optional system prompt.
            max_tokens: Override the default max_tokens for this call.
            temperature: Sampling temperature in [0.0, 1.0]. Use 0.0 for
                deterministic, reproducible outputs (e.g. verification agents).

        Returns:
            An instance of response_model populated from the LLM response.

        Raises:
            LLMClientError: If the call or parsing fails after all retries.
        """
        tokens = max_tokens or self._max_tokens

        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict[str, Any] = {
                    "model": self._model,
                    "max_tokens": tokens,
                    "messages": messages,
                    "response_model": response_model,
                }
                if system:
                    kwargs["system"] = system
                if temperature is not None:
                    kwargs["temperature"] = temperature

                result = self._instructor_client.messages.create(**kwargs)  # type: ignore[return-value]
                try:
                    import json as _json
                    from ai_skill.core import history as _hist
                    _hist.log_llm_call(
                        system=system,
                        messages=messages,
                        response=_json.dumps(result.model_dump(), ensure_ascii=False),
                        model=self._model,
                        node=_current_node,
                    )
                except Exception:
                    pass
                return result

            except anthropic.RateLimitError:
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Rate limit hit on structured call (attempt %d/%d). "
                    "Retrying in %.1fs.",
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
            except Exception as exc:
                # Truncate the exception string — raw instructor/Anthropic
                # Message objects can be thousands of chars and flood the log.
                exc_summary = str(exc)[:300]
                raise LLMClientError(
                    f"Structured completion failed: {exc_summary}"
                ) from exc

        raise LLMClientError(
            f"Structured LLM call failed after {_MAX_RETRIES} attempts."
        )
