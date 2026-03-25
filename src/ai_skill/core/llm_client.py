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

        Uses the raw Anthropic streaming API to avoid the "Streaming required for
        requests > 10 minutes" rejection, then validates the complete JSON response
        with Pydantic. The JSON schema is injected into the system prompt so the
        model knows the exact structure expected.

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
        import json as _json
        import re as _re
        import pydantic

        tokens = max_tokens or self._max_tokens

        # Inject JSON schema into the system prompt once — the model outputs raw
        # JSON that we validate with Pydantic after the stream closes.
        schema_str = _json.dumps(
            response_model.model_json_schema(), ensure_ascii=False
        )
        json_instruction = (
            "\n\nRespond with ONLY a valid JSON object that matches this JSON Schema "
            "(no markdown fences, no explanatory text before or after):\n" + schema_str
        )
        full_system = (system + json_instruction) if system else json_instruction.lstrip()

        stream_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": tokens,
            "messages": messages,
            "system": full_system,
        }
        if temperature is not None:
            stream_kwargs["temperature"] = temperature

        for attempt in range(_MAX_RETRIES):
            try:
                # Stream the response to avoid Anthropic's 10-minute non-streaming
                # limit. get_final_text() waits for the stream to close and returns
                # the complete assistant text — no partial-model issues.
                with self._raw_client.messages.stream(**stream_kwargs) as stream:
                    full_text = stream.get_final_text()

                # Strip markdown code fences the model may add despite instructions.
                text = full_text.strip()
                text = _re.sub(r"^```(?:json)?\s*\n?", "", text)
                text = _re.sub(r"\n?```\s*$", "", text)
                text = text.strip()

                result: T = response_model.model_validate_json(text)

                try:
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
            except (pydantic.ValidationError, ValueError) as exc:
                # JSON / schema mismatch — retry; the model may self-correct.
                if attempt < _MAX_RETRIES - 1:
                    logger.warning(
                        "Structured output validation failed (attempt %d/%d): %s — retrying.",
                        attempt + 1,
                        _MAX_RETRIES,
                        str(exc)[:200],
                    )
                    continue
                raise LLMClientError(
                    f"Structured completion failed after {_MAX_RETRIES} attempts: "
                    f"{str(exc)[:300]}"
                ) from exc
            except Exception as exc:
                exc_summary = str(exc)[:300]
                raise LLMClientError(
                    f"Structured completion failed: {exc_summary}"
                ) from exc

        raise LLMClientError(
            f"Structured LLM call failed after {_MAX_RETRIES} attempts."
        )
