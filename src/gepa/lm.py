# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Thin LM abstraction over LiteLLM that handles model family detection,
parameter normalization, retries, and truncation warnings.

Usage::

    from gepa.lm import LM

    lm = LM("openai/gpt-5-mini")          # auto-detects reasoning model
    lm = LM("openai/gpt-4.1", temperature=0.7, max_tokens=4096)
    response: str = lm("Solve this problem...")

    # Also works with chat messages
    response = lm([{"role": "user", "content": "Hello"}])

The returned callable conforms to the ``LanguageModel`` protocol
(``(str | list[dict]) -> str``) used throughout GEPA.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Regex detecting OpenAI reasoning models that require special parameter handling:
#   o1, o3, o4, o5 (with optional -mini/-nano/-pro suffix and optional date suffix)
#   gpt-5 family (excluding -chat variants which are non-reasoning)
_REASONING_MODEL_RE = re.compile(
    r"^(?:o[1-9]\d*(?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$"
)


def _is_reasoning_model(model: str) -> bool:
    """Return True if *model* is an OpenAI reasoning model (o1/o3/o4/gpt-5 family)."""
    family = model.split("/")[-1].lower() if "/" in model else model.lower()
    return bool(_REASONING_MODEL_RE.match(family))


class LM:
    """A lightweight language model wrapper over LiteLLM.

    Handles:

    - **Reasoning model detection** (o1/o3/o4/gpt-5): enforces ``temperature=1``
      and maps ``max_tokens`` to ``max_completion_tokens``.
    - **Retries** with exponential backoff via LiteLLM's ``num_retries``.
    - **Truncation detection** — logs a warning when ``finish_reason='length'``.
    - **drop_params=True** so unsupported params are silently ignored.

    Conforms to the :class:`~gepa.proposer.reflective_mutation.base.LanguageModel`
    protocol, so it can be used anywhere GEPA expects a ``LanguageModel``.

    Args:
        model: LiteLLM model identifier, e.g. ``"openai/gpt-4.1"`` or ``"anthropic/claude-sonnet-4-6"``.
        temperature: Sampling temperature.  Reasoning models require 1.0 or None.
        max_tokens: Maximum tokens to generate.  For reasoning models this is
            mapped to ``max_completion_tokens`` and defaults to 16000.
        num_retries: Number of retries on transient failures (default 3).
        **kwargs: Extra keyword arguments forwarded to ``litellm.completion``
            (e.g. ``top_p``, ``stop``, ``api_key``, ``api_base``).
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        num_retries: int = 3,
        **kwargs: Any,
    ):
        self.model = model
        self.num_retries = num_retries
        self.is_reasoning = _is_reasoning_model(model)

        if self.is_reasoning:
            if temperature is not None and temperature != 1.0:
                raise ValueError(
                    f"Reasoning model '{model}' requires temperature=1.0 or None, got {temperature}. "
                    "Set temperature=1.0 or omit it."
                )
            self.completion_kwargs: dict[str, Any] = {
                "temperature": 1.0,
                "max_completion_tokens": max_tokens or 16000,
                **kwargs,
            }
        else:
            self.completion_kwargs = {
                **({"temperature": temperature} if temperature is not None else {}),
                **({"max_tokens": max_tokens} if max_tokens is not None else {}),
                **kwargs,
            }

    def _check_truncation(self, choices: list[Any]) -> None:
        if any(getattr(c, "finish_reason", None) == "length" for c in choices):
            max_tok = self.completion_kwargs.get("max_tokens") or self.completion_kwargs.get("max_completion_tokens")
            logger.warning(
                f"LM response was truncated (finish_reason='length', max_tokens={max_tok}). "
                "Consider increasing max_tokens for better results."
            )

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        import litellm

        if isinstance(prompt, str):
            messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        completion = litellm.completion(
            model=self.model,
            messages=messages,
            num_retries=self.num_retries,
            drop_params=True,
            **self.completion_kwargs,
        )

        self._check_truncation(completion.choices)
        return completion.choices[0].message.content  # type: ignore[union-attr]

    def batch_complete(self, messages_list: list[list[dict[str, Any]]], max_workers: int = 10) -> list[str]:
        """Run multiple completions in parallel using ``litellm.batch_completion``.

        Args:
            messages_list: List of message lists, one per request.
            max_workers: Maximum concurrent requests.

        Returns:
            List of response strings, one per input.
        """
        import litellm

        responses = litellm.batch_completion(
            model=self.model,
            messages=messages_list,
            max_workers=max_workers,
            num_retries=self.num_retries,
            drop_params=True,
            **self.completion_kwargs,
        )

        results: list[str] = []
        for resp in responses:
            self._check_truncation(resp.choices)
            results.append(resp.choices[0].message.content.strip())

        return results

    def __repr__(self) -> str:
        params = [f"model={self.model!r}"]
        if self.is_reasoning:
            params.append("reasoning=True")
        for k, v in self.completion_kwargs.items():
            params.append(f"{k}={v!r}")
        return f"LM({', '.join(params)})"
