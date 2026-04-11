# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Thin LM abstraction over LiteLLM that handles retries, truncation
warnings, and cross-model compatibility.

Usage::

    from gepa.lm import LM

    lm = LM("openai/gpt-4.1", temperature=0.7, max_tokens=4096)
    response: str = lm("Solve this problem...")

    # Also works with chat messages
    response = lm([{"role": "user", "content": "Hello"}])

The returned callable conforms to the ``LanguageModel`` protocol
(``(str | list[dict]) -> str``) used throughout GEPA.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LM:
    """A lightweight language model wrapper over LiteLLM.

    Handles:

    - **Retries** with exponential backoff via LiteLLM's ``num_retries``.
    - **Truncation detection** — logs a warning when ``finish_reason='length'``.
    - **drop_params=True** so unsupported params are silently ignored
      (with a warning logged for transparency).

    Conforms to the :class:`~gepa.proposer.reflective_mutation.base.LanguageModel`
    protocol, so it can be used anywhere GEPA expects a ``LanguageModel``.

    Args:
        model: LiteLLM model identifier, e.g. ``"openai/gpt-4.1"`` or ``"anthropic/claude-sonnet-4-6"``.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
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
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0

        self.completion_kwargs: dict[str, Any] = {
            **({"temperature": temperature} if temperature is not None else {}),
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **kwargs,
        }

    @property
    def total_cost(self) -> float:
        """Cumulative USD cost of all calls made through this LM instance."""
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        """Cumulative input (prompt) tokens across all calls."""
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        """Cumulative output (completion) tokens across all calls."""
        return self._total_tokens_out

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

        # Non-streaming calls always return ModelResponse (not CustomStreamWrapper)
        self._check_truncation(completion.choices)  # type: ignore[union-attr]

        # Accumulate cost
        try:
            cost = litellm.completion_cost(completion_response=completion) or 0.0  # type: ignore[attr-defined]
        except Exception:
            cost = 0.0
        self._total_cost += cost

        # Accumulate token usage
        usage = getattr(completion, "usage", None)
        if usage is not None:
            self._total_tokens_in += getattr(usage, "prompt_tokens", 0) or 0
            self._total_tokens_out += getattr(usage, "completion_tokens", 0) or 0

        return completion.choices[0].message.content  # type: ignore[union-attr]

    def batch_complete(
        self, messages_list: list[list[dict[str, Any]]], max_workers: int = 10, **kwargs: Any
    ) -> list[str]:
        """Run multiple completions in parallel using ``litellm.batch_completion``.

        Args:
            messages_list: List of message lists, one per request.
            max_workers: Maximum concurrent requests.
            **kwargs: Extra keyword arguments forwarded to ``litellm.batch_completion``
                (e.g. ``timeout``, ``api_base``).  These override any matching keys
                set during ``__init__``.

        Returns:
            List of response strings, one per input.
        """
        import litellm

        merged = {**self.completion_kwargs, **kwargs}
        responses = litellm.batch_completion(
            model=self.model,
            messages=messages_list,
            max_workers=max_workers,
            num_retries=self.num_retries,
            drop_params=True,
            **merged,
        )

        results: list[str] = []
        for resp in responses:
            self._check_truncation(resp.choices)
            results.append(resp.choices[0].message.content.strip())

        return results

    def __repr__(self) -> str:
        params = [f"model={self.model!r}"]
        for k, v in self.completion_kwargs.items():
            params.append(f"{k}={v!r}")
        return f"LM({', '.join(params)})"


class TrackingLM:
    """Wraps an arbitrary callable to track estimated token usage.

    For callables that don't go through LiteLLM, we can't get real token
    counts or costs.  This wrapper estimates tokens from string lengths
    (~4 chars/token) and reports ``total_cost = 0.0``.

    This ensures that *every* reflection LM — whether an ``LM`` instance
    or a plain callable — exposes ``total_cost``, ``total_tokens_in``,
    and ``total_tokens_out``.
    """

    _CHARS_PER_TOKEN = 4

    def __init__(self, fn: Any):
        self._fn = fn
        self._total_cost: float = 0.0
        self._total_tokens_in: int = 0
        self._total_tokens_out: int = 0

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_tokens_in(self) -> int:
        return self._total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._total_tokens_out

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // self._CHARS_PER_TOKEN)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if isinstance(prompt, str):
            self._total_tokens_in += self._estimate_tokens(prompt)
        else:
            self._total_tokens_in += self._estimate_tokens(str(prompt))

        result = self._fn(prompt)

        if isinstance(result, str):
            self._total_tokens_out += self._estimate_tokens(result)

        return result

    def __repr__(self) -> str:
        return f"TrackingLM({self._fn!r})"
