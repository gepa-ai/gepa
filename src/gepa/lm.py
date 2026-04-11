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
import threading
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

        self.completion_kwargs: dict[str, Any] = {
            **({"temperature": temperature} if temperature is not None else {}),
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
            **kwargs,
        }

        self._cumulative_cost: float = 0.0
        self._cost_lock = threading.Lock()

    @property
    def cost(self) -> float:
        """Cumulative USD cost across all calls made through this instance."""
        return self._cumulative_cost

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

        try:
            call_cost = litellm.completion_cost(completion_response=completion) or 0.0  # type: ignore[attr-defined]
        except Exception:
            call_cost = 0.0
        with self._cost_lock:
            self._cumulative_cost += call_cost

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

        batch_cost = 0.0
        results: list[str] = []
        for resp in responses:
            self._check_truncation(resp.choices)
            results.append(resp.choices[0].message.content.strip())
            try:
                batch_cost += litellm.completion_cost(completion_response=resp) or 0.0  # type: ignore[attr-defined]
            except Exception:
                pass
        with self._cost_lock:
            self._cumulative_cost += batch_cost

        return results

    def __repr__(self) -> str:
        params = [f"model={self.model!r}"]
        for k, v in self.completion_kwargs.items():
            params.append(f"{k}={v!r}")
        return f"LM({', '.join(params)})"


class CostTrackingLM:
    """Wrap any ``LanguageModel`` callable with a ``.cost`` property.

    For custom callables that don't go through LiteLLM, the actual cost
    is unobservable — ``.cost`` stays ``0.0``.  This wrapper exists so that
    code that reads ``.cost`` (e.g. :class:`~gepa.utils.MaxReflectionCostStopper`)
    works uniformly without type-checking the LM.
    """

    def __init__(self, lm: Any):
        self._inner = lm

    @property
    def cost(self) -> float:
        return getattr(self._inner, "cost", 0.0)

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        return self._inner(prompt)

    def __repr__(self) -> str:
        return f"CostTrackingLM({self._inner!r})"
