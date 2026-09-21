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
from typing import Any, cast

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
        self._cost_lock = threading.Lock()

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

        # Accumulate token usage
        usage = getattr(completion, "usage", None)
        tokens_in = (getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
        tokens_out = (getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0

        with self._cost_lock:
            self._total_cost += cost
            self._total_tokens_in += tokens_in
            self._total_tokens_out += tokens_out

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
        batch_tokens_in = 0
        batch_tokens_out = 0
        results: list[str] = []
        for resp in responses:
            self._check_truncation(resp.choices)
            results.append(resp.choices[0].message.content.strip())
            try:
                batch_cost += litellm.completion_cost(completion_response=resp) or 0.0  # type: ignore[attr-defined]
            except Exception:
                pass
            usage = getattr(resp, "usage", None)
            if usage is not None:
                batch_tokens_in += getattr(usage, "prompt_tokens", 0) or 0
                batch_tokens_out += getattr(usage, "completion_tokens", 0) or 0

        with self._cost_lock:
            self._total_cost += batch_cost
            self._total_tokens_in += batch_tokens_in
            self._total_tokens_out += batch_tokens_out

        return results

    def __repr__(self) -> str:
        params = [f"model={self.model!r}"]
        for k, v in self.completion_kwargs.items():
            params.append(f"{k}={v!r}")
        return f"LM({', '.join(params)})"


def make_vllm_lm(
    model: str,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    temperature: float | None = None,
    max_tokens: int | None = None,
    num_retries: int = 3,
    **kwargs: Any,
) -> LM:
    """Build an :class:`LM` backed by a local vLLM (or any OpenAI-compatible) server.

    vLLM serves Hugging Face models behind an OpenAI-compatible endpoint::

        python -m vllm.entrypoints.openai.api_server \\
            --model Qwen/Qwen2.5-7B-Instruct \\
            --served-model-name qwen2.5-7b-instruct \\
            --port 8000

    This helper points GEPA's LiteLLM-backed :class:`LM` at that endpoint. It is a
    thin convenience wrapper — the equivalent explicit form is
    ``LM("openai/<served-model-name>", api_base=..., api_key="EMPTY")``.

    **No import-time dependency is added:** ``vllm``/``transformers`` only need to be
    installed on the machine *serving* the model, never in the GEPA process. LiteLLM
    (already an optional GEPA dependency, imported lazily) does the HTTP call. The
    returned :class:`LM` is a plain ``LanguageModel`` callable, usable as either the
    task model or the ``reflection_lm``, and mixes freely with hosted providers.

    Args:
        model: The vLLM ``--served-model-name`` (e.g. ``"qwen2.5-7b-instruct"``).
            A LiteLLM provider prefix is prepended automatically when missing
            (``"openai/"``), so pass the served name as-is. To force a specific
            route, prefix it yourself (e.g. ``"hosted_vllm/qwen2.5-7b-instruct"``);
            an existing ``openai/`` or ``hosted_vllm/`` prefix is left untouched.
        api_base: The server's OpenAI-compatible base URL, **including** the ``/v1``
            suffix. Defaults to ``"http://localhost:8000/v1"``.
        api_key: Ignored by vLLM but required by the OpenAI client contract; defaults
            to ``"EMPTY"``.
        temperature: Optional sampling temperature.
        max_tokens: Optional maximum tokens to generate.
        num_retries: Retries on transient failures (default 3).
        **kwargs: Extra keyword arguments forwarded to ``litellm.completion``
            (e.g. ``top_p``, ``timeout``).

    Returns:
        An :class:`LM` instance targeting the local endpoint.
    """
    if not model.startswith(("openai/", "hosted_vllm/")):
        model = f"openai/{model}"
    return LM(
        model,
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=num_retries,
        api_base=api_base,
        api_key=api_key,
        **kwargs,
    )


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

    def __getattr__(self, name: str):
        # Conditionally expose batch_complete: hasattr(tracking_lm,
        # "batch_complete") must be True exactly when the wrapped callable
        # provides it, so batched reflection (StatelessReflectionLM) is not
        # silently downgraded to the per-task path by this wrapper.
        if name == "batch_complete":
            inner = getattr(self._fn, "batch_complete", None)
            if not callable(inner):
                raise AttributeError(name)

            def tracked_batch_complete(messages_list):
                for messages in messages_list:
                    self._total_tokens_in += self._estimate_tokens(str(messages))
                results = list(cast(Any, inner)(messages_list))
                for result in results:
                    if isinstance(result, str):
                        self._total_tokens_out += self._estimate_tokens(result)
                return results

            return tracked_batch_complete
        raise AttributeError(name)

    def __repr__(self) -> str:
        return f"TrackingLM({self._fn!r})"
