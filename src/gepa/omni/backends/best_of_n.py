"""Best-of-N backend.

A naive baseline: in a loop, draw an independent candidate from a single
LLM call (no conversation history, no past attempts in the prompt),
evaluate it through the eval server, and keep the best score seen. The
loop stops when the LLM-spend or eval budget is exhausted, the score
reaches ``stop_at_score``, or the optional ``max_n`` cap is hit.

This isolates the *sampling* component of LLM-driven optimization: no
proposer engineering, no parent selection, no reflection. Useful as a
floor for baseline comparisons in Terrarium-style evaluations.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from gepa.lm import LM
from gepa.omni._helpers import warn_unknown_config_keys
from gepa.omni.backend import Result

if TYPE_CHECKING:
    from gepa.omni.config import OmniConfig
    from gepa.omni.eval_server import EvalServer
    from gepa.omni.task import Task

logger = logging.getLogger(__name__)

_BON_CONFIG_KEYS = ("model", "temperature", "max_n", "lm_kwargs")

# Match a fenced code block. Captures the body. Allows an optional language tag.
_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)

_PROMPT_TEMPLATE = """You are writing a single candidate solution to the task below.

{sections}## Output format
Respond with a single fenced code block. Place your full final solution
between the fences and write nothing else outside the block.
"""


def _build_prompt(task: Task) -> str:
    sections = ""
    if task.objective:
        sections += f"## Objective\n{task.objective}\n\n"
    if task.background:
        sections += f"## Background\n{task.background}\n\n"
    if task.initial_candidate:
        sections += f"## Seed candidate\n```\n{task.initial_candidate}\n```\n\n"
    return _PROMPT_TEMPLATE.format(sections=sections)


def _parse_candidate(response: str) -> str | None:
    m = _FENCE_RE.search(response)
    if m:
        return m.group(1).rstrip("\n")
    # Fallback: if the model didn't fence, treat the raw text as the candidate
    # only when it is non-empty. Empty / whitespace-only is a parse failure.
    body = response.strip()
    return body or None


class BestOfNBackend:
    """Stateless sample-and-keep-best baseline.

    Backend-specific keys read from ``OmniConfig.config``:

    - ``model``: LiteLLM model id. Default ``"claude-sonnet-4-6"``.
    - ``temperature``: Sampling temperature. Default ``1.0`` (sampling on
      by default — N independent calls would otherwise produce N copies of
      the same response).
    - ``max_n``: Optional hard cap on the number of samples. Default
      ``None`` (run until the budget is exhausted).
    - ``lm_kwargs``: Extra kwargs forwarded to :class:`gepa.lm.LM`.
    """

    name = "best_of_n"

    def __init__(self, config: OmniConfig) -> None:
        extras = config.config
        warn_unknown_config_keys(self.name, extras, _BON_CONFIG_KEYS)
        self.model: str = extras.get("model", "claude-sonnet-4-6")
        self.temperature: float = float(extras.get("temperature", 1.0))
        self.max_n: int | None = extras.get("max_n")
        self.lm_kwargs: dict[str, Any] = dict(extras.get("lm_kwargs") or {})
        self.stop_at_score = config.stop_at_score
        self.effort = config.effort
        self.max_thinking_tokens = config.max_thinking_tokens

    def run(self, task: Task, server: EvalServer) -> Result:
        budget = server.budget

        lm_kwargs = dict(self.lm_kwargs)
        if self.effort is not None:
            lm_kwargs.setdefault("reasoning_effort", self.effort)
        if self.max_thinking_tokens is not None:
            lm_kwargs.setdefault("thinking", {"type": "enabled", "budget_tokens": self.max_thinking_tokens})

        lm = LM(self.model, temperature=self.temperature, **lm_kwargs)
        prompt = _build_prompt(task)

        best_score = float("-inf")
        best_candidate = task.initial_candidate
        eval_log: list[dict[str, Any]] = []
        n_samples = 0
        n_parse_failures = 0

        while True:
            if self.max_n is not None and n_samples >= self.max_n:
                break
            if budget.max_token_cost is not None:
                remaining = budget.max_token_cost - lm.total_cost - server.total_cost
                if remaining <= 0:
                    break
            if budget.max_evals is not None and budget.remaining is not None and budget.remaining <= 0:
                break

            try:
                response = lm(prompt)
            except Exception as e:
                logger.warning("LM call failed (sample %d): %s", n_samples, e)
                break

            n_samples += 1
            candidate = _parse_candidate(response)
            if candidate is None:
                n_parse_failures += 1
                eval_log.append({"sample": n_samples, "parse_failed": True})
                continue

            if task.has_dataset and task.train_set:
                scores = []
                for ex in task.train_set:
                    s, _info = server.evaluate(candidate, ex)
                    scores.append(s)
                score = sum(scores) / len(scores) if scores else float("-inf")
            else:
                score, _info = server.evaluate(candidate)

            eval_log.append({"sample": n_samples, "score": score, "candidate_len": len(candidate)})

            if score > best_score:
                best_score = score
                best_candidate = candidate

            if self.stop_at_score is not None and best_score >= self.stop_at_score:
                break

        if best_score == float("-inf"):
            best_score = 0.0  # never produced a parseable, evaluable candidate

        return Result(
            best_candidate=best_candidate,
            best_score=best_score,
            total_evals=n_samples,
            eval_log=eval_log,
            metadata={
                "adapter_cost": lm.total_cost,
                "n_samples": n_samples,
                "n_parse_failures": n_parse_failures,
                "model": self.model,
                "temperature": self.temperature,
            },
        )
