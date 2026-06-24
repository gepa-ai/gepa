"""Best-of-N engine.

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

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from gepa.lm import LM
from gepa.oa._helpers import warn_unknown_config_keys
from gepa.oa.engine import Result

if TYPE_CHECKING:
    from pathlib import Path

    from gepa.oa.config import OptimizeAnythingConfig
    from gepa.oa.eval_server import EvalServer
    from gepa.oa.task import Task

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
    if task.seed_candidate:
        sections += f"## Seed candidate\n```\n{task.seed_candidate}\n```\n\n"
    return _PROMPT_TEMPLATE.format(sections=sections)


def _parse_candidate(response: str | None) -> str | None:
    # ``LM.__call__`` returns the assistant message ``content`` directly,
    # which can be ``None`` when the model produced only tool calls or
    # finish_reason was non-textual. Guard so we treat it as a parse failure.
    if not response:
        return None
    m = _FENCE_RE.search(response)
    if m:
        return m.group(1).rstrip("\n")
    # Fallback: if the model didn't fence, treat the raw text as the candidate
    # only when it is non-empty. Empty / whitespace-only is a parse failure.
    body = response.strip()
    return body or None


class BestOfNEngine:
    """Stateless sample-and-keep-best baseline.

    Engine-specific keys read from ``OptimizeAnythingConfig.engine_config``:

    - ``model``: LiteLLM model id. Default ``"claude-sonnet-4-6"``.
    - ``temperature``: Sampling temperature. Default ``1.0`` (sampling on
      by default — N independent calls would otherwise produce N copies of
      the same response).
    - ``max_n``: Optional hard cap on the number of samples. Default
      ``None`` (run until the budget is exhausted).
    - ``lm_kwargs``: Extra kwargs forwarded to :class:`gepa.lm.LM`.
    """

    name = "best_of_n"

    def __init__(self, config: OptimizeAnythingConfig) -> None:
        extras = config.engine_config
        warn_unknown_config_keys(self.name, extras, _BON_CONFIG_KEYS)
        self.model: str = extras.get("model", "claude-sonnet-4-6")
        self.temperature: float = float(extras.get("temperature", 1.0))
        self.max_n: int | None = extras.get("max_n")
        self.lm_kwargs: dict[str, Any] = dict(extras.get("lm_kwargs") or {})
        self.stop_at_score = config.stop_at_score
        self.effort = config.effort
        self.max_thinking_tokens = config.max_thinking_tokens
        # Proposer-cost cap: USD this engine may spend sampling candidates.
        # Eval-side cost (server.total_cost) is a separate bucket.
        self.max_token_cost = config.max_token_cost

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
        best_candidate = task.seed_candidate or ""
        eval_log: list[dict[str, Any]] = []
        n_samples = 0
        n_parse_failures = 0

        while True:
            if self.max_n is not None and n_samples >= self.max_n:
                break
            if self.max_token_cost is not None:
                # Proposer-cost cap: only this engine's own LLM spend counts.
                # Eval-side cost (``server.total_cost``) is a separate bucket,
                # summed in only by the api's final report.
                if lm.total_cost >= self.max_token_cost:
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
                eval_log.append(
                    {
                        "sample": n_samples,
                        "parse_failed": True,
                        "lm_cost": lm.total_cost,
                        "eval_cost": server.total_cost,
                    }
                )
                continue

            if task.has_dataset and task.train_set:
                scores = []
                for ex in task.train_set:
                    s, _info = server.evaluate(candidate, ex)
                    scores.append(s)
                score = sum(scores) / len(scores) if scores else float("-inf")
            else:
                score, _info = server.evaluate(candidate)

            if score > best_score:
                best_score = score
                best_candidate = candidate

            # Snapshot cost *after* both the LM call and the eval(s) for this
            # sample. ``best_score`` is the running best including this sample,
            # making this row directly plottable as cost-vs-best-score.
            eval_log.append(
                {
                    "sample": n_samples,
                    "score": score,
                    "best_score": best_score,
                    "candidate_len": len(candidate),
                    "lm_cost": lm.total_cost,
                    "eval_cost": server.total_cost,
                }
            )

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
                "bon_cost_log": list(eval_log),
            },
        )

    def process_result(self, result: Result, output_dir: Path | None) -> None:
        """Persist per-sample (lm_cost, eval_cost, score) to ``bon_cost_log.jsonl``.

        The optimize_anything api / terrarium runner overwrite ``Result.eval_log`` with the
        eval server's record (which doesn't carry per-LM-call cost), so this
        side file is the only place the sampling-cost timeline survives.
        """
        if output_dir is None:
            return
        log = result.metadata.get("bon_cost_log") or []
        if not log:
            return
        out = output_dir / "bon_cost_log.jsonl"
        with out.open("w") as f:
            for entry in log:
                f.write(json.dumps(entry, default=str) + "\n")
