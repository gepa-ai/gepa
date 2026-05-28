from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from gepa.core.adapter import EvaluationBatch, GEPAAdapter, ProposalFn

logger = logging.getLogger(__name__)

DataInst = Mapping[str, Any]
RolloutState = dict[str, Any]
RolloutFn = Callable[[dict[str, str], DataInst], RolloutState]
EvalFn = Callable[[DataInst, RolloutState], tuple[float, str]]
ReflectiveRecordFn = Callable[[DataInst, RolloutState, float, str], Mapping[str, Any]]
ReflectionLM = Callable[[str], str]


def make_reflection_lm(model: BaseChatModel | Mapping[str, Any] | str) -> ReflectionLM:
    """Wrap a LangChain chat model into the `(prompt: str) -> str` callable GEPA expects.

    `model` accepts either:
      - a `BaseChatModel` instance,
      - a `provider:model` string (passed to `init_chat_model`), or
      - a kwargs mapping (passed via `init_chat_model(**model)`).
    """
    if not isinstance(model, BaseChatModel):
        from langchain.chat_models import init_chat_model

        model = init_chat_model(model) if isinstance(model, str) else init_chat_model(**dict(model))

    def reflection_lm(prompt: str) -> str:
        logger.debug("reflection prompt (%d chars):\n%s", len(prompt), prompt)
        result = model.invoke([HumanMessage(content=prompt)])
        content = result.content
        if isinstance(content, list):
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
        else:
            text = str(content)
        logger.debug("reflection response (%d chars):\n%s", len(text), text)
        return text

    return reflection_lm


def last_message_text(state: RolloutState) -> str:
    """Return the text content of the last message in a rollout state.

    Works for both single-turn rollouts (`{"messages": [AIMessage(...)]}`) and
    agent rollouts that follow LangGraph's standard state shape.
    """
    messages = state.get("messages") if isinstance(state, dict) else None
    if not messages:
        return ""
    msg = messages[-1]
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def _default_reflective_record(data: DataInst, state: RolloutState, score: float, feedback: str) -> Mapping[str, Any]:
    inputs = data.get("input", data)
    return {
        "Inputs": inputs,
        "Generated Outputs": last_message_text(state),
        "Feedback": feedback,
    }


def _truncate(text: str, limit: int = 800) -> str:
    text = text.replace("\n", " ⏎ ")
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [+{len(text) - limit} chars]"


class LangChainAdapter(GEPAAdapter):
    """GEPA adapter for arbitrary LangChain rollouts.

    Caller supplies:
      - `rollout_fn(candidate, example) -> state dict`: any LangChain pipeline —
        a single chat-model invocation, an agent built with `create_agent`, a
        custom LangGraph graph, RAG, etc. Must return a state dict; for single-turn
        cases return a dict with the messages key
        `{"messages": messages + [AIMessage("llm response")]}`.
        For agents, return the full agent state directly (e.g. `agent.invoke(...)`).
      - `eval_fn(example, state) -> (score, feedback)`: scores the rollout state.
        Use `last_message_text(state)` if you only need the final assistant text;
        agents can inspect tool calls in `state["messages"]` directly.
        Note: if `rollout_fn` raises, the adapter substitutes a stand-in state
        of the form `{"messages": [AIMessage("ERROR: <type>: <msg>")], "error": e}`
        and still calls `eval_fn` with it. Check `state.get("error")` to detect
        rollout failures and score them appropriately (e.g. return 0.0 with a
        feedback string explaining the failure to the reflection LM).
      - `reflective_record_fn(example, state, score, feedback) -> mapping`
        (optional): builds the per-example record passed to the reflection LM.
        Defaults to `{"Inputs", "Generated Outputs", "Feedback"}` derived from
        `example["input"]` and `last_message_text(state)`. Override to surface
        tool-call traces, intermediate steps, or domain-specific context.
      - `num_threads`: parallelism for `evaluate` (default 32).
      - `custom_proposer`: optional `ProposalFn` to override GEPA's default
        text-proposal behavior.
      - `show_progress`: whether to render a tqdm progress bar during
        `evaluate` (default True).

    The candidate is a `dict[str, str]` of named text components; `rollout_fn`
    decides how those components are wired into the call.
    """

    def __init__(
        self,
        rollout_fn: RolloutFn,
        eval_fn: EvalFn,
        num_threads: int = 32,
        custom_proposer: ProposalFn | None = None,
        reflective_record_fn: ReflectiveRecordFn = _default_reflective_record,
        show_progress: bool = True,
    ):
        self.rollout_fn = rollout_fn
        self.eval_fn = eval_fn
        self.num_threads = num_threads
        self.reflective_record_fn = reflective_record_fn
        self.propose_new_texts = custom_proposer
        self.show_progress = show_progress

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        logger.info(
            "evaluate: batch=%d capture_traces=%s components=%s",
            len(batch),
            capture_traces,
            list(candidate.keys()),
        )
        for name, text in candidate.items():
            logger.debug("candidate[%s]: %s", name, _truncate(text, limit=400))

        outputs: list[Any] = [None] * len(batch)
        scores: list[float] = [0.0] * len(batch)
        trajectories: list[Any] | None = [None] * len(batch) if capture_traces else None

        def run_one(idx: int, example: DataInst):
            t0 = time.perf_counter()
            error: Exception | None = None
            try:
                state: RolloutState = self.rollout_fn(candidate, example)
                if not isinstance(state, dict):
                    raise TypeError(
                        f"rollout_fn must return a dict, got {type(state).__name__}. "
                        f"For single-turn chat use {{'messages': [SystemMessage(...), HumanMessage(...), AIMessage(...)]}}; "
                        f"for LangGraph agents return the agent state dict directly."
                    )
            except Exception as e:
                state = {"messages": [AIMessage(content=f"ERROR: {type(e).__name__}: {e}")], "error": e}
                error = e
            elapsed_ms = (time.perf_counter() - t0) * 1000
            score, feedback = self.eval_fn(example, state)
            if error is not None:
                input_preview = _truncate(str(example.get("input", example)), limit=120)
                logger.warning(
                    "example %d FAILED after %.0fms: %s: %s | input=%s",
                    idx,
                    elapsed_ms,
                    type(error).__name__,
                    error,
                    input_preview,
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )
            else:
                logger.debug("example %d ok in %.0fms (score=%.2f)", idx, elapsed_ms, score)
            return idx, state, score, feedback

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
            futures = [pool.submit(run_one, i, ex) for i, ex in enumerate(batch)]
            iterator: Any = as_completed(futures)
            if self.show_progress:
                try:
                    from tqdm.auto import tqdm

                    iterator = tqdm(iterator, total=len(futures), desc="evaluate", leave=False)
                except ImportError:
                    pass

            running_correct = 0
            for f in iterator:
                idx, state, score, feedback = f.result()
                outputs[idx] = {"state": state, "response": last_message_text(state)}
                scores[idx] = score
                if trajectories is not None:
                    trajectories[idx] = {
                        "data": batch[idx],
                        "state": state,
                        "score": score,
                        "feedback": feedback,
                    }
                if score == 1.0:
                    running_correct += 1
                if self.show_progress and hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(correct=running_correct, refresh=False)
                logger.debug(
                    "  example %d: score=%.3f feedback=%s",
                    idx,
                    score,
                    _truncate(feedback, limit=200),
                )

        elapsed = time.perf_counter() - start
        total = sum(scores)
        mean = total / len(scores) if scores else 0.0

        failure_counts: Counter[str] = Counter()
        for o in outputs:
            if isinstance(o, dict) and isinstance(o.get("state"), dict) and o["state"].get("error") is not None:
                err_type = type(o["state"]["error"]).__name__
                failure_counts[err_type] += 1
        if failure_counts:
            logger.warning(
                "evaluate failures: %d/%d (by type: %s)",
                sum(failure_counts.values()),
                len(batch),
                dict(failure_counts),
            )

        logger.info(
            "evaluate done: mean=%.3f sum=%.3f n=%d in %.2fs",
            mean,
            total,
            len(scores),
            elapsed,
        )
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        records = [
            self.reflective_record_fn(traj["data"], traj["state"], traj["score"], traj["feedback"])
            for traj in (eval_batch.trajectories or [])
        ]
        if not records:
            raise RuntimeError("No trajectories captured; cannot build reflective dataset.")
        logger.info(
            "reflective dataset: %d records for components=%s",
            len(records),
            components_to_update,
        )
        return dict.fromkeys(components_to_update, records)
