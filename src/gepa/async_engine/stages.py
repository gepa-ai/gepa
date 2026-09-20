# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Stage bodies. Each runs on a worker thread and never touches optimizer state.

A stage body takes the items of one task and returns one result per item. The
head thread writes results back onto the items, updates state, and routes the
items to the next buffer.
"""

from __future__ import annotations

import inspect
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from gepa.async_engine.items import WorkItem
from gepa.core.adapter import EvaluationBatch, GEPAAdapter, ProposalFn, invoke_batch_evaluate
from gepa.core.data_loader import DataLoader
from gepa.proposer.reflective_mutation.base import LanguageModel
from gepa.proposer.reflective_mutation.reflection_lm import ReflectionLM


def metric_calls(batch: EvaluationBatch, fallback: int) -> int:
    return batch.num_metric_calls if batch.num_metric_calls is not None else fallback


# ----------------------------------------------------------------------
# rollout: evaluate parents on their minibatches, with traces
# ----------------------------------------------------------------------


def run_rollout(adapter: GEPAAdapter, items: list[WorkItem]) -> list[EvaluationBatch]:
    pairs = [(it.parent_candidate, it.minibatch) for it in items]
    return invoke_batch_evaluate(adapter, pairs, capture_traces=True)


# ----------------------------------------------------------------------
# propose: reflective dataset + reflection
# ----------------------------------------------------------------------


@dataclass
class ProposeResult:
    reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]] | None
    new_texts: dict[str, str]
    prompts: dict[str, Any]
    raw_lm_outputs: dict[str, str]
    metadata: dict[str, Any]
    error: BaseException | None = None


class Reflector:
    """Produces new component texts, honoring the same precedence as the synchronous proposer.

    ``adapter.propose_new_texts`` wins, then ``custom_candidate_proposer``, then
    the :class:`ReflectionLM`. A stateful ``ReflectionLM`` returns a successor
    from every call; with more than one propose worker the order in which
    successors are chained is the order in which calls finish.
    """

    def __init__(
        self,
        adapter: GEPAAdapter,
        reflection_lm: ReflectionLM | None,
        custom_candidate_proposer: ProposalFn | None,
    ):
        self.adapter = adapter
        self._lm = reflection_lm
        self.custom = custom_candidate_proposer
        self._lock = threading.Lock()
        self._custom_accepts_metadata = False
        if custom_candidate_proposer is not None:
            try:
                sig = inspect.signature(custom_candidate_proposer)
                self._custom_accepts_metadata = "metadata" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                )
            except (TypeError, ValueError):
                self._custom_accepts_metadata = False
        if adapter.propose_new_texts is None and custom_candidate_proposer is None and reflection_lm is None:
            raise ValueError("reflection_lm must be provided when adapter.propose_new_texts is None.")

    def reflect(
        self,
        candidate: dict[str, str],
        dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components: list[str],
        metadata: Mapping[str, Any],
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, str], dict[str, Any]]:
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(candidate, dataset, components), {}, {}, {}
        if self.custom is not None:
            kwargs = {"metadata": metadata} if self._custom_accepts_metadata else {}
            return self.custom(candidate, dataset, components, **kwargs), {}, {}, {}
        with self._lock:
            lm = self._lm
        assert lm is not None
        proposal, next_lm = lm.reflect(candidate, dataset, components)
        if next_lm is not lm:
            with self._lock:
                self._lm = next_lm
        return proposal.new_texts, dict(proposal.prompts), dict(proposal.raw_lm_outputs), dict(proposal.metadata)


def run_propose(
    adapter: GEPAAdapter,
    reflector: Reflector,
    items: list[WorkItem],
    metadatas: Sequence[Mapping[str, Any]],
) -> list[ProposeResult]:
    """Build each item's reflective dataset and reflect. Failures are isolated per item."""
    results: list[ProposeResult] = []
    for item, md in zip(items, metadatas, strict=True):
        assert item.parent_eval is not None and item.components is not None
        try:
            dataset = adapter.make_reflective_dataset(item.parent_candidate, item.parent_eval, item.components)
            new_texts, prompts, raw, meta = reflector.reflect(item.parent_candidate, dataset, item.components, md)
            results.append(ProposeResult(dataset, dict(new_texts or {}), prompts, raw, meta))
        except Exception as e:
            results.append(ProposeResult(None, {}, {}, {}, {}, error=e))
    return results


# ----------------------------------------------------------------------
# screen: evaluate children (and a rebased parent, when needed) on the minibatch
# ----------------------------------------------------------------------


@dataclass
class ScreenResult:
    child_eval: EvaluationBatch
    parent_eval: EvaluationBatch | None


def run_screen(adapter: GEPAAdapter, items: list[WorkItem]) -> list[ScreenResult]:
    pairs: list[tuple[dict[str, str], list[Any]]] = []
    slots: list[tuple[int, int | None]] = []
    for it in items:
        assert it.child_candidate is not None
        child_slot = len(pairs)
        pairs.append((it.child_candidate, it.minibatch))
        parent_slot: int | None = None
        if it.needs_parent_eval:
            parent_slot = len(pairs)
            pairs.append((it.parent_candidate, it.minibatch))
        slots.append((child_slot, parent_slot))
    evals = invoke_batch_evaluate(adapter, pairs, capture_traces=True)
    return [ScreenResult(evals[c], evals[p] if p is not None else None) for c, p in slots]


# ----------------------------------------------------------------------
# validate: evaluate accepted children on (part of) the validation set
# ----------------------------------------------------------------------


def run_validate(
    adapter: GEPAAdapter,
    valset: DataLoader,
    items: list[WorkItem],
    id_lists: list[list[Any]],
) -> list[EvaluationBatch | None]:
    todo = [(i, ids) for i, ids in enumerate(id_lists) if ids]
    pairs = []
    for i, ids in todo:
        candidate = items[i].child_candidate
        assert candidate is not None
        pairs.append((candidate, valset.fetch(ids)))
    fresh = invoke_batch_evaluate(adapter, pairs, capture_traces=False) if pairs else []
    out: list[EvaluationBatch | None] = [None] * len(items)
    for (i, _ids), batch in zip(todo, fresh, strict=True):
        out[i] = batch
    return out


# ----------------------------------------------------------------------
# patch: reconcile a stale child with what the pool gained meanwhile
# ----------------------------------------------------------------------

PATCH_PROMPT = """You maintain a text component that is being improved by several workers at once.

A worker proposed a revision, but it started from an older version. Since then the component has been improved by
others. Decide whether the stale revision still contributes something the current best version lacks.

# Version the worker started from
```
<base>
```

# The worker's revision of that version
```
<revision>
```

# Current best version
```
<current>
```

# Versions accepted since the worker started (oldest first)
<history>

Compare the worker's revision with the version it started from to see what it changed. If every useful change is
already covered by the current best version, or the change only fits the specific examples the worker saw, answer
with the single word DISCARD.

Otherwise rewrite the current best version so that it also carries the transferable part of the worker's change.
Keep everything that makes the current best version good. Provide the full new text within ``` blocks."""


@dataclass
class PatchJob:
    component: str
    base_text: str
    revision_text: str
    current_text: str
    history: list[str]


@dataclass
class PatchResult:
    #: component -> patched text. Empty means discard.
    texts: dict[str, str]
    raw_outputs: dict[str, str]


def render_patch_prompt(job: PatchJob) -> str:
    history = "\n\n".join(f"## Accepted version {i + 1}\n```\n{t}\n```" for i, t in enumerate(job.history)) or "(none)"
    return (
        PATCH_PROMPT.replace("<base>", job.base_text)
        .replace("<revision>", job.revision_text)
        .replace("<current>", job.current_text)
        .replace("<history>", history)
    )


def parse_patch_output(raw: str) -> str | None:
    """Return the patched text, or ``None`` for a discard verdict."""
    text = raw.strip()
    if "```" not in text:
        return None if not text or text.upper().startswith("DISCARD") else text
    start = text.find("```") + 3
    end = text.rfind("```")
    if end <= start:
        return None
    body = text[start:end]
    # Drop an optional language tag on the opening fence (same rule as the instruction parser).
    tag = re.match(r"^\S*\n", body)
    if tag:
        body = body[tag.end() :]
    body = body.strip()
    return body or None


def run_patch(lm: LanguageModel, jobs_per_item: list[list[PatchJob]]) -> list[PatchResult]:
    results: list[PatchResult] = []
    for jobs in jobs_per_item:
        texts: dict[str, str] = {}
        raws: dict[str, str] = {}
        discard = False
        for job in jobs:
            raw = lm(render_patch_prompt(job))
            raws[job.component] = raw
            patched = parse_patch_output(raw)
            if patched is None:
                discard = True
                break
            texts[job.component] = patched
        results.append(PatchResult({} if discard else texts, raws))
    return results
