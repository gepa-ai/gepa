# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Windowed test-before-train optimization using the existing GEPA optimizer."""

from __future__ import annotations

import inspect
import os
import pickle
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, cast

from gepa.api import optimize
from gepa.core.adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from gepa.core.data_loader import DataId, DataLoader, ensure_loader


@dataclass(frozen=True)
class OnlineWindowResult(Generic[DataId]):
    """A completed window, scored with ``candidate`` before its feedback was used.

    ``next_candidate`` is selected using already observed data. Its training
    score is deliberately not included in the online performance measure.
    """

    example_ids: list[DataId]
    candidate: dict[str, str]
    scores: list[float]
    next_candidate: dict[str, str]
    evaluation_metric_calls: int
    optimization_metric_calls: int


@dataclass(frozen=True)
class OnlineResult(Generic[DataId]):
    """Online scores and the candidate to use for the next unseen window."""

    candidate: dict[str, str]
    windows: list[OnlineWindowResult[DataId]]

    @property
    def num_examples(self) -> int:
        return sum(len(window.scores) for window in self.windows)

    @property
    def online_score(self) -> float | None:
        """Sample-weighted pre-update score; ``None`` when no data was processed."""
        count = self.num_examples
        return sum(sum(window.scores) for window in self.windows) / count if count else None

    @property
    def total_metric_calls(self) -> int:
        return sum(window.evaluation_metric_calls + window.optimization_metric_calls for window in self.windows)


@dataclass
class _Checkpoint(Generic[DataId]):
    seed_candidate: dict[str, str]
    window_size: int
    history_size: int
    max_metric_calls_per_window: int
    candidate: dict[str, str]
    windows: list[OnlineWindowResult[DataId]] = field(default_factory=list)
    pending: OnlineWindowResult[DataId] | None = None


def _save_checkpoint(state: _Checkpoint, directory: Path | None) -> None:
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as file:
            temporary = file.name
            pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, directory / "online_state.bin")
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def optimize_online(
    seed_candidate: dict[str, str],
    data: list[DataInst] | DataLoader[DataId, DataInst],
    adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
    *,
    window_size: int = 20,
    max_metric_calls_per_window: int,
    history_size: int | None = None,
    optimize_kwargs: Mapping[str, Any] | None = None,
    run_dir: str | None = None,
    max_windows: int | None = None,
) -> OnlineResult[DataId]:
    """Evaluate each window before allowing GEPA to learn from it.

    Unlike an ``EvaluationPolicy``, this controller also owns when examples
    become available to the proposer. It does not change the offline engine.

    Args:
        seed_candidate: Initial program, used unchanged for the first window.
        data: An ordered dataset. IDs must be unique and stable, with immutable
            examples. Between calls with the same ``run_dir``, append examples;
            do not remove or reorder the previously processed prefix. A list
            uses positions as IDs. IDs are snapshotted at the start of a call.
        adapter: Used for both pre-update evaluation and offline optimization.
            Its evaluation must not itself train on the evaluation feedback.
        window_size: Maximum number of examples per update. A final partial
            window is evaluated and learned from without waiting for more data.
        max_metric_calls_per_window: GEPA's optimization budget for each window,
            excluding the pre-update evaluation. This has the same stopping
            semantics as ``optimize(max_metric_calls=...)``.
        history_size: Number of most recent observed examples available to an
            update. Defaults to ``window_size``; must be at least that large.
        optimize_kwargs: Other ``optimize`` options, such as ``reflection_lm``,
            ``custom_candidate_proposer``, or ``reflection_minibatch_size``.
            Dataset, adapter, validation policy, budget, and run directory are
            owned by this controller and cannot be overridden.
        run_dir: Optional, dedicated, single-writer checkpoint directory. Only
            resume checkpoints you trust: like GEPA state, they use pickle.
            Keep the adapter and optimizer settings consistent when resuming
            an interrupted update. A persisted pre-update evaluation is reused;
            a call interrupted before persistence may be repeated. This does
            not provide exactly-once execution of external API calls.
        max_windows: Optional limit on updates in this invocation, including a
            pending update resumed from disk. Later calls can process more data.

    Returns:
        All completed windows and their sample-weighted pre-update score. The
        returned candidate has seen the last window and is for subsequent data,
        not a candidate whose score equals ``online_score``. Checkpoint and
        result metadata grow with the number of evaluated examples; training
        history is bounded by ``history_size``.
    """
    if window_size <= 0 or max_metric_calls_per_window <= 0:
        raise ValueError("window_size and max_metric_calls_per_window must be positive")
    history_size = window_size if history_size is None else history_size
    if history_size < window_size:
        raise ValueError("history_size must be at least window_size")
    if max_windows is not None and max_windows < 0:
        raise ValueError("max_windows must be nonnegative")
    options = dict(optimize_kwargs or {})
    reserved = {
        "seed_candidate",
        "trainset",
        "valset",
        "adapter",
        "run_dir",
        "max_metric_calls",
        "val_evaluation_policy",
    }
    if reserved & options.keys():
        raise ValueError(f"optimize_kwargs cannot override {sorted(reserved & options.keys())}")
    inspect.signature(optimize).bind_partial(**options)

    loader = ensure_loader(data)
    ids = list(loader.all_ids())
    if len(set(ids)) != len(ids):
        raise ValueError("Online data IDs must be unique")
    directory = Path(run_dir) if run_dir is not None else None
    state: _Checkpoint[DataId] = _Checkpoint(
        dict(seed_candidate), window_size, history_size, max_metric_calls_per_window, dict(seed_candidate)
    )
    checkpoint = directory / "online_state.bin" if directory is not None else None
    if checkpoint is not None and checkpoint.exists():
        with checkpoint.open("rb") as file:
            state = pickle.load(file)
        if (
            state.seed_candidate != seed_candidate
            or state.window_size != window_size
            or state.history_size != history_size
            or state.max_metric_calls_per_window != max_metric_calls_per_window
        ):
            raise ValueError("Online checkpoint seed, window, history, and budget must match")
    observed_ids = [example_id for window in state.windows for example_id in window.example_ids]
    known_ids = observed_ids + (state.pending.example_ids if state.pending is not None else [])
    if ids[: len(known_ids)] != known_ids:
        raise ValueError("Online data must preserve the checkpoint's processed prefix")

    completed = 0
    while len(observed_ids) < len(ids) and (max_windows is None or completed < max_windows):
        if state.pending is None:
            window_ids = ids[len(observed_ids) : len(observed_ids) + window_size]
            candidate = dict(state.candidate)
            evaluation = adapter.evaluate(loader.fetch(window_ids), dict(candidate), capture_traces=False)
            if len(evaluation.scores) != len(window_ids):
                raise ValueError("Online evaluation must return one score per example")
            state.pending = OnlineWindowResult(
                list(window_ids),
                candidate,
                list(evaluation.scores),
                {},
                evaluation.num_metric_calls if evaluation.num_metric_calls is not None else len(window_ids),
                0,
            )
            # Commit the test result before any training or candidate selection.
            _save_checkpoint(state, directory)

        pending = state.pending
        history_ids = (observed_ids + pending.example_ids)[-history_size:]
        history = loader.fetch(history_ids)
        update_dir = str(directory / f"update_{len(state.windows):06d}") if directory is not None else None
        result = optimize(
            seed_candidate=dict(pending.candidate),
            trainset=history,
            adapter=adapter,
            max_metric_calls=max_metric_calls_per_window,
            run_dir=update_dir,
            **options,
        )
        next_candidate = dict(cast(dict[str, str], result.best_candidate))
        state.windows.append(
            OnlineWindowResult(
                pending.example_ids,
                pending.candidate,
                pending.scores,
                next_candidate,
                pending.evaluation_metric_calls,
                result.total_metric_calls or 0,
            )
        )
        state.candidate = next_candidate
        observed_ids.extend(pending.example_ids)
        state.pending = None
        _save_checkpoint(state, directory)
        completed += 1

    return OnlineResult(dict(state.candidate), state.windows)
