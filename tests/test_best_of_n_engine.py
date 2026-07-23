# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Regression tests for the best_of_n engine's dataset scoring.

Two bugs, both in the ``run()`` eval block:

1. Over-reporting on budget exhaustion. When the eval budget ran out
   *mid-dataset-loop*, ``BudgetExhausted`` escaped ``run()`` and the api
   fallback reported ``server.best_score`` — the single highest *per-example*
   score — instead of the engine's candidate-aggregate (mean-over-dataset)
   best. A run that never had one candidate clear the whole dataset could
   still report a per-example max it never achieved as an aggregate.

2. Crash on a val-only task. A task with a ``val_set`` but no ``train_set``
   took the no-example ``server.evaluate(candidate)`` branch, which calls the
   dataset evaluator with no example and raises ``TypeError``.
"""

from __future__ import annotations

import gepa.oa.engines.best_of_n as bon
from gepa.oa.budget import BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.eval_server import EvalServer
from gepa.oa.task import Task


class _FakeLM:
    """Stand-in for ``gepa.lm.LM``: returns a distinct fenced candidate per
    call and carries a ``total_cost`` attribute (read by the engine)."""

    def __init__(self, *args, **kwargs):
        self.total_cost = 0.0
        self._n = 0

    def __call__(self, prompt):
        self._n += 1
        return f"```\ncandidate-{self._n}\n```"


def _engine(monkeypatch, *, max_evals, engine_config):
    monkeypatch.setattr(bon, "LM", _FakeLM)
    config = OptimizeAnythingConfig(
        engine="best_of_n",
        max_evals=max_evals,
        engine_config=engine_config,
    )
    return bon.BestOfNEngine(config)


def test_budget_exhaustion_reports_aggregate_not_per_example_max(monkeypatch):
    """When the budget runs out mid-dataset, the engine must return the best
    fully-evaluated candidate's *aggregate* score — never the server's single
    highest per-example score."""
    train = [{"id": f"e{i}"} for i in range(4)]
    task = Task(name="bon-budget", seed_candidate="seed", train_set=train)

    # e0 scores 1.0, the rest 0.0 → per-example max = 1.0, dataset mean = 0.25.
    def eval_fn(candidate, example):
        return (1.0 if example["id"] == "e0" else 0.0), {}

    # 6 evals: candidate 1 scores the whole set (4 evals, mean 0.25); candidate
    # 2 exhausts mid-loop (evals 5 and 6, then check() raises on the 3rd ex).
    budget = BudgetTracker(max_evals=6)
    server = EvalServer(task, evaluate=eval_fn, budget=budget)
    engine = _engine(monkeypatch, max_evals=6, engine_config={"max_n": 10})

    result = engine.run(task, server)

    assert server.best_score == 1.0  # the server did see a 1.0 per-example
    assert result.best_score == 0.25  # but the engine reports the aggregate best
    assert budget.used == 6  # budget was actually exhausted


def test_val_only_task_scores_over_val_set(monkeypatch):
    """A task with only a ``val_set`` (no ``train_set``) must be scored over the
    val examples, not routed through the no-example evaluate() branch (which
    would raise TypeError in a dataset evaluator)."""
    val = [{"v": 0.2}, {"v": 0.8}]
    task = Task(name="bon-valonly", seed_candidate="seed", val_set=val)

    def eval_fn(candidate, example):
        return float(example["v"]), {}

    budget = BudgetTracker(max_evals=100)
    server = EvalServer(task, evaluate=eval_fn, budget=budget)
    engine = _engine(monkeypatch, max_evals=100, engine_config={"max_n": 3})

    result = engine.run(task, server)

    # eval_fn ignores the candidate, so every sample scores mean(0.2, 0.8)=0.5.
    assert result.best_score == 0.5
    assert budget.used == 3 * len(val)  # 3 samples × 2 val examples, no crash
