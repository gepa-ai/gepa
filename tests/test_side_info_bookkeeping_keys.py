# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""``_``-prefixed side_info keys are bookkeeping: summed as cost, never shown to the reflection LM."""

from gepa.adapters.optimize_anything_adapter.optimize_anything_adapter import OptimizeAnythingAdapter
from gepa.core.adapter import EvaluationBatch
from gepa.oa.budget import BudgetTracker
from gepa.oa.eval_server import EvalServer
from gepa.oa.task import Task


def test_reflective_dataset_hides_underscore_keys():
    adapter = OptimizeAnythingAdapter(evaluator=lambda c, example=None: (0.0, None, {}), cache_mode="off")
    side_info = {
        "Feedback": "wrong",
        "scores": {"acc": 0.0},
        "_cost": 0.25,
        "_budget": {"used": 1},
        "p_specific_info": {"Detail": "only for p"},
    }
    batch = EvaluationBatch(outputs=[None], scores=[0.0], trajectories=[side_info])
    ds = adapter.make_reflective_dataset({"p": "x", "q": "y"}, batch, ["p", "q"])
    assert ds["p"] == [{"Feedback": "wrong", "Scores (Higher is Better)": {"acc": 0.0}, "Detail": "only for p"}]
    assert ds["q"] == [{"Feedback": "wrong", "Scores (Higher is Better)": {"acc": 0.0}}]


def test_eval_server_sums_underscore_cost():
    server = EvalServer(
        Task(name="t", seed_candidate="s"), lambda candidate: (1.0, {"_cost": 0.5}), BudgetTracker(max_evals=3)
    )
    server.evaluate("a")
    server.evaluate("b")
    assert server.total_cost == 1.0
