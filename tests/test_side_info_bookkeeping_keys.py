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


def test_refiner_and_reflection_prompts_hide_underscore_keys(tmp_path):
    """With a refiner, each attempt's side_info is shown to the refiner LM and nested in the reflection prompt."""
    from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig, RefinerConfig
    from gepa.optimize_anything import optimize_anything

    reflection_prompts, refiner_prompts = [], []

    def reflection_lm(prompt):
        reflection_prompts.append(str(prompt))
        return "```\nimproved\n```"

    def refiner_lm(prompt):
        refiner_prompts.append(str(prompt))
        return '{"current_candidate": "refined improved"}'

    def evaluator(candidate, example):
        return (1.0 if "improved" in candidate else 0.2), {"Feedback": "visible", "_cost": 0.01, "_note": "hidden"}

    optimize_anything(
        seed_candidate="seed",
        evaluator=evaluator,
        dataset=[1, 2, 3, 4],
        objective="o",
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=16, run_dir=str(tmp_path)),
            reflection=ReflectionConfig(reflection_lm=reflection_lm),
            refiner=RefinerConfig(refiner_lm=refiner_lm),
        ),
    )

    assert reflection_prompts and refiner_prompts
    for prompt in reflection_prompts + refiner_prompts:
        assert "visible" in prompt
        assert "_cost" not in prompt and "_note" not in prompt
