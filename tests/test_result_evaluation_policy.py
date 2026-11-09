# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json

import pytest

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.core.result import GEPAResult
from gepa.gepa_launcher import EngineConfig, GEPAConfig, ReflectionConfig
from gepa.oa.config import OptimizeAnythingConfig
from gepa.optimize_anything import optimize_anything
from gepa.strategies.eval_policy import FullEvaluationPolicy


class _KeepSeed(FullEvaluationPolicy):
    def get_best_program(self, state):
        return 0


class _IndexLike:
    def __index__(self):
        return 0


class _KeepSeedIndexLike(_KeepSeed):
    def get_best_program(self, state):
        return _IndexLike()


class _LiveStateOnly(_KeepSeed):
    """A policy using runtime state that cannot be reconstructed from a checkpoint."""

    def __init__(self):
        self.live_state = None
        self.recovery_failed = False

    def get_best_program(self, state):
        if self.live_state is None:
            self.live_state = state
        if state is not self.live_state:
            self.recovery_failed = True
            raise LookupError("runtime policy context is unavailable")
        return super().get_best_program(state)


class _RecordWinner:
    def __init__(self):
        self.winners = []

    def on_optimization_end(self, event):
        self.winners.append(event["best_candidate_idx"])


def _score(candidate, example):
    return (0.3 if candidate == "seed" else 0.9), {"example": example}


class _Adapter:
    def evaluate(self, batch, candidate, capture_traces=False):
        scores = [_score(candidate["prompt"], item)[0] for item in batch]
        return EvaluationBatch(
            outputs=[candidate["prompt"]] * len(batch),
            scores=scores,
            trajectories=[{"score": score} for score in scores] if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {name: [{"score": eval_batch.scores[0]}] for name in components_to_update}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        return dict.fromkeys(components_to_update, "mutated")


@pytest.mark.parametrize("policy, winner", [(FullEvaluationPolicy(), 1), (_KeepSeed(), 0), (_KeepSeedIndexLike(), 0)])
def test_optimize_result_and_roundtrip_use_terminal_policy(policy, winner):
    observer = _RecordWinner()
    result = gepa.optimize(
        seed_candidate={"prompt": "seed"},
        trainset=["train"],
        valset=["val"],
        adapter=_Adapter(),
        reflection_minibatch_size=1,
        max_metric_calls=4,
        display_progress_bar=False,
        val_evaluation_policy=policy,
        callbacks=[observer],
    )

    assert [winner.__index__() for winner in observer.winners] == [winner]
    assert result.val_aggregate_scores == [0.3, 0.9]
    assert type(result.best_idx) is int
    assert result.best_idx == winner
    assert result.best_score == result.val_aggregate_scores[winner]
    assert result.best_candidate == result.candidates[winner]

    serialized = json.loads(json.dumps(result.to_dict()))
    assert GEPAResult.from_dict(serialized).best_candidate == result.best_candidate
    serialized.pop("best_idx")
    assert GEPAResult.from_dict(serialized).best_idx == 1
    serialized["best_idx"] = len(result.candidates)
    assert GEPAResult.from_dict(serialized).best_idx == 1


@pytest.mark.parametrize("mode", ["legacy", "modern", "recover", "recover_unavailable_policy"])
def test_optimize_anything_respects_policy_across_result_paths(tmp_path, mode):
    observer = _RecordWinner()
    policy = _LiveStateOnly() if mode == "recover_unavailable_policy" else _KeepSeed()
    recovers = mode.startswith("recover")
    budget = 5 if recovers else 4
    engine = EngineConfig(
        val_evaluation_policy=policy,
        max_metric_calls=budget,
        display_progress_bar=False,
        run_dir=str(tmp_path / "run"),
    )
    proposals = iter(["mutated", "interrupted"])
    reflection = ReflectionConfig(reflection_lm=lambda _: next(proposals), reflection_minibatch_size=1)
    if mode == "legacy":
        config = GEPAConfig(engine=engine, reflection=reflection, callbacks=[observer])
    else:
        config = OptimizeAnythingConfig(
            engine="gepa",
            max_evals=budget,
            output_dir=str(tmp_path / "output"),
            engine_config={"engine": engine, "reflection": reflection, "callbacks": [observer]},
        )
    result = optimize_anything(
        seed_candidate="seed", evaluator=_score, dataset=["train"], valset=["val"], config=config
    )

    assert result.total_evals == budget
    assert result.val_aggregate_scores == [0.3, 0.9]
    assert observer.winners == ([] if recovers else [0])
    if isinstance(policy, _LiveStateOnly):
        assert policy.recovery_failed
        assert result.best_candidate == "mutated"
    else:
        assert result.best_candidate == "seed"
