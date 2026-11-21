# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Public ensemble re-scoring must preserve the evaluator's input/return contract."""

import pytest

from gepa.oa.budget import BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engine import Result
from gepa.oa.ensemble import optimize_vote, optimize_vote_with_server
from gepa.oa.eval_server import EvalServer
from gepa.oa.task import Task


class _CandidateEngine:
    """Replace candidate generation only; score through the real eval server."""

    name = "fixed-candidate"

    def __init__(self, candidate):
        self.candidate = candidate

    def run(self, task, server):
        examples = task.train_set or task.val_set
        if examples:
            scores = [server.evaluate(self.candidate, example)[0] for example in examples]
            score = sum(scores) / len(scores)
        else:
            score, _ = server.evaluate(self.candidate)
        return Result(best_candidate=self.candidate, best_score=score)

    def process_result(self, result, output_dir):
        pass


def _configs(tmp_path):
    return [
        OptimizeAnythingConfig(
            engine=_CandidateEngine(candidate),
            max_evals=2,
            output_dir=tmp_path / candidate,
        )
        for candidate in ("weak", "strong")
    ]


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("split", [None, "dataset", "valset"])
def test_vote_normalizes_returns_and_passes_examples(tmp_path, scalar, split):
    calls = []

    def evaluate(candidate, example=None):
        if split is not None:
            assert example in ("one", "two")
        calls.append((candidate, example))
        score = 0.8 if candidate == "strong" else 0.2
        return score if scalar else (score, {})

    kwargs = {split: ["one", "two"]} if split else {}
    result = optimize_vote("seed", evaluator=evaluate, configs=_configs(tmp_path), **kwargs)

    assert result.best_candidate == "strong"
    assert result.metadata["vote_scores"] == pytest.approx([0.2, 0.8])
    # Both engines evaluate first, then the vote scores each candidate again.
    # The vote must not consume or enlarge either engine's eval budget.
    count = 2 if split else 1
    assert len(calls) == 4 * count
    assert [r.total_evals for r in result.metadata["all_results"]] == [count, count]


def test_vote_uses_validation_data_instead_of_training_or_test_data(tmp_path):
    calls = []

    def evaluate(candidate, example):
        calls.append((candidate, example))
        preferred = "strong" if example == "validation" else "weak"
        return float(candidate == preferred), {}

    result = optimize_vote(
        "seed",
        evaluator=evaluate,
        configs=_configs(tmp_path),
        dataset=["train"],
        valset=["validation"],
        test_set=["held-out"],
    )

    assert result.best_candidate == "strong"
    assert result.metadata["vote_scores"] == [0.0, 1.0]
    assert calls[-2:] == [("weak", "validation"), ("strong", "validation")]


def test_vote_compares_dataset_means_instead_of_best_example(tmp_path):
    def evaluate(candidate, example):
        return (float(example == "one") if candidate == "weak" else 0.6), {}

    result = optimize_vote("seed", evaluator=evaluate, configs=_configs(tmp_path), dataset=["one", "two"])

    assert result.best_candidate == "strong"
    assert result.metadata["vote_scores"] == pytest.approx([0.5, 0.6])


@pytest.mark.parametrize("scalar", [False, True])
def test_vote_with_server_preserves_candidate_only_evaluator_and_budgets(tmp_path, scalar):
    def train_evaluate(candidate, example):
        assert example == "training"
        return float(candidate == "weak"), {}

    servers = [
        EvalServer(
            Task(name=f"vote-{i}", seed_candidate="seed", train_set=["training"]),
            train_evaluate,
            BudgetTracker(max_evals=1),
        )
        for i in range(2)
    ]

    def vote_evaluate(candidate):
        score = float(candidate == "strong")
        return score if scalar else (score, {})

    try:
        result = optimize_vote_with_server(servers, _configs(tmp_path), vote_evaluate)
        assert result.best_candidate == "strong"
        assert result.metadata["vote_scores"] == [0.0, 1.0]
        assert [server.budget.used for server in servers] == [1, 1]
    finally:
        for server in servers:
            server.stop()


def test_vote_failed_rescore_is_still_excluded(tmp_path):
    calls = {"weak": 0, "strong": 0}

    def evaluate(candidate):
        calls[candidate] += 1
        if candidate == "weak" and calls[candidate] > 1:
            raise RuntimeError("candidate failed re-scoring")
        return 0.5, {}

    result = optimize_vote("seed", evaluator=evaluate, configs=_configs(tmp_path))

    assert result.best_candidate == "strong"
    assert result.metadata["vote_scores"] == [float("-inf"), 0.5]
