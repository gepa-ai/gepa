# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for the evaluation caching functionality."""

from pathlib import Path

import pytest

from gepa.core.state import EvaluationCache

# RECORDER_DIR paths for cached tests (imported lazily to avoid module conflicts)
# Source: tests/test_aime_prompt_optimization/test_aime_prompt_optimize.py
AIME_RECORDER_DIR = Path(__file__).parent / "test_aime_prompt_optimization"
# Source: tests/test_pareto_frontier_types/test_pareto_frontier_types.py
PARETO_RECORDER_DIR = Path(__file__).parent / "test_pareto_frontier_types"


class TestEvaluationCache:
    """Tests for EvaluationCache class."""

    def test_empty_cache_returns_none(self):
        """Get on empty cache should return None."""
        cache: EvaluationCache = EvaluationCache()
        assert cache.get({"prompt": "test"}, "example_1") is None

    def test_put_and_get_basic(self):
        """Basic put and get functionality."""
        cache: EvaluationCache = EvaluationCache()
        cache.put({"prompt": "test"}, "example_1", "output1", 0.8)
        result = cache.get({"prompt": "test"}, "example_1")
        assert result is not None
        assert result.output == "output1"
        assert result.score == 0.8

    def test_different_examples_separate_entries(self):
        """Different examples should have separate cache entries."""
        cache: EvaluationCache = EvaluationCache()
        candidate = {"prompt": "test"}
        cache.put(candidate, "ex1", "out1", 0.5)
        cache.put(candidate, "ex2", "out2", 0.7)
        assert cache.get(candidate, "ex1").output == "out1"
        assert cache.get(candidate, "ex2").output == "out2"

    def test_different_candidates_separate_entries(self):
        """Different candidates should have separate cache entries."""
        cache: EvaluationCache = EvaluationCache()
        cache.put({"prompt": "test1"}, "ex1", "out1", 0.5)
        cache.put({"prompt": "test2"}, "ex1", "out2", 0.7)
        assert cache.get({"prompt": "test1"}, "ex1").output == "out1"
        assert cache.get({"prompt": "test2"}, "ex1").output == "out2"

    def test_get_batch(self):
        """Test batch get functionality."""
        cache: EvaluationCache = EvaluationCache()
        candidate = {"prompt": "test"}
        cache.put(candidate, "ex1", "out1", 0.5)
        cache.put(candidate, "ex2", "out2", 0.6)
        cached_results, uncached_ids = cache.get_batch(candidate, ["ex1", "ex2", "ex3"])
        assert "ex1" in cached_results and "ex2" in cached_results
        assert uncached_ids == ["ex3"]

    def test_put_batch(self):
        """Test batch put functionality."""
        cache: EvaluationCache = EvaluationCache()
        candidate = {"prompt": "test"}
        cache.put_batch(candidate, ["ex1", "ex2"], ["out1", "out2"], [0.5, 0.6], [{"acc": 0.9}, {"acc": 0.8}])
        assert cache.get(candidate, "ex1").score == 0.5
        assert cache.get(candidate, "ex2").objective_scores == {"acc": 0.8}


class TestEvaluationCacheIntegration:
    """Integration tests for evaluation cache with optimize function."""

    @staticmethod
    def _create_dummy_adapter():
        """Create a dummy adapter for testing."""
        from gepa.core.adapter import EvaluationBatch

        class DummyAdapter:
            def __init__(self):
                self.propose_new_texts = self._propose_new_texts

            def evaluate(self, batch, candidate, capture_traces=False):
                weight = hash(candidate.get("system_prompt", "")) % 10
                outputs = [{"id": i, "weight": weight} for i in range(len(batch))]
                scores = [(weight + 1) / 10 for _ in batch]
                trajectories = [{"score": s} for s in scores] if capture_traces else None
                return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

            def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
                return dict.fromkeys(components_to_update, [{"score": s} for s in eval_batch.scores])

            def _propose_new_texts(self, candidate, reflective_dataset, components_to_update):
                return dict.fromkeys(components_to_update, f"{candidate.get('system_prompt', '')} v2")

        return DummyAdapter()

    def test_optimize_with_caching_enabled(self, tmp_path):
        """Test that optimize runs correctly with caching enabled."""
        import gepa

        result = gepa.optimize(
            seed_candidate={"system_prompt": "test"},
            trainset=[{"id": i} for i in range(5)],
            valset=[{"id": i} for i in range(5)],
            adapter=self._create_dummy_adapter(),
            max_metric_calls=20,
            reflection_lm=None,
            run_dir=str(tmp_path / "cache_test"),
            cache_evaluation=True,
        )
        assert result is not None and result.total_metric_calls > 0

    def test_caching_does_not_break_optimization(self, tmp_path):
        """Verify that caching doesn't break the optimization process."""
        import gepa

        trainset = [{"id": i} for i in range(5)]
        valset = [{"id": i} for i in range(5)]
        seed = {"system_prompt": "test"}

        result_no_cache = gepa.optimize(
            seed_candidate=seed,
            trainset=trainset,
            valset=valset,
            adapter=self._create_dummy_adapter(),
            max_metric_calls=20,
            reflection_lm=None,
            run_dir=str(tmp_path / "no_cache"),
            cache_evaluation=False,
        )
        result_with_cache = gepa.optimize(
            seed_candidate=seed,
            trainset=trainset,
            valset=valset,
            adapter=self._create_dummy_adapter(),
            max_metric_calls=20,
            reflection_lm=None,
            run_dir=str(tmp_path / "with_cache"),
            cache_evaluation=True,
        )
        assert result_no_cache.total_metric_calls > 0 and result_with_cache.total_metric_calls > 0


@pytest.fixture(scope="module")
def recorder_dir():
    """
    Provides the path to the AIME test recording directory.
    Source: tests/test_aime_prompt_optimization/test_aime_prompt_optimize.py
    """
    AIME_RECORDER_DIR.mkdir(parents=True, exist_ok=True)
    return AIME_RECORDER_DIR


@pytest.fixture(scope="module")
def pareto_recorder_dir():
    """
    Provides the path to the Pareto frontier test recording directory.
    Source: tests/test_pareto_frontier_types/test_pareto_frontier_types.py
    """
    PARETO_RECORDER_DIR.mkdir(parents=True, exist_ok=True)
    return PARETO_RECORDER_DIR


def test_aime_prompt_optimize_with_cache(mocked_lms, recorder_dir):
    """
    Tests the GEPA optimization process with evaluation caching enabled.
    Uses the same recorded/replayed LLM calls as the non-cached test.
    """
    import gepa
    from gepa.adapters.default_adapter.default_adapter import DefaultAdapter

    task_lm, reflection_lm = mocked_lms
    adapter = DefaultAdapter(model=task_lm)

    trainset, valset, _ = gepa.examples.aime.init_dataset()
    trainset = trainset[:10]
    valset = valset[:10]

    seed_prompt = {
        "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
    }

    # Run with cache_evaluation=True
    gepa_result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        max_metric_calls=4,
        reflection_lm=reflection_lm,
        display_progress_bar=True,
        cache_evaluation=True,
    )

    # Verify the optimization completed and produced a valid result
    assert gepa_result is not None
    assert gepa_result.total_metric_calls is not None and gepa_result.total_metric_calls > 0
    assert gepa_result.total_metric_calls == 10
    best_prompt = gepa_result.best_candidate["system_prompt"]
    assert isinstance(best_prompt, str) and len(best_prompt) > 0
    # With caching, we may use fewer metric calls since cached results are reused
    # The result may differ slightly from non-cached due to different eval counts
    # affecting stopping conditions, but should still be a valid optimization result


# --- Pareto Frontier Tests with Cache ---
# Source: tests/test_pareto_frontier_types/test_pareto_frontier_types.py

# Import the helper from conftest to avoid duplicating fixture logic
from conftest import create_mocked_lms_context


@pytest.fixture(scope="module")
def pareto_mocked_lms(pareto_recorder_dir):
    """
    Fixture for Pareto frontier tests using their own recorder directory.
    Re-uses the mocked_lms helper from conftest.py with the pareto recorder.
    """
    yield from create_mocked_lms_context(pareto_recorder_dir)


@pytest.mark.parametrize("frontier_type", ["objective", "hybrid", "instance"])
def test_pareto_frontier_type_with_cache(pareto_mocked_lms, pareto_recorder_dir, frontier_type):
    """
    End-to-end test of GEPA optimization on the PUPA dataset with caching enabled.
    Source: tests/test_pareto_frontier_types/test_pareto_frontier_types.py
    """
    # Import init_pupa_dataset lazily to avoid module name conflicts with pytest
    import random

    from datasets import load_dataset

    import gepa
    from gepa.adapters.default_adapter.default_adapter import (
        DefaultAdapter,
        DefaultDataInst,
        EvaluationResult,
    )

    def init_pupa_dataset():
        raw_ds = load_dataset("Columbia-NLP/PUPA", "pupa_tnb")["train"]

        def _to_inst(item):
            return {
                "input": item["user_query"],
                "additional_context": {
                    "predicted_category": str(item.get("predicted_category", "")),
                    "pii_units": str(item.get("pii_units", "")),
                    "target_response": str(item.get("target_response", "")),
                    "redacted_query": str(item.get("redacted_query", "")),
                },
                "answer": str(item["redacted_query"]),
            }

        data = [_to_inst(item) for item in raw_ds]
        rng = random.Random(0)
        rng.shuffle(data)

        mid = len(data) // 2
        trainset = data[:mid]
        valset = data[mid:]
        testset = data[: min(20, len(data))]

        return trainset, valset, testset

    task_lm, reflection_lm = pareto_mocked_lms

    def evaluator(data: DefaultDataInst, response: str) -> EvaluationResult:
        judge_prompt = (
            "You are a strict grader. Compare the assistant response to the gold redaction.\n"
            f"GOLD:\n{data['answer'].strip()}\n\nRESPONSE:\n{response.strip()}\n\n"
            "Return only a number between 0 and 1."
        )
        quality_str = reflection_lm(judge_prompt)
        try:
            quality = float(quality_str.strip())
        except ValueError:
            quality = 0.0

        pii_units = data["additional_context"].get("pii_units", "")
        pii_list = [p.strip() for p in pii_units.split("||") if p.strip()]
        leaked = sum(1 for pii in pii_list if pii and pii in response)
        leakage_frac = leaked / len(pii_list) if pii_list else 0.0
        leakage_score = 1.0 - leakage_frac

        total_score = (quality + leakage_score) / 2

        if total_score > 0.0:
            feedback = f"The generated response is correct. The response include the correct answer '{data['answer']}'"
        else:
            additional_context_str = "\n".join(f"{k}: {v}" for k, v in data["additional_context"].items())
            feedback = f"The generated response is incorrect. The correct answer is '{data['answer']}'. Ensure that the correct answer is included in the response exactly as it is. Here is some additional context that might be helpful:\n{additional_context_str}"
        return EvaluationResult(
            score=total_score,
            feedback=feedback,
            objective_scores={"quality": quality, "leakage": leakage_score},
        )

    adapter = DefaultAdapter(model=task_lm, evaluator=evaluator)

    trainset, valset, _ = init_pupa_dataset()
    trainset = trainset[:20]
    valset = valset[:12]

    seed_prompt = {"system_prompt": "You are a helpful assistant."}

    gepa_result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        frontier_type=frontier_type,
        max_metric_calls=10,
        reflection_minibatch_size=3,
        display_progress_bar=False,
        cache_evaluation=True,
    )
    assert gepa_result.total_metric_calls == 12
    assert gepa_result is not None
    assert gepa_result.total_metric_calls is not None and gepa_result.total_metric_calls > 0
    best_prompt = gepa_result.best_candidate["system_prompt"]
    assert isinstance(best_prompt, str) and len(best_prompt) > 0
