# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""End-to-end tests for trace-aware batched evaluation.

A multi-proposal sampler (``IndependentSampling``) can accept several candidates
in one iteration. Their full-valset evaluations are batched into a single
``adapter.batch_evaluate`` call (where any parallelism lives), while the engine
stays single-threaded and pool/Pareto mutation stays sequential.
"""

import pytest

from gepa.core.adapter import EvaluationBatch, invoke_batch_evaluate
from gepa.strategies.proposal_sampling import IndependentSampling, SingleMutationSampling
from gepa.strategies.proposal_selection import AllImprovements


class ImprovingAdapter:
    """Each proposal lengthens the prompt; score grows with length, so the
    proposed candidate beats its parent and is accepted."""

    def __init__(self):
        self.propose_new_texts = self._propose_new_texts
        self._suffix = 0
        self.evaluate_calls: list[tuple[str, str, bool]] = []

    def evaluate(self, batch, candidate, capture_traces=False):
        self.evaluate_calls.append((batch[0]["split"], candidate["system_prompt"], capture_traces))
        score = min(1.0, len(candidate["system_prompt"]) / 80.0)
        outputs = [score for _ in batch]
        scores = [score for _ in batch]
        trajectories = [{"score": score} for _ in batch] if capture_traces else None
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {c: [{"score": s} for s in eval_batch.scores] for c in components_to_update}

    def _propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        out = {}
        for c in components_to_update:
            self._suffix += 1
            out[c] = candidate.get(c, "") + f" w{self._suffix}"  # distinct + longer => improves
        return out


class TraceAwareBatchImprovingAdapter(ImprovingAdapter):
    def __init__(self):
        super().__init__()
        self.batch_calls: list[tuple[str, int, bool]] = []

    def batch_evaluate(self, items, *, capture_traces=True):
        self.batch_calls.append((items[0][1][0]["split"], len(items), capture_traces))
        return [
            self.evaluate(batch, candidate, capture_traces=capture_traces) for candidate, batch in items
        ]


def _run(adapter, sampling_strategy, tmp_path):
    import gepa

    return gepa.optimize(
        seed_candidate={"system_prompt": "s"},
        trainset=[{"id": i, "split": "train"} for i in range(4)],
        valset=[{"id": i, "split": "val"} for i in range(4)],
        adapter=adapter,
        max_metric_calls=300,
        reflection_lm=None,
        sampling_strategy=sampling_strategy,
        selection_strategy=AllImprovements(),
        run_dir=str(tmp_path),
    )


def test_multiple_candidates_accepted_and_evaluated_in_one_iteration(tmp_path):
    """IndependentSampling(3) + AllImprovements accepts several candidates per
    iteration; all are full-eval'd and added to the pool."""
    result = _run(ImprovingAdapter(), IndependentSampling(3), tmp_path)
    # Seed plus multiple accepted candidates (3 per iteration when all improve).
    assert len(result.candidates) > 2
    assert result.total_metric_calls > 0
    # Best discovered candidate is at least as good as the seed.
    assert max(result.val_aggregate_scores) >= result.val_aggregate_scores[0]


def test_trace_aware_adapter_batches_accepted_candidates_without_traces(tmp_path):
    adapter = TraceAwareBatchImprovingAdapter()
    result = _run(adapter, IndependentSampling(3), tmp_path)
    assert len(result.candidates) > 2
    assert adapter.batch_calls[0] == ("val", 1, False)
    assert any(split == "val" and n > 1 and capture_traces is False for split, n, capture_traces in adapter.batch_calls)
    assert any(split == "train" and capture_traces is True for split, _, capture_traces in adapter.batch_calls)


def test_legacy_batch_adapter_gets_clear_contract_error():
    class LegacyAdapter(ImprovingAdapter):
        def batch_evaluate(self, items):
            return []

    with pytest.raises(TypeError, match="unexpected keyword argument 'capture_traces'"):
        invoke_batch_evaluate(
            LegacyAdapter(),
            [({"system_prompt": "s"}, [{"split": "train"}])],
            capture_traces=True,
        )


def test_batch_error_without_budget_progress_does_not_loop_when_exceptions_are_suppressed(tmp_path):
    class BrokenAdapter(ImprovingAdapter):
        def __init__(self):
            super().__init__()
            self.batch_calls = 0

        def batch_evaluate(self, items, *, capture_traces=True):
            self.batch_calls += 1
            if capture_traces:
                raise TypeError("adapter bug")
            return [
                self.evaluate(batch, candidate, capture_traces=False)
                for candidate, batch in items
            ]

    adapter = BrokenAdapter()
    with pytest.raises(TypeError, match="adapter bug"):
        import gepa

        gepa.optimize(
            seed_candidate={"system_prompt": "s"},
            trainset=[{"id": 0, "split": "train"}],
            valset=[{"id": 0, "split": "val"}],
            adapter=adapter,
            max_metric_calls=20,
            reflection_lm=None,
            raise_on_exception=False,
            run_dir=str(tmp_path),
        )

    assert adapter.batch_calls == 2  # One seed evaluation, then one failed reflective evaluation.


def test_runtime_batch_type_error_is_not_retried_sequentially():
    class BrokenAdapter:
        def __init__(self):
            self.batch_calls = 0
            self.evaluate_calls = 0

        def batch_evaluate(self, items, *, capture_traces=True):
            self.batch_calls += 1
            raise TypeError("adapter bug")

        def evaluate(self, batch, candidate, capture_traces=False):
            self.evaluate_calls += 1
            return EvaluationBatch(outputs=["unexpected"], scores=[0.0])

    adapter = BrokenAdapter()
    with pytest.raises(TypeError, match="adapter bug"):
        invoke_batch_evaluate(adapter, [({"system_prompt": "s"}, [{"split": "val"}])], capture_traces=False)
    assert adapter.batch_calls == 1
    assert adapter.evaluate_calls == 0


def test_issue_428_call_sequence_ends_with_untraced_accepted_val(tmp_path):
    class ReproAdapter:
        def __init__(self):
            self.calls = []
            self.propose_new_texts = self.propose

        def evaluate(self, batch, candidate, capture_traces=False):
            self.calls.append((batch[0]["split"], candidate["prompt"], capture_traces))
            score = 0.1 * len(candidate["prompt"])
            return EvaluationBatch(
                outputs=["ok"] * len(batch),
                scores=[score] * len(batch),
                trajectories=[{}] * len(batch) if capture_traces else None,
            )

        def make_reflective_dataset(self, candidate, evaluation, components):
            return {component: [{}] for component in components}

        def propose(self, candidate, dataset, components):
            return {"prompt": candidate["prompt"] + "x"}

    import gepa

    adapter = ReproAdapter()
    gepa.optimize(
        seed_candidate={"prompt": "a"},
        trainset=[{"split": "train"}],
        valset=[{"split": "val"}],
        adapter=adapter,
        max_metric_calls=4,
        reflection_lm=None,
        run_dir=str(tmp_path),
    )

    assert adapter.calls == [
        ("val", "a", False),
        ("train", "a", True),
        ("train", "ax", True),
        ("val", "ax", False),
    ]


def test_single_mutation_still_works(tmp_path):
    """Default single-proposal path (one accepted candidate) is unaffected."""
    result = _run(ImprovingAdapter(), SingleMutationSampling(), tmp_path)
    assert len(result.candidates) >= 1
    assert result.total_metric_calls > 0
