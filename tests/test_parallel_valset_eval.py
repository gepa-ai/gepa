# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""End-to-end test for batched full-valset eval of multiple accepted candidates.

A multi-proposal sampler (``IndependentSampling``) can accept several candidates
in one iteration. Their full-valset evaluations are batched into a single
``adapter.batch_evaluate`` call (where any parallelism lives), while the engine
stays single-threaded and pool/Pareto mutation stays sequential.
"""

from gepa.core.adapter import EvaluationBatch
from gepa.strategies.proposal_sampling import IndependentSampling, SingleMutationSampling
from gepa.strategies.proposal_selection import AllImprovements


class ImprovingAdapter:
    """Each proposal lengthens the prompt; score grows with length, so the
    proposed candidate beats its parent and is accepted."""

    def __init__(self):
        self.propose_new_texts = self._propose_new_texts
        self._suffix = 0

    def evaluate(self, batch, candidate, capture_traces=False):
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


class BatchImprovingAdapter(ImprovingAdapter):
    """Adds a real ``batch_evaluate`` and records the items of each call."""

    def __init__(self):
        super().__init__()
        self.batch_calls: list[int] = []

    def batch_evaluate(self, items):
        self.batch_calls.append(len(items))
        return [self.evaluate(batch, candidate, capture_traces=True) for candidate, batch in items]


def _run(adapter, sampling_strategy, tmp_path):
    import gepa

    return gepa.optimize(
        seed_candidate={"system_prompt": "s"},
        trainset=[{"id": i} for i in range(4)],
        valset=[{"id": i} for i in range(4)],
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


def test_accepted_candidates_share_one_batch_evaluate_call(tmp_path):
    """When the adapter implements batch_evaluate, the valset evals of all
    candidates accepted in one iteration go out as a single batched call."""
    adapter = BatchImprovingAdapter()
    result = _run(adapter, IndependentSampling(3), tmp_path)
    assert len(result.candidates) > 2
    # At least one iteration batched several candidates into one batch_evaluate call.
    assert any(n > 1 for n in adapter.batch_calls)


def test_single_mutation_still_works(tmp_path):
    """Default single-proposal path (one accepted candidate) is unaffected."""
    result = _run(ImprovingAdapter(), SingleMutationSampling(), tmp_path)
    assert len(result.candidates) >= 1
    assert result.total_metric_calls > 0
