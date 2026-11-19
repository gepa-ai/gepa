import re

import gepa
from gepa.core.adapter import EvaluationBatch
from gepa.strategies.proposal_sampling import SameParentSampling

EXAMPLES = [{"k": k} for k in range(6)]


def _level(text):
    return int(re.search(r"\d+", text).group())


class LevelAdapter:
    """Even examples score component a's level, odd examples component b's."""

    propose_new_texts = None

    def evaluate(self, batch, candidate, capture_traces=False):
        scores = [_level(candidate["a" if ex["k"] % 2 == 0 else "b"]) / 10 for ex in batch]
        return EvaluationBatch(
            outputs=[None] * len(batch),
            scores=scores,
            trajectories=[{"score": s} for s in scores] if capture_traces else None,
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {c: [{"Feedback": str(t["score"])} for t in eval_batch.trajectories] for c in components_to_update}


def _raise_a_then_b():
    calls = []

    def propose(candidate, reflective_dataset, components_to_update):
        calls.append(None)
        name = "a" if len(calls) % 2 else "b"
        return {name: f"{name}{_level(candidate[name]) + 1}"}

    return propose


def test_merges_scheduled_before_the_cap_do_not_exceed_max_merge_invocations():
    # Two children are accepted per iteration, so two merges get scheduled before
    # the first one is accepted.
    merges = []

    class MergeLog:
        def on_merge_accepted(self, event):
            merges.append(event["new_candidate_idx"])

    gepa.optimize(
        seed_candidate={"a": "a0", "b": "b0"},
        trainset=EXAMPLES,
        valset=EXAMPLES,
        adapter=LevelAdapter(),
        custom_candidate_proposer=_raise_a_then_b(),
        sampling_strategy=SameParentSampling(n=2),
        reflection_minibatch_size=len(EXAMPLES),
        skip_perfect_score=False,
        use_merge=True,
        max_merge_invocations=1,
        stop_callbacks=[lambda state: state.i + 1 >= 6],
        callbacks=[MergeLog()],
        seed=0,
    )

    assert len(merges) == 1
