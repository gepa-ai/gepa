"""End-to-end: the engine's crossover slot fires and lands an accepted crossover.

A deterministic adapter scores an instruction 1.0 on an example when the
instruction contains that example's required token ("ALPHA" for type-A inputs,
"BETA" for type-B), else 0.2. Reflection is mocked to diversify: given an
instruction containing one token it proposes the other, so the run produces one
candidate strong on type-A inputs and another strong on type-B — a complementary
pair. The crossover synthesis prompt (which contains "VARIANT A") returns an
instruction with BOTH tokens, which wins everywhere and is accepted.
"""

from gepa import optimize
from gepa.core.adapter import EvaluationBatch
from gepa.core.callbacks import GEPACallback


class _TokenAdapter:
    propose_new_texts = None

    def evaluate(self, batch, candidate, capture_traces=False):
        instr = candidate["instructions"]
        outputs, scores, trajectories = [], [], []
        for ex in batch:
            token = "ALPHA" if ex["type"] == "A" else "BETA"
            score = 1.0 if token in instr else 0.2
            outputs.append(instr)
            scores.append(score)
            trajectories.append({"type": ex["type"], "score": score, "instr": instr})
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories if capture_traces else None,
            num_metric_calls=len(batch),
        )

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [
            {
                "Inputs": {"type": t["type"]},
                "Generated Outputs": t["instr"],
                "Feedback": f"Scored {t['score']} on a type-{t['type']} input.",
            }
            for t in (eval_batch.trajectories or [])
        ]
        return {name: list(records) for name in components_to_update}


class _AcceptCapture(GEPACallback):
    def __init__(self):
        self.accepts = []

    def on_candidate_accepted(self, event):
        self.accepts.append(dict(event))


def test_engine_crossover_slot_produces_accepted_crossover():
    data = [{"type": "A"}, {"type": "A"}, {"type": "B"}, {"type": "B"}]

    def reflection_lm(prompt):
        # Crossover synthesis prompt -> combine both strengths.
        if "VARIANT A" in prompt:
            return "```\nInstruction with ALPHA and BETA\n```"
        # Reflective mutation -> diversify toward the missing token.
        if "ALPHA" in prompt:
            return "```\nInstruction with BETA\n```"
        return "```\nInstruction with ALPHA\n```"

    capture = _AcceptCapture()

    result = optimize(
        seed_candidate={"instructions": "plain instruction"},
        trainset=data,
        valset=data,
        adapter=_TokenAdapter(),
        reflection_lm=reflection_lm,
        use_crossover=True,
        max_crossover_invocations=5,
        reflection_minibatch_size=4,  # every minibatch is the whole set -> deterministic
        max_metric_calls=150,
        callbacks=[capture],
        seed=0,
    )

    # A crossover child has two parents AND carries both tokens (the synthesized text).
    crossover_candidates = [
        c
        for c in result.candidates
        if "ALPHA" in c["instructions"] and "BETA" in c["instructions"]
    ]
    assert crossover_candidates, "expected at least one synthesized crossover candidate in the pool"

    two_parent_accepts = [e for e in capture.accepts if len(e["parent_ids"]) == 2]
    assert two_parent_accepts, "expected an accepted proposal with two parents (a crossover)"
