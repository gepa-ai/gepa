from turbo_gepa.interfaces import Candidate, EvalResult


def test_candidate_with_meta():
    candidate = Candidate(text="hello world")
    enriched = candidate.with_meta(source="seed")
    assert enriched.text == candidate.text
    assert enriched.meta["source"] == "seed"


def test_eval_result_objective_default():
    result = EvalResult(objectives={"quality": 0.9}, traces=[], n_examples=1)
    assert result.objective("quality") == 0.9


def test_candidate_fingerprint_stable():
    candidate = Candidate(text="prompt")
    assert candidate.fingerprint == Candidate(text="prompt").fingerprint
