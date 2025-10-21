from ufast_gepa.archive import Archive, dominates
from ufast_gepa.interfaces import Candidate, EvalResult


def test_archive_inserts_pareto_and_qd():
    archive = Archive(bins_length=4, bins_bullets=4, flags=("step", "format"))
    candidate_a = Candidate(text="Prompt with Example and step-by-step guidance.")
    candidate_b = Candidate(text="Compact prompt")
    result_a = EvalResult(objectives={"quality": 0.8, "neg_cost": -100}, traces=[], n_examples=5)
    result_b = EvalResult(objectives={"quality": 0.6, "neg_cost": -40}, traces=[], n_examples=5)
    archive.insert(candidate_a, result_a)
    archive.insert(candidate_b, result_b)
    assert candidate_a in archive.pareto_candidates()
    assert archive.sample_qd(2)


def test_dominance_checks():
    lhs = EvalResult(objectives={"quality": 0.8, "neg_cost": -20}, traces=[], n_examples=1)
    rhs = EvalResult(objectives={"quality": 0.7, "neg_cost": -25}, traces=[], n_examples=1)
    assert dominates(lhs, rhs)
