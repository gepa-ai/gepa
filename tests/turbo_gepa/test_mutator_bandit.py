import pytest

from turbo_gepa.mutator import Mutator, MutationConfig


def make_mutator() -> Mutator:
    config = MutationConfig(reflection_batch_size=2, max_mutations=16, max_tokens=2048, objective_key="quality")
    return Mutator(config=config, batch_reflection_runner=None, spec_induction_runner=None, temperature_mutations_enabled=False)


def test_bandit_alloc_even_split_with_no_signal() -> None:
    mutator = make_mutator()
    budget = mutator._allocate_operator_budget(10, ["incremental_reflection", "spec_induction"])
    assert budget["incremental_reflection"] + budget["spec_induction"] == 10
    # With no reward data, slots split roughly evenly (difference at most 1)
    assert abs(budget["incremental_reflection"] - budget["spec_induction"]) <= 1


def test_bandit_alloc_favors_high_reward_operator() -> None:
    mutator = make_mutator()

    for _ in range(8):
        mutator.report_outcome("incremental_reflection", 0.15)
    for _ in range(8):
        mutator.report_outcome("spec_induction", -0.05)

    budget = mutator._allocate_operator_budget(12, ["incremental_reflection", "spec_induction"])
    assert budget["incremental_reflection"] > budget["spec_induction"]
    assert budget["incremental_reflection"] >= 8
    assert budget["spec_induction"] <= 4


def test_bandit_alloc_handles_small_budget() -> None:
    mutator = make_mutator()
    for _ in range(3):
        mutator.report_outcome("spec_induction", 0.2)
    budget = mutator._allocate_operator_budget(1, ["incremental_reflection", "spec_induction"])
    assert budget["spec_induction"] == 1
    assert budget["incremental_reflection"] == 0
