"""Quick tests for agent setup."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


def test_imports():
    from examples.arc_agi.tracked_llm import TrackedLLM
    from examples.arc_agi.main import SEED_AGENT_CODE, load_arc_dataset
    from examples.arc_agi.evaluate import compare_grid, evaluate_predictions, evaluate_test
    print("Imports OK")


def test_seed_code_syntax():
    from examples.arc_agi.main import SEED_AGENT_CODE
    compile(SEED_AGENT_CODE, "<seed>", "exec")
    print("Seed code syntax OK")


def test_compare_grid():
    from examples.arc_agi.evaluate import compare_grid

    # Correct
    correct, fb = compare_grid([[1, 2], [3, 4]], [[1, 2], [3, 4]])
    assert correct, f"Should be correct: {fb}"

    # Wrong values
    correct, fb = compare_grid([[1, 2], [3, 5]], [[1, 2], [3, 4]])
    assert not correct, "Should be wrong"
    assert "(1, 1)" in fb, f"Should mention wrong position: {fb}"

    # Wrong shape
    correct, fb = compare_grid([[1, 2]], [[1, 2], [3, 4]])
    assert not correct, "Should be wrong shape"

    print("compare_grid OK")


def test_evaluate_predictions():
    from examples.arc_agi.evaluate import evaluate_predictions

    preds = [[[1, 2]], [[3, 4]]]
    golds = [[[1, 2]], [[3, 4]]]

    score, results = evaluate_predictions(preds, golds)
    assert score == 1.0, f"Expected 1.0, got {score}"
    print("evaluate_predictions OK")


def test_evaluate_test():
    from examples.arc_agi.evaluate import evaluate_test

    # First attempt wrong, second correct
    test_preds = [
        [[[0, 0]], [[1, 2]]],  # 2 attempts for first test
    ]
    test_out = [[[1, 2]]]

    score, results = evaluate_test(test_preds, test_out)
    assert score == 1.0, f"Expected 1.0 (2nd attempt should pass), got {score}"
    assert results[0]["correct"], "Should pass with 2nd attempt"
    print("evaluate_test OK")


def test_dataset():
    from examples.arc_agi.main import load_arc_dataset

    train_set, val_set, test_set = load_arc_dataset(seed=0)

    assert len(train_set) > 0, "train_set should not be empty"
    assert len(val_set) == 200, f"val_set should have 200, got {len(val_set)}"
    assert len(test_set) > 0, "test_set should not be empty"

    ex = train_set[0]
    assert hasattr(ex, "test_out"), "Missing test_out"

    print(f"Dataset OK: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")


if __name__ == "__main__":
    test_imports()
    test_seed_code_syntax()
    test_compare_grid()
    test_evaluate_predictions()
    test_evaluate_test()
    test_dataset()
    print("\nAll tests passed!")
