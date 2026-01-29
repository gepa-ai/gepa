#!/usr/bin/env python3
"""
Test script to verify KernelBench setup.

Run: uv run python -m examples.kernelbench.tests.test_setup

Tests (in order):
1. Module imports
2. Scoring logic (compute_score)
3. Error formatting (format_error)
4. Code extraction (extract_code)
5. Dataset loading (requires cloned KernelBench data)
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("\n[1/5] Testing module imports...")

    try:
        from examples.kernelbench.eval import (
            compute_score,
            execute_kernel,
            execute_baseline,
            extract_code,
            format_error,
            load_dataset,
            load_or_measure_baselines,
            acquire_gpu,
            get_free_gpus,
        )
        print("  OK eval module imported")
    except ImportError as e:
        print(f"  FAIL Failed to import eval: {e}")
        return False

    try:
        from examples.kernelbench.prompts import (
            BACKGROUND,
            KERNEL_GEN_PROMPT,
            REFINER_PROMPT,
            KernelGenSig,
            RefinerSig,
        )
        print("  OK prompts module imported")
    except ImportError as e:
        print(f"  FAIL Failed to import prompts: {e}")
        return False

    try:
        import dspy
        print("  OK dspy imported")
    except ImportError:
        print("  FAIL dspy not installed. Run: uv pip install dspy")
        return False

    return True


def test_scoring():
    """Test compute_score with mock results."""
    print("\n[2/5] Testing scoring logic...")

    from examples.kernelbench.eval import compute_score

    # Test 1: Complete failure (compilation failed)
    result_fail = {"CompilationSucceeded": False}
    score = compute_score(result_fail, baseline_time=1.0)
    print(f"  Compilation failure score: {score:.4f} (expected ~-0.07)")
    assert score < 0, "Failure should have negative score"

    # Test 2: All stages passed, exactly at baseline
    result_baseline = {
        "CompilationSucceeded": True,
        "ModelInitializeSucceeded": True,
        "NoRuntimeErrorDuringCorrectnessCheck": True,
        "NoOutputShapeMismatch": True,
        "CorrectnessSucceeded": True,
        "NoRuntimeErrorDuringPerformanceCheck": True,
        "PerformanceStatsMean": 1.0,  # same as baseline
    }
    score = compute_score(result_baseline, baseline_time=1.0)
    print(f"  At baseline (1.0ms) score: {score:.4f} (expected ~0.5)")
    assert 0.4 < score < 0.6, "Baseline performance should score ~0.5"

    # Test 3: Faster than baseline (2x speedup)
    result_fast = {
        "CompilationSucceeded": True,
        "ModelInitializeSucceeded": True,
        "NoRuntimeErrorDuringCorrectnessCheck": True,
        "NoOutputShapeMismatch": True,
        "CorrectnessSucceeded": True,
        "NoRuntimeErrorDuringPerformanceCheck": True,
        "PerformanceStatsMean": 0.5,  # 2x faster
    }
    score = compute_score(result_fast, baseline_time=1.0)
    print(f"  2x speedup (0.5ms) score: {score:.4f} (expected >0.6)")
    assert score > 0.6, "2x speedup should score >0.6"

    # Test 4: Slower than baseline
    result_slow = {
        "CompilationSucceeded": True,
        "ModelInitializeSucceeded": True,
        "NoRuntimeErrorDuringCorrectnessCheck": True,
        "NoOutputShapeMismatch": True,
        "CorrectnessSucceeded": True,
        "NoRuntimeErrorDuringPerformanceCheck": True,
        "PerformanceStatsMean": 2.0,  # 2x slower
    }
    score = compute_score(result_slow, baseline_time=1.0)
    print(f"  2x slower (2.0ms) score: {score:.4f} (expected ~0.3)")
    assert 0.2 < score < 0.4, "2x slower should score ~0.3"

    print("  OK All scoring tests passed")
    return True


def test_error_formatting():
    """Test format_error with different error types."""
    print("\n[3/5] Testing error formatting...")

    from examples.kernelbench.eval import format_error

    test_cases = [
        ({"ErrorType": "CompilationFailure", "ErrorDetail": "undefined symbol"}, "Compilation"),
        ({"ErrorType": "OutputShapeMismatch", "ErrorDetail": "(2, 3) vs (3, 2)"}, "shape"),
        ({"ErrorType": "OutputMismatch", "ErrorDetail": "max diff: 0.1"}, "match"),
        ({"ErrorType": "RuntimeFailure", "ErrorDetail": "CUDA OOM"}, "Runtime"),
    ]

    for result, expected_substring in test_cases:
        msg = format_error(result)
        assert expected_substring.lower() in msg.lower(), f"Expected '{expected_substring}' in: {msg}"
        print(f"  OK {result['ErrorType']}: {msg[:50]}...")

    print("  OK All error formatting tests passed")
    return True


def test_code_extraction():
    """Test extract_code with various formats."""
    print("\n[4/5] Testing code extraction...")

    from examples.kernelbench.eval import extract_code

    # Test 1: Python code block
    text1 = """Here's the code:
```python
def hello():
    print("world")
```
"""
    code = extract_code(text1)
    assert code and "def hello" in code, "Should extract Python code"
    print("  OK Python code block extraction")

    # Test 2: Generic code block
    text2 = """
```
int main() { return 0; }
```
"""
    code = extract_code(text2)
    assert code and "int main" in code, "Should extract generic code"
    print("  OK Generic code block extraction")

    # Test 3: No code block
    text3 = "Just plain text without code blocks"
    code = extract_code(text3)
    assert code is None, "Should return None for no code block"
    print("  OK No code block returns None")

    print("  OK All code extraction tests passed")
    return True


def test_dataset_loading():
    """Test dataset loading (requires KernelBench data)."""
    print("\n[5/5] Testing dataset loading...")

    from examples.kernelbench.eval import KERNELBENCH_ROOT, BASELINE_PATH, load_dataset

    print(f"  KERNELBENCH_ROOT: {KERNELBENCH_ROOT}")
    print(f"  BASELINE_PATH: {BASELINE_PATH}")

    # Check if data directory exists
    if not KERNELBENCH_ROOT.exists():
        print(f"  SKIP KernelBench data not found at: {KERNELBENCH_ROOT}")
        print("  To fix, run:")
        print(f"    cd {KERNELBENCH_ROOT.parent}")
        print("    git clone https://github.com/ScalingIntelligence/KernelBench.git KernelBench")
        return None  # Skip, not failure

    # Check baseline times
    if not BASELINE_PATH.exists():
        print(f"  SKIP Baseline times not found at: {BASELINE_PATH}")
        print("  You may need to run KernelBench baseline benchmarks first.")
        return None

    # Try loading dataset
    try:
        dataset = load_dataset(levels=["level1"])
        print(f"  OK Loaded {len(dataset)} problems from level1")

        if len(dataset) > 0:
            ex = dataset[0]
            print(f"    First problem: {ex.problem_id}")
            print(f"    Baseline time: {ex.baseline_time:.3f}ms")
            print(f"    Source length: {len(ex.ref_arch)} chars")

        return True
    except Exception as e:
        print(f"  FAIL Failed to load dataset: {e}")
        return False


def main():
    print("=" * 60)
    print("KernelBench Setup Test")
    print("=" * 60)

    results = {}

    # Run tests in order
    results["imports"] = test_imports()
    if not results["imports"]:
        print("\nFAIL Import test failed. Fix imports before continuing.")
        return 1

    results["scoring"] = test_scoring()
    results["error_format"] = test_error_formatting()
    results["code_extract"] = test_code_extraction()
    results["dataset"] = test_dataset_loading()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, result in results.items():
        if result is True:
            status = "PASS"
        elif result is False:
            status = "FAIL"
        else:
            status = "SKIP"
        print(f"  {name}: {status}")

    # Return code
    failures = [k for k, v in results.items() if v is False]
    if failures:
        print(f"\nFAIL {len(failures)} test(s) failed: {failures}")
        return 1
    else:
        print("\nOK All tests passed (some may be skipped)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
