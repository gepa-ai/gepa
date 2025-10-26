#!/usr/bin/env python3
"""
Quick test of circle packing implementation with small workloads
"""

import importlib.util
import os
import sys

# Add parent directory to path to import gepa
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Import the circle packing module directly
spec = importlib.util.spec_from_file_location(
    "circle_packing_multi_n", os.path.join(os.path.dirname(__file__), "06_circle_packing_multi_n.py")
)
circle_packing_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(circle_packing_module)

evaluate_circle_packing = circle_packing_module.evaluate_circle_packing
INITIAL_PACKING_CODE = circle_packing_module.INITIAL_PACKING_CODE
validate_packing = circle_packing_module.validate_packing

import numpy as np


def test_initial_code():
    """Test that the initial code works."""
    print("Testing initial packing code...")

    seed_candidate = {"packing_function": INITIAL_PACKING_CODE}

    # Test on a few small n values
    test_ns = [7, 10]

    results = evaluate_circle_packing(seed_candidate, test_ns)

    for i, n in enumerate(test_ns):
        res = results[i]
        print(f"\nn={n}:")
        print(f"  Score: {res['score']:.4f}")
        print(f"  Valid: {res['context_and_feedback'].get('valid', False)}")
        if res["context_and_feedback"].get("valid", False):
            print(f"  Sum radii: {res['context_and_feedback']['sum_radii']:.6f}")
            print(f"  Target: {res['context_and_feedback']['target']:.6f}")
        else:
            print(f"  Error: {res['context_and_feedback'].get('feedback', 'Unknown error')[:200]}")


def test_validation():
    """Test the validation function."""
    print("\n\nTesting validation function...")

    # Valid packing
    n = 4
    centers = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    radii = np.array([0.2, 0.2, 0.2, 0.2])

    is_valid, details = validate_packing(n, centers, radii)
    print(f"\nValid packing test: {is_valid}")
    print(f"  Details: {details}")

    # Invalid - overlapping
    centers_invalid = np.array([[0.5, 0.5], [0.5, 0.5], [0.25, 0.75], [0.75, 0.75]])
    radii_invalid = np.array([0.2, 0.2, 0.2, 0.2])

    is_valid_inv, details_inv = validate_packing(n, centers_invalid, radii_invalid)
    print(f"\nOverlapping circles test: {is_valid_inv} (should be False)")
    print(f"  Overlaps detected: {len(details_inv['overlaps'])}")

    # Invalid - outside boundary
    centers_out = np.array([[0.1, 0.1], [0.9, 0.9], [0.25, 0.75], [0.75, 0.75]])
    radii_out = np.array([0.15, 0.15, 0.2, 0.2])

    is_valid_out, details_out = validate_packing(n, centers_out, radii_out)
    print(f"\nBoundary violation test: {is_valid_out} (should be False)")
    print(f"  Violations detected: {len(details_out['boundary_violations'])}")


def test_empty_code():
    """Test error handling for empty code."""
    print("\n\nTesting error handling...")

    # Empty code
    empty_candidate = {"packing_function": ""}

    results = evaluate_circle_packing(empty_candidate, [7])
    print("\nEmpty code test:")
    print(f"  Score: {results[0]['score']} (should be 0.0)")
    print(f"  Error: {results[0]['context_and_feedback'].get('error', 'No error')}")

    # Invalid code (missing function)
    invalid_candidate = {"packing_function": "import numpy as np\nprint('hello')"}

    results = evaluate_circle_packing(invalid_candidate, [7])
    print("\nMissing function test:")
    print(f"  Score: {results[0]['score']} (should be 0.0)")
    print(f"  Error: {results[0]['context_and_feedback'].get('error', 'No error')}")


def main():
    """Run all tests."""
    print("=" * 80)
    print("Circle Packing Implementation Tests")
    print("=" * 80)

    test_validation()
    test_initial_code()
    test_empty_code()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
