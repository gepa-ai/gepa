#!/usr/bin/env python3
"""Test script for debugging the cloudcast evaluator."""

import sys
import os
import tempfile
import traceback

# Add repo root to path
from pathlib import Path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from examples.adrs.cloudcast.evaluator import (
    create_fitness_function,
    load_config_dataset,
    evaluate_stage1,
    run_single_config,
    FAILED_SCORE,
)
from examples.adrs.cloudcast.main import INITIAL_PROGRAM


def test_dependencies():
    """Test that all required dependencies are installed."""
    print("=" * 60)
    print("Testing Dependencies")
    print("=" * 60)
    
    deps = ['networkx', 'pandas', 'numpy']
    all_ok = True
    
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {dep} (version: {version})")
        except ImportError as e:
            print(f"  ✗ {dep} MISSING: {e}")
            all_ok = False
    
    return all_ok


def test_config_loading():
    """Test that configuration files load correctly."""
    print("\n" + "=" * 60)
    print("Testing Config Loading")
    print("=" * 60)
    
    try:
        samples = load_config_dataset()
        print(f"  Loaded {len(samples)} configuration samples:")
        for s in samples:
            config_name = os.path.basename(s['config_file'])
            exists = os.path.exists(s['config_file'])
            status = "✓" if exists else "✗ MISSING"
            print(f"    {status} {config_name}")
        return len(samples) > 0
    except Exception as e:
        print(f"  ✗ Failed to load configs: {e}")
        traceback.print_exc()
        return False


def test_syntax_check():
    """Test the syntax checking stage."""
    print("\n" + "=" * 60)
    print("Testing Syntax Check (Stage 1)")
    print("=" * 60)
    
    # Write initial program to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(INITIAL_PROGRAM)
        temp_path = f.name
    
    try:
        result = evaluate_stage1(temp_path)
        if result.get('runs_successfully', 0.0) >= 1.0:
            print(f"  ✓ Syntax check passed")
            return True
        else:
            print(f"  ✗ Syntax check failed: {result.get('error', 'Unknown error')}")
            return False
    finally:
        os.unlink(temp_path)


def test_single_config_execution():
    """Test execution on a single configuration."""
    print("\n" + "=" * 60)
    print("Testing Single Config Execution")
    print("=" * 60)
    
    samples = load_config_dataset()
    if not samples:
        print("  ✗ No samples to test")
        return False
    
    # Write initial program to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(INITIAL_PROGRAM)
        temp_path = f.name
    
    try:
        sample = samples[0]
        config_name = os.path.basename(sample['config_file'])
        print(f"  Testing on: {config_name}")
        
        success, cost, transfer_time, error_msg, detailed_info = run_single_config(
            temp_path, 
            sample['config_file'],
            sample.get('num_vms', 2)
        )
        
        if success:
            print(f"  ✓ Execution succeeded")
            print(f"    Cost: ${cost:.4f}")
            print(f"    Transfer time: {transfer_time:.2f}s")
            return True
        else:
            print(f"  ✗ Execution failed: {error_msg}")
            if 'traceback' in detailed_info:
                print(f"    Traceback:\n{detailed_info['traceback']}")
            return False
    finally:
        os.unlink(temp_path)


def test_fitness_function():
    """Test the full fitness function."""
    print("\n" + "=" * 60)
    print("Testing Fitness Function")
    print("=" * 60)
    
    samples = load_config_dataset()
    if not samples:
        print("  ✗ No samples to test")
        return False
    
    fitness_fn = create_fitness_function()
    candidate = {'program': INITIAL_PROGRAM}
    
    all_ok = True
    for sample in samples:
        config_name = os.path.basename(sample['config_file'])
        
        try:
            score, output, side_info = fitness_fn(candidate, sample)
            
            if score == FAILED_SCORE:
                print(f"  ✗ {config_name}: FAILED (score={score})")
                if 'Error' in side_info:
                    print(f"      Error: {side_info['Error']}")
                if 'error' in output:
                    print(f"      Output error: {output['error']}")
                all_ok = False
            else:
                print(f"  ✓ {config_name}: score={score:.6f}, cost=${output.get('cost', 'N/A')}")
        except Exception as e:
            print(f"  ✗ {config_name}: EXCEPTION - {e}")
            traceback.print_exc()
            all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("CLOUDCAST EVALUATOR TEST SUITE")
    print("=" * 60)
    
    results = []
    
    results.append(("Dependencies", test_dependencies()))
    results.append(("Config Loading", test_config_loading()))
    results.append(("Syntax Check", test_syntax_check()))
    results.append(("Single Config", test_single_config_execution()))
    results.append(("Fitness Function", test_fitness_function()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed! The evaluator should be working correctly.")
    else:
        print("Some tests failed. Check the output above for details.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
