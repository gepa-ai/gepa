import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from kernelbench import timing
from kernelbench.profile import NSIGHT_AVAILABLE, check_ncu_available

"""
Test Timing
We want to systematically study different timing methodologies.
"""
REPO_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# use exampls in the few shot directory
EXAMPLES_PATH = os.path.join(REPO_PATH, "src", "kernelbench", "prompts", "few_shot")

# Configure your test cases here
TEST_REF_FILE = "model_ex_tiled_matmul.py"
TEST_KERNEL_FILE = "model_new_ex_tiled_matmul.py"

assert os.path.exists(os.path.join(EXAMPLES_PATH, TEST_REF_FILE)), f"Reference file {TEST_REF_FILE} does not exist in {EXAMPLES_PATH}"
assert os.path.exists(os.path.join(EXAMPLES_PATH, TEST_KERNEL_FILE)), f"Kernel file {TEST_KERNEL_FILE} does not exist in {EXAMPLES_PATH}"


def _run_timing_smoke_test_matmul(timing_func_name:str, device:str="cuda"):
    """
    Scaffold function for timing smoke tests.
    Smoke test for using 2048x2048x2048 matmul with 5 warmup and 100 trials.

    Args:
        timing_fn: The timing function to test
        use_args: Whether the timing function expects args parameter (True for cuda_event/time_dot_time, False for do_bench)
    """
    # Skip if CUDA is not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping timing tests")
    
    # Create simple test matrices
    M = 2048
    N = 2048
    K = 2048
    a = torch.randn(M, K, device=device)
    b = torch.randn(K, N, device=device)
    
    num_warmup = 5
    num_trials = 100
    
    # Define the kernel function to time
    def matmul_kernel(a, b):
        return torch.matmul(a, b)
    
    timing_func = timing.get_timing_function(timing_func_name)
    elapsed_times = timing_func(
        matmul_kernel,
        args=[a, b],
        num_warmup=num_warmup,
        num_trials=num_trials,
        verbose=False,
        device=device    
    )
    
    # Validate results
    assert isinstance(elapsed_times, list), "Expected list of elapsed times"
    
    # disabled this check as do_bench does not use num_trials
    # assert len(elapsed_times) == num_trials, f"Expected {num_trials} timing results, got {len(elapsed_times)}"
    assert all(isinstance(t, float) for t in elapsed_times), "All timing results should be floats"
    assert all(t > 0 for t in elapsed_times), "All timing results should be positive"
    # DEBUG print times
    # print(f"smoke test matmul elapsed times with {timing_func_name} (in ms): {elapsed_times}")

    stats = timing.get_timing_stats(elapsed_times, device=device)
    print("Timing stats")
    print(stats)
    

# test all currently available timing methods
def run_all_timing_tests(device="cuda"):
    timing_methods = ["cuda_event", "host_time", "do_bench", "do_bench_impl"]
    for timing_method in timing_methods:
        _run_timing_smoke_test_matmul(timing_method, device=device)


def run_nsight_timing_test(device="cuda"):
    """
    Run nsight-python timing test if available.
    
    Nsight requires:
    - nsight-python package installed
    - ncu (Nsight Compute CLI) in PATH
    
    Compares nsight GPU time against cuda_event timing for the same matmul operation.
    """
    if not NSIGHT_AVAILABLE:
        print("[SKIP] nsight-python not installed")
        return None
    
    if not check_ncu_available():
        print("[SKIP] ncu not found in PATH")
        return None
    
    print("\n" + "=" * 60)
    print("Running nsight-python timing benchmark")
    print("=" * 60)
    
    # Run nsight timing
    print("\n--- nsight_python_time ---")
    _run_timing_smoke_test_matmul("nsight_python_time", device=device)
    
    # Run cuda_event for comparison
    print("\n--- cuda_event (for comparison) ---")
    _run_timing_smoke_test_matmul("cuda_event", device=device)
    
    print("\n" + "=" * 60)
    print("nsight-python timing benchmark complete")
    print("=" * 60)


def run_all_timing_tests_with_nsight(device="cuda"):
    """
    Run all timing methods including nsight-python if available.
    """
    # Standard timing methods (always available)
    run_all_timing_tests(device=device)
    
    # Nsight timing (requires ncu + nsight-python)
    run_nsight_timing_test(device=device)


# select a free GPU here or set CUDA_VISIBLE_DEVICES
test_device = torch.device("cuda:5")
run_all_timing_tests(test_device)
run_nsight_timing_test(test_device)



def test_do_bench_simple_smoke():
    """
    Smoke test for do_bench itself on a simple CUDA operation.
    Just checks it runs and returns timings.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping do_bench smoke test")

    from do_bench import do_bench

    x = torch.randn(1024, device="cuda")

    def fn():
        # simple GPU op; do_bench will sync/timestamp internally
        return (x * 2).sum()

    rep = 5
    times = do_bench(fn, warmup=2, rep=rep, return_mode="all")
    assert isinstance(times, list)
    assert len(times) == rep

