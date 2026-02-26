"""
Nsight Profiling Module for KernelBench
========================================

This module provides GPU profiling capabilities using NVIDIA Nsight Compute (ncu).
It allows collecting hardware-level metrics from kernels.

NOTE: this is an experimental module, not part of the default eval pass. 
You need hardware counter access (usually requires sudo) to accesss hardware counter.
We only support local mode with this feature, not avaliable on Modal.

Key Features:
- Profile arbitrary PyTorch functions with hardware metrics
- Profile KernelBench models (ModelNew) with automatic setup/teardown
- Combine metrics from multi-kernel operations (common in PyTorch)

Requirements:
- NVIDIA Nsight Compute CLI (ncu) must be installed and in PATH
- nsight-python package

Common Metrics:
- gpu__time_duration.sum: Total GPU time in nanoseconds
- sm__cycles_elapsed.sum: Total SM cycles elapsed
- sm__cycles_active.avg: Average active cycles per SM

Reference: https://docs.nvidia.com/nsight-python
"""

import os
from shutil import which

import torch

# =============================================================================
# Nsight Availability Check
# =============================================================================

try:
    import nsight
    NSIGHT_AVAILABLE = True
except ImportError:
    NSIGHT_AVAILABLE = False


# =============================================================================
# Utility Functions
# =============================================================================

def check_ncu_available() -> bool:
    """
    Check if NVIDIA Nsight Compute CLI (ncu) is available in PATH.
    
    The ncu command-line tool is required for collecting GPU hardware metrics.
    It's typically installed with the CUDA Toolkit or NVIDIA Nsight Compute.
    
    Returns:
        True if ncu is found in PATH, False otherwise.
    """
    return which('ncu') is not None


# =============================================================================
# Core Profiling Functions
# =============================================================================

def profile_with_nsight(func, metrics=None, num_trials=1):
    """
    Profile a PyTorch function and collect hardware metrics.
    
    Handles complexity:
    - Setting up the Nsight kernel analyzer
    - Combining metrics when PyTorch ops launch multiple CUDA kernels
    - Extracting results from Nsight's DataFrame format
    
    Args:
        func: A callable (no arguments) that executes the code to profile.
              Typically a closure that captures the model and inputs.
        metrics: List of Nsight metric names to collect. If None, defaults to
                 ['sm__cycles_active.avg']. Can also pass a single string.
        num_trials: Number of times to run the function for averaging.
    
    Returns:
        Dictionary mapping metric names to their values (float).
        Returns None for metrics that couldn't be collected.
    
    Example:
        >>> def my_kernel():
        ...     return torch.matmul(a, b)
        >>> results = profile_with_nsight(my_kernel, ['gpu__time_duration.sum'])
        >>> print(results['gpu__time_duration.sum'])  # Time in nanoseconds
    
    Raises:
        RuntimeError: If nsight-python is not installed.
    """
    if not NSIGHT_AVAILABLE:
        raise RuntimeError(
            "nsight-python not available."
        )
    
    # Normalize metrics to a list
    if metrics is None:
        metrics = ['sm__cycles_active.avg']
    elif isinstance(metrics, str):
        metrics = [metrics]
    
    # Define the profiled function with Nsight decorator
    # NOTE: PyTorch operations often launch multiple CUDA kernels (e.g., a matmul
    # might have separate kernels for the computation and memory operations).
    # We use combine_kernel_metrics to sum these together for a single measurement.
    @nsight.analyze.kernel(
        metrics=metrics,
        runs=num_trials,
        configs=[(0,)],  # Use default GPU config
        combine_kernel_metrics=lambda a, b: (0 if a is None else a) + (0 if b is None else b),
    )
    def profiled(_):
        # The nsight.annotate context marks the region we care about
        with nsight.annotate("kernel"):
            return func()
    
    try:
        # Run profiling - this invokes ncu under the hood
        result = profiled()
        
        # Convert results to DataFrame
        df = result.to_dataframe() if result else None
        if df is None or df.empty:
            return {m: None for m in metrics}
        
        # Nsight returns a DataFrame with columns like:
        # - 'Metric': The metric name (e.g., 'gpu__time_duration.sum')
        # - 'AvgValue': The measured value
        # We need to find these columns (names may vary slightly)
        metric_col = next((c for c in df.columns if c.lower() == 'metric'), None)
        value_col = next((c for c in df.columns if 'value' in c.lower()), None)
        
        if not metric_col or not value_col:
            return {m: None for m in metrics}
        
        # Build a dictionary of all metrics in the DataFrame
        metric_dict = {
            row[metric_col]: float(row[value_col]) 
            for _, row in df.iterrows()
        }
        
        # Return only the requested metrics (None if not found)
        return {m: metric_dict.get(m) for m in metrics}
        
    except Exception as e:
        print(f"Error profiling: {e}")
        return {m: None for m in metrics}


def profile_kernelbench_model_with_nsight(
    custom_model_src: str,
    ref_model_src: str = None,
    metrics: list = None,
    num_trials: int = 1,
    seed: int = 42,
    device: torch.device = None,
    backend: str = "cuda",
    precision: torch.dtype = torch.float32,
    build_dir: str = None,
    verbose: bool = False,
) -> dict:
    """
    Profile a KernelBench model (ModelNew) using Nsight hardware metrics.
    
    This is the high-level profiling function designed for KernelBench workflows.
    It handles the full lifecycle:
    1. Load and compile the custom model from source code
    2. Generate inputs using the model's get_inputs() function
    3. Profile the forward pass with Nsight
    4. Clean up resources
    
    IMPORTANT: This function assumes the model has already been validated for
    correctness via eval. No correctness checking is performed here.
    
    Args:
        custom_model_src: Python source code string containing the ModelNew class.
        ref_model_src: Optional source code for the reference model. Used to get
                       get_inputs() and get_init_inputs() if they're not in
                       custom_model_src. If None, uses custom_model_src.
        metrics: List of Nsight metrics to collect. Defaults to ['sm__cycles_active.avg'].
        num_trials: Number of profiling runs for averaging. Default: 1.
        seed: Random seed for reproducible input generation. Default: 42.
        device: CUDA device to run on. Default: cuda:0.
        backend: Compilation backend ('cuda', 'triton', 'tilelang', 'cute').
        precision: torch.dtype for computation. Default: torch.float32.
        build_dir: Directory for compiled kernel artifacts. Default: None.
        verbose: Print progress messages. Default: False.
    
    Returns:
        Dictionary mapping metric names to their measured values.
        Values are None if the metric couldn't be collected.
    
    Example:
        >>> results = profile_kernelbench_model_with_nsight(
        ...     custom_model_src=my_model_code,
        ...     ref_model_src=ref_model_code,
        ...     metrics=['gpu__time_duration.sum', 'sm__cycles_elapsed.sum'],
        ...     verbose=True
        ... )
        >>> print(f"GPU time: {results['gpu__time_duration.sum']} ns")
    """
    # Import eval utilities (deferred to avoid circular imports)
    from kernelbench.eval import (
        load_custom_model,
        load_custom_model_with_tempfile,
        load_original_model_and_inputs,
        _process_input_tensor,
        set_seed,
        graceful_eval_cleanup,
    )
    
    # Set defaults
    device = device or torch.device("cuda:0")
    if metrics is None:
        metrics = ['sm__cycles_active.avg']
    elif isinstance(metrics, str):
        metrics = [metrics]
    
    torch.cuda.set_device(device)
    
    # -------------------------------------------------------------------------
    # Step 1: Load input generation functions from model source
    # -------------------------------------------------------------------------
    # The model source should define get_inputs() and get_init_inputs() functions
    # that return the tensors needed to run the model.
    input_source = ref_model_src or custom_model_src
    context = {}
    _, get_init_inputs, get_inputs = load_original_model_and_inputs(input_source, context)
    
    # Generate initialization inputs (for model constructor)
    set_seed(seed)
    init_inputs = [
        _process_input_tensor(x, device, backend, precision) 
        for x in get_init_inputs()
    ]
    
    # -------------------------------------------------------------------------
    # Step 2: Load and compile the custom model
    # -------------------------------------------------------------------------
    if verbose:
        print("[Profile] Loading and compiling custom model...")
    
    # Enable CUDA Device-Side Assertions for better error messages
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    tempfile = None
    
    # Different backends require different loading mechanisms
    if backend.lower() in ["triton", "tilelang", "cute"]:
        # These backends need a temp file for proper module loading
        ModelNew, tempfile = load_custom_model_with_tempfile(
            custom_model_src, entry_point="ModelNew"
        )
    else:
        # Standard CUDA backend
        ModelNew = load_custom_model(custom_model_src, {}, build_dir)
    
    torch.cuda.synchronize(device=device)
    
    # -------------------------------------------------------------------------
    # Step 3: Instantiate the model
    # -------------------------------------------------------------------------
    with torch.no_grad():
        set_seed(seed)
        custom_model = ModelNew(*init_inputs)
        custom_model = custom_model.to(device=device, dtype=precision)
        torch.cuda.synchronize(device=device)
    
    if verbose:
        print("[Profile] Model instantiated successfully")
    
    # -------------------------------------------------------------------------
    # Step 4: Profile the forward pass
    # -------------------------------------------------------------------------
    # Generate forward pass inputs
    set_seed(seed)
    inputs = [
        _process_input_tensor(x, device, backend, precision) 
        for x in get_inputs()
    ]
    
    if verbose:
        print(f"[Profile] Profiling with nsight (metrics: {metrics})...")
    
    # Create a closure for the forward pass
    def model_forward():
        with torch.no_grad():
            return custom_model(*inputs)
    
    # Run profiling
    metric_values = profile_with_nsight(
        model_forward, 
        metrics=metrics, 
        num_trials=num_trials
    )
    
    if verbose:
        print("[Profile] Profiling completed successfully")
    
    # -------------------------------------------------------------------------
    # Step 5: Cleanup
    # -------------------------------------------------------------------------
    graceful_eval_cleanup(context, device, tempfile)
    
    return metric_values


# =============================================================================
# Examples and Tests
# =============================================================================

def example_ncu_python_profile():
    """
    Simple example demonstrating how to profile a basic matrix multiplication.
    
    This shows the minimal setup needed to use profile_with_nsight().
    """
    print("Creating test tensors...")
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    
    # Create a closure that captures the tensors
    def matmul_kernel():
        return a @ b
    
    print("Running nsight profiling...")
    
    metric_values = profile_with_nsight(
        matmul_kernel,
        metrics=[
            'sm__cycles_active.avg',           # Average active cycles per SM
            'sm__cycles_elapsed.sum',          # Total cycles elapsed
            'smsp__inst_executed_pipe_tensor_op_hmma.sum',  # Tensor core ops
        ],
        num_trials=1,
    )
    
    print("\nProfiling results:")
    for metric_name, value in metric_values.items():
        print(f"  {metric_name}: {value}")


def test_flash_attention_profile():
    """
    Test profiling a Flash Attention model from the KernelBench examples.
    
    This demonstrates the full workflow of profiling a KernelBench model
    using profile_kernelbench_model_with_nsight().
    """
    from kernelbench.utils import read_file
    
    # Locate the example model files
    REPO_ROOT = os.path.dirname(__file__)
    ref_model_path = os.path.join(
        REPO_ROOT, "prompts/few_shot/model_ex_flash_attn.py"
    )
    custom_model_path = os.path.join(
        REPO_ROOT, "prompts/few_shot/model_new_ex_flash_attn.py"
    )
    
    print("[Test] Reading model source files...")
    ref_model_src = read_file(ref_model_path)
    custom_model_src = read_file(custom_model_path)
    
    print("[Test] Starting profiling with nsight...")
    
    metrics = profile_kernelbench_model_with_nsight(
        custom_model_src=custom_model_src,
        ref_model_src=ref_model_src,
        metrics=[
            'gpu__time_duration.sum',   # Total GPU execution time (ns)
            'sm__cycles_elapsed.sum',   # Total SM cycles
        ],
        seed=42,
        backend="cuda",
        precision=torch.float32,
        verbose=True
    )
    
    print("\n[Test] Profiling results:")
    print("=" * 60)
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"  {metric_name}: {value:,.0f}")
        else:
            print(f"  {metric_name}: <not available>")
    print("=" * 60)
    
    return metrics


# Optional: Decorated benchmark function for direct use with nsight
if NSIGHT_AVAILABLE:
    @nsight.analyze.kernel
    def benchmark_matmul(n):
        """
        Standard benchmark following nsight-python documentation style.
        
        This shows how to use the @nsight.analyze.kernel decorator directly
        for simple benchmarking scenarios.
        """
        a = torch.randn(n, n, device="cuda")
        b = torch.randn(n, n, device="cuda")
        with nsight.annotate("matmul"):
            c = a @ b
        return c


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Verify prerequisites
    if not check_ncu_available():
        print("ERROR: ncu not found in PATH.")
        print("Install NVIDIA Nsight Compute from:")
        print("  https://developer.nvidia.com/nsight-compute")
        exit(1)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        exit(1)
    
    print("=" * 60)
    print("Running Nsight Profiling Examples")
    print("=" * 60)
    
    # Run the simple example first
    print("\n--- Example: Basic Matrix Multiplication ---\n")
    example_ncu_python_profile()
    
    # Run the full KernelBench model test
    print("\n--- Test: Flash Attention Model ---\n")
    test_flash_attention_profile()
