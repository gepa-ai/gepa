import torch
import json
import numpy as np
import time
from typing import Any, Optional, Union
import os


def measure_ref_program_time(
    ref_arch_name: str,
    ref_arch_src: str, # PyTorch program code string
    num_warmup: int = 5,
    num_trials: int = 100,
    discard_first: int = 1,
    timing_method: str = "cuda_event",
    # Torch eager or torch.compile configuration
    use_torch_compile: bool = False,
    torch_compile_backend: str = "inductor",
    torch_compile_options: str = "default",
    device: torch.device = torch.device("cuda:0"),
    verbose: bool = False,
    precision: Union[str, torch.dtype] = "fp32", # fp16, fp32, bf16 or torch.dtype
) -> dict:
    """Measure the runtime of a KernelBench *reference* program.

    This measures the execution time of the reference `Model` defined in
    `ref_arch_src` (i.e., *not* `ModelNew`). It can optionally run the reference
    model under `torch.compile`.

    NOTE: This function is for PyTorch-only reference models, so no `backend` parameter is needed.
    For pure PyTorch program, we assume it operates all on main stream (as torch operators execute on the default cuda stream).
    Standard PyTorch ops do NOT spawn extra streams.
    """
    from kernelbench.eval import load_original_model_and_inputs, set_seed

    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )

    try:
        with torch.no_grad():
            if isinstance(device, str):
                device = torch.device(device)
            elif isinstance(device, int):
                device = torch.device(f"cuda:{device}")
            torch.cuda.set_device(device)

            torch.cuda.synchronize(device=device)
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()

            from kernelbench.eval import get_torch_dtype_from_string
            if isinstance(precision, str):
                precision_dtype = get_torch_dtype_from_string(precision)
            else:
                precision_dtype = precision


            # set model weights and inputs to specified precision
            inputs = [
                x.to(device=device, dtype=precision_dtype) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.to(device=device, dtype=precision_dtype) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            model = Model(*init_inputs)
            model = model.to(device=device, dtype=precision_dtype)

            # convert all precision so torch compile can target specific dtype
            if use_torch_compile:
                torch._dynamo.reset() # reset torch dynamo cache (clear memory and reset graph)
                print(
                    f"Using torch.compile to compile model {ref_arch_name} with {torch_compile_backend} backend and {torch_compile_options} mode"
                )
                # NOTE: torch compile uses lazy compilation (triggered by first forward pass)
                # the warmup in the timing function handles that and should not affect timed trials
                model = torch.compile(
                    model,
                    backend=torch_compile_backend,
                    mode=torch_compile_options,
                )
            else:
                print(f"Using PyTorch Eager Execution on {ref_arch_name}")

            torch.cuda.synchronize(device=device)

            timing_fn = get_timing_function(timing_method)
            elapsed_times = timing_fn(
                model,
                inputs,
                num_warmup=num_warmup,
                num_trials=num_trials,
                discard_first=discard_first,
                verbose=verbose,
                device=device,
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)

            if verbose:
                print(f"{ref_arch_name} {runtime_stats}")

            return runtime_stats
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")
        return None


def measure_program_time(*args, **kwargs):
    """Alias for backwards compatibility. See `measure_ref_program_time`."""
    return measure_ref_program_time(*args, **kwargs)

################################################################################
# timing.py
# Various timing methods and utilities for performance evaluation
# please make a PR if you have suggestions!

# Try them out at src/unit_tests/test_eval_timing.py
################################################################################

def clear_l2_cache(device: torch.device | str = "cuda"):
    """
    Clear L2 Cache line by thrashing with a large tensor
    Acknowledge GPU mode reference kernel repo:
    https://github.com/gpu-mode/reference-kernels/commit/7c15075a39286e88939d99d3f3a60be88b8e6223#diff-3a30a71cbf8db2badd224f4d92f9a2546925a5b522632a31d353526b7a5f3338R158-R163
    """
    # don't reserve space for persisting lines
    # cp.cuda.runtime.cudaDeviceSetLimit(cp.cuda.runtime.cudaLimitPersistingL2CacheSize, 0)
    
    # Thrash L2 cache by creating a larger dummy tensor, effectively flushing the cache
    # 32 * 1024 * 1024 * 8B = 256MB 
    # NOTE: we can make this more adaptive based on device
    # L2 cache sizes: A100=40MB, H100=50MB, H200=90MB, RTX4090=72MB, L40S=48MB, Blackwell≈192MB → overwrite >200MB to fully thrash L2
    dummy = torch.empty((32, 1024, 1024), dtype=torch.int64, device=device)
    # write to tensor with inplace fill
    dummy.fill_(42) 
    del dummy

def clear_l2_cache_triton(cache=None, device: str = "cuda"):
    """
    Thrash the cache by making a large dummy tensor, using triton runtime's functionality
    """
    from triton import runtime as triton_runtime
    with torch.cuda.device(device):
        cache = triton_runtime.driver.active.get_empty_cache_for_benchmark()
        # this effectively thrashes L2 cache under the hood too
        triton_runtime.driver.active.clear_cache(cache)


def get_timing_function(
    method: str = "cuda_event", # by default 
) -> callable:
    """
    Get timing function by method name.

    Available methods:
        - "cuda_event": torch.cuda.event timing (default, explicit trial control)
        - "do_bench": Use triton's do_bench (adaptive trial count based on time budget)
        - "do_bench_impl": Mirrors Triton's do_bench implementation (explicit control)
        - "host_time": Host side wall-clock timing (might include overhead)
    
    Args:
        method: Name of timing method to use
    
    Returns:
        Timing function with signature (kernel_fn, args, num_warmup, num_trials, 
        discard_first, verbose, device) -> list[float]
    """
    print(
        f"[Profiling] Using timing method: {method}"
    )
    # NOTE: here are all the timing methods we supporting for now
    match method:
        case "cuda_event":
            return time_execution_with_cuda_event
        case "do_bench":
            # caveat: just using do_bench as it is 
            # do not have precise control over number of trials
            return time_execution_with_do_bench_interface
        case "do_bench_impl":
            # do_bench equivalent implementations for transparency and control
            return time_execution_with_do_bench_impl
        case "host_time":
            return time_execution_with_host_time 
        case "nsight_python_time":
            return time_execution_with_nsight_python
        # we might add other methods in the future
        case _: 
            raise ValueError(f"Unsupported timing method: {method}")

"""
Kernel Timing Functions
NOTE: we have a WIP blogpost on this topic covering the various timing approaches   
"""

def time_execution_with_cuda_event(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1, # set to 0 to disable
    verbose: bool = True,
    device: torch.device = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using torch.cuda.event
    The first version of KernelBench used this for evaluation.
    We care about cold cache performance here.

    Note: this version does not guard against adverserial cuda streams yet.
    It assumes computation is done on the current stream for current device. 
    Stay tuned for future PRs. 

    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        num_warmup: Number of warmup iterations before timing
        num_trials: Number of timing trials to run
        discard_first: Number of first trials to discard, for consistency with host_time, set to 0 to disable
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, defaults to current device

    Returns:
        List of elapsed times in milliseconds (length = num_trials)
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    with torch.cuda.device(device):
        
        # Warm ups
        for _ in range(num_warmup):
            kernel_fn(*args)
            torch.cuda.synchronize(device=device)
        
        # note this only release PyTorch’s CUDA caching allocator, not necessarily clearing device's L2 cache
        torch.cuda.empty_cache()
        
        print(f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}"
        )

        elapsed_times: list[float] = [] # in ms

        # Timing trials
        for trial in range(num_trials + discard_first):
            torch.cuda.synchronize(device=device) # block on all streams

            # create event marker default is not interprocess
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            clear_l2_cache(device=device) # measuring cold cache performance

            # note cuda events mark event on current stream
            start_event.record()
            _ = kernel_fn(*args)
            end_event.record() 

            # waits for all streams on that device
            # though it is important to note the events only record time between on current stream
            # TODO: find ways to check hacks by launching work on additional stream
            torch.cuda.synchronize(device=device)

            # Calculate the elapsed time in milliseconds
            elapsed_time_ms = start_event.elapsed_time(end_event)
            
            if trial >= discard_first:
                if verbose:
                    logical_idx = trial - discard_first + 1
                    print(f"Trial {logical_idx}: {elapsed_time_ms:.3g} ms")
                elapsed_times.append(elapsed_time_ms)


    return elapsed_times


def time_execution_with_do_bench_interface(
    kernel_fn: callable,
    args: list[Any],
    # Not used, as triton do_bench handles adaptive trials
    num_warmup: int = 3, 
    num_trials: int = 10,
    discard_first: int = 1, # not used here
    verbose: bool = True,
    device: torch.device | None = None) -> list[float]:
    """
    Wrapper around Triton's do_bench for kernel timing.

    Uses Triton's adaptive benchmarking with fixed time budgets (warmup=25ms, rep=100ms) [Triton do_bench default].
    The number of trials is determined automatically based on kernel runtime.

    Note: num_warmup, num_trials, discard_first are ignored - included only for 
    API compatibility with other timing functions.

    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        num_warmup: (ignored) Triton controls warmup
        num_trials: (ignored) Triton controls trial count  
        discard_first: (ignored) Not used
        verbose: Whether to print timing info
        device: CUDA device to use

    Returns:
        List of elapsed times in milliseconds

    See: https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()


    from triton import testing as triton_testing
    do_bench_fn = lambda : kernel_fn(*args) # wrap function with arguments
    with torch.cuda.device(device):
        return triton_testing.do_bench(fn=do_bench_fn,
            warmup=25,
            rep=100, 
            grad_to_none=None, 
            quantiles=None, 
            return_mode="all")


def time_execution_with_do_bench_impl(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1, # not used here
    verbose: bool = True,
    device: torch.device | None = None) -> list[float]:
    """
    This is modifying the triton do_bench codebase
    See Triton's implementation for more details
    https://github.com/triton-lang/triton/blob/9073370d5979218d1afa44ec895bbd80e7419a8c/python/triton/testing.py#L127

    Note we duplicate triton's implementation and modify / comment out parts
    to use num_warmup and num_trials that explicitly follows what user define here
    instead of do_bench's version that computes how many times to run warmup and 
    profile based on total warmup and repetition time

    We commented out unused parts and kept only what's needed for kernelbench timing eval
    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        num_warmup: Number of warmup iterations
        num_trials: Number of timing trials
        discard_first: (not used) Trials to discard
        verbose: Whether to print timing info
        device: CUDA device to use, defaults to current device
    Returns:
        List of elapsed times in milliseconds (length = num_trials)
    """

    from triton import runtime as triton_runtime
    device = device if device is not None else torch.cuda.current_device()
    if verbose: 
        print(f"Using do_bench to evaluate kernel on {device}")


    # added to constraint to this device
    with torch.cuda.device(device):  

        # specify device interface (supports both nvidia and amd)
        # under the hood, di is torch.cuda (amd uses a cuda compatible interface)
        di = triton_runtime.driver.active.get_device_interface()

        kernel_fn(*args)
        di.synchronize(device=device)

        # clear l2 cache
        cache = triton_runtime.driver.active.get_empty_cache_for_benchmark()

        # do_bench Estimate the runtime of the function 
        # Here we are not using it not needed since now the warmup and repeat steps are set by the user)
        # start_event = di.Event(enable_timing=True)
        # end_event = di.Event(enable_timing=True)
        # start_event.record()
        # for _ in range(5):
        #     triton_runtime.driver.active.clear_cache(cache)
        #     kernel_fn(*args)
        # end_event.record()
        # di.synchronize()
        # estimate_ms = start_event.elapsed_time(end_event) / 5

        # compute number of warmup and repeat
        # Change
        # n_warmup = max(1, int(warmup / estimate_ms))
        # n_repeat = max(1, int(rep / estimate_ms))
        # n_warmup = warmup
        # n_repeat = rep
        # end of change
        start_event = [di.Event(enable_timing=True) for i in range(num_trials)]
        end_event = [di.Event(enable_timing=True) for i in range(num_trials)]
        # Warm-up
        for _ in range(num_warmup):
            kernel_fn(*args)
        di.synchronize(device=device) 
        
        # Benchmark
        for i in range(num_trials):
            # All KernelBench functions are forward passes, so we don't need to reset gradients
            # we don't want `fn` to accumulate gradient values
            # if it contains a backward pass. So we clear the
            # provided gradients
            # if grad_to_none is not None:
            #     for x in grad_to_none:
            #         x.grad = None
            
            # we clear the L2 cache before each run
            triton_runtime.driver.active.clear_cache(cache)
            # record time of `fn`
            start_event[i].record()
            kernel_fn(*args)
            end_event[i].record()
        # Record clocks
        di.synchronize(device=device)
        times = [s.elapsed_time(e) for s, e in zip(start_event, end_event)]

    if verbose: print('Done with do_bench evaluation')
    return times


def time_execution_with_host_time(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1, # to reduce impact of initialization overhead
    verbose: bool = True,
    device: torch.device | None = None,
) -> list[float]:
    """
    Time a CUDA kernel function over multiple trials using Host (CPU) side timing
    
    This measures host-side wall clock time, E2E latency observed by host
    Note that could take including Python overhead, CUDA launch/runtime costs, synchronization, all GPU work across all streams, and host OS overhaed
    Hence results might be longer than device-side (CUDA event) timings

    Args:
        kernel_fn: Function to time
        args: Arguments to pass to kernel_fn
        num_trials: Number of timing trials to run
        discard_first: Number of first few trials to discard (due to some initialization overhead)
        verbose: Whether to print per-trial timing info
        device: CUDA device to use, if None, use current device

    Returns:
        List of elapsed times in milliseconds
    """
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    # Warm ups
    for _ in range(num_warmup):
        kernel_fn(*args)
        torch.cuda.synchronize(device=device)

    print(f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}")
    elapsed_times = []

    # clear PyTorch allocator cache
    torch.cuda.empty_cache()

    # Actual trials
    for trial in range(num_trials + discard_first):
        # block all streams on device
        torch.cuda.synchronize(device=device)

        # focus on cold_cache performance
        clear_l2_cache(device=device) 

        # CPU-side wall clock time using perf_counter (high-resolution timer)
        start_time = time.perf_counter()
        kernel_fn(*args)
        torch.cuda.synchronize(device=device) # wait for all stream to finish
        # this blocks the CPU until all GPU work on device is done
        # this means all kernels on all streams
        end_time = time.perf_counter()

        # Calculate the elapsed time in milliseconds
        elapsed_time_ms = (end_time - start_time) * 1000
        if trial >= discard_first:
            if verbose:
                logical_idx = trial - discard_first + 1
                print(f"Trial {logical_idx}: {elapsed_time_ms:.3g} ms")
            elapsed_times.append(elapsed_time_ms)

    return elapsed_times

def time_execution_with_nsight_python(
    kernel_fn: callable,
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1, # not used here
    verbose: bool = True,
    device: torch.device | None = None) -> list[float]:
    """
    Time a CUDA kernel function using nsight-python.
    
    Note: nsight returns an average time across num_trials runs.
    Returns a list with a single value (average time) for API consistency.
    GPU time from nsight is in nanoseconds, converted to milliseconds.
    
    Returns:
        List containing one float: average elapsed time in milliseconds
    """
    
    from kernelbench.profile import profile_with_nsight
    
    if device is None:
        if verbose:
            print(f"Using current device: {torch.cuda.current_device()}")
        device = torch.cuda.current_device()

    with torch.cuda.device(device):
        # Warm ups
        for _ in range(num_warmup):
            kernel_fn(*args)
            torch.cuda.synchronize(device=device)
        
        # Clear cache for cold start
        torch.cuda.empty_cache()
        clear_l2_cache(device=device)
        
        if verbose:
            print(f"[Profiling] Using device: {device} {torch.cuda.get_device_name(device)}, warm up {num_warmup}, trials {num_trials}")
        
        # Profile with nsight - returns average time in nanoseconds
        # Wrap kernel function
        def wrapped_kernel():
            return kernel_fn(*args)
        
        # Profile with nsight, use gpu_time_duration.sum metric for GPU time
        metric_values = profile_with_nsight(
            wrapped_kernel,
            metrics=["gpu__time_duration.sum"],
            num_trials=num_trials
        )
        
        # Convert from nanoseconds to milliseconds
        gpu_time_ns = metric_values.get("gpu__time_duration.sum")
        if gpu_time_ns is None:
            raise RuntimeError("Failed to get GPU time from nsight")
        
        # Convert nanoseconds to milliseconds
        # nsight returns average across num_trials, so we return a single value in a list
        gpu_time_ms = gpu_time_ns / 1_000_000.0
        
        if verbose:
            print(f"Average GPU time: {gpu_time_ms:.3f} ms (across {num_trials} trials)")
        
        # NOTE: nsight only returns average time across num_trials, so we return a single value in a list
        # it did run num_trials times, but we only return the average (1 item)
        # Return list with single average value for API consistency
        return [gpu_time_ms]

########################################################
# Timing stats
# tools to help compute speedup and other time
#########################################################
def fetch_baseline_time(
    level_name: str, problem_id: int, dataset: "BaseDataset", baseline_time_filepath: str
) -> Optional[float]:
    """
    Fetch the baseline time from the time

    Note: might be better to just run the refernece using torch eager and compile sometimes
    Will add this as a functionality for eval revamp
    """
    if not os.path.exists(baseline_time_filepath):
        raise FileNotFoundError(
            f"Baseline time file not found at {baseline_time_filepath}"
        )

    with open(baseline_time_filepath, "r") as f:
        baseline_json = json.load(f)

    problem = dataset.get_problem_by_id(problem_id)
    problem_name = problem.name
    baseline_time = baseline_json[level_name].get(problem_name, None)
    return baseline_time


def get_timing_stats(elapsed_times: list[float], device: torch.device = None) -> dict:
    """Get timing statistics from a list of elapsed times.

    Args:
        elapsed_times: List of elapsed times in milliseconds
        device: CUDA device, record device info
    Returns:
        Dict containing mean, std, min, max and num_trials
        all timing are in ms
    """

    stats = {
        "mean": float(f"{np.mean(elapsed_times):.3g}"),
        "std": float(f"{np.std(elapsed_times):.3g}"),
        "min": float(f"{np.min(elapsed_times):.3g}"),
        "max": float(f"{np.max(elapsed_times):.3g}"),
        "num_trials": len(elapsed_times),
    }

    if device:
        stats["hardware"] = torch.cuda.get_device_name(device=device)
        stats["device"] = str(device)  # for debugging

    return stats

