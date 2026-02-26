"""
Helpers for Evaluations
"""

import hashlib
import importlib
import json
import linecache
import os, subprocess
import random
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Union, Optional

import numpy as np
import requests
import torch
import torch.nn as nn
from pydantic import BaseModel

from . import timing, dataset

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def get_error_name(e: Exception) -> str:
    """
    Get the error name, for logging purposes
    """
    return f"{e.__class__.__module__}.{e.__class__.__name__}"


def fetch_ref_arch_from_problem_id(problem_id: int, dataset: "BaseDataset", with_name=False) -> Union[str, tuple[str, str]]:
    """
    Fetches the reference architecture for a given problem_id from the dataset.
    """
    if isinstance(problem_id, str):
        problem_id = int(problem_id)

    problem = dataset.get_problem_by_id(problem_id)
    ref_arch = problem.code
    
    if not with_name:
        return ref_arch
    else:
        # Use problem.name as fallback when path is None (e.g., for HuggingFace datasets)
        name = problem.path if problem.path is not None else problem.name
        return (name, ref_arch)


def fetch_ref_arch_from_level_problem_id(level, problem_id, with_name=False):
    kb_dataset = dataset.construct_kernelbench_dataset(level)
    return fetch_ref_arch_from_problem_id(problem_id, kb_dataset, with_name)


def set_seed(seed: int):
    torch.manual_seed(seed)
    # NOTE: this only sets on current cuda device
    torch.cuda.manual_seed(seed)

def get_torch_dtype_from_string(precision: str) -> torch.dtype:
    """
    Get the torch dtype for specific precision
    """
    if precision == "fp32":
        return torch.float32
    elif precision == "fp16":
        return torch.float16
    elif precision == "bf16":
        return torch.bfloat16
    else: # future, FP8, FP4, etc. support?
        raise ValueError(f"Invalid precision not supported: {precision}")

def get_tolerance_for_precision(precision: str | torch.dtype) -> float:
    """
    Get the tolerance from a string representing the percision.
    These tolerances are inspired by torchbench (PyTorch Benchmarking Suite): 
    Reference:
    https://github.com/pytorch/benchmark/blob/cfd835c35d04513ced9a59bd074eeb21dc8187d7/torchbenchmark/util/env_check.py#L519
    """
    if isinstance(precision, str):
        precision = get_torch_dtype_from_string(precision)

    PRECISION_TOLERANCES = {
        # By default for fp32, 1e-4 is used according to torchbench.
        torch.float32: 1e-4,
        # torchbench states for bf16 and fp16, use 1e-3 as tolerance and 1e-2 if it's too strict. 
        # @todo: Let user configure own tolerance as an option
        torch.float16: 1e-2, 
        torch.bfloat16: 1e-2,
    }
    assert precision in PRECISION_TOLERANCES, f"Invalid precision not supported: {precision}"
    return PRECISION_TOLERANCES[precision]
    

class KernelExecResult(BaseModel):
    """
    Single Kernel Execution
    """
    # Execution
    compiled: bool = False
    correctness: bool = False
    metadata: dict = {} # NOTE: to include warning if any

    # Timing
    runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    runtime_stats: dict = {}  # only recorded if we decide to measure performance

    # new: added ref time either through fetching prev runs or through execution
    # could do eager for level 1 and compile for level 2 and 3
    ref_runtime: float = -1.0  # in us, only recorded if we decide to measure performance
    ref_runtime_stats: dict = {} # only recorded if we decide to measure performance


def load_original_model_and_inputs(
    model_original_src: str, context: dict
) -> tuple[nn.Module, callable, callable]:
    """
    Load class from original NN.module pytorch code
    this is pytorch reference and we feed that to model to see if there will be any improvement
    """

    try:
        compile(model_original_src, "<string>", "exec")
    except SyntaxError as e:
        print(f"Syntax Error in original code {e}")
        return None

    try:
        exec(model_original_src, context)  # expose to current namespace
    except Exception as e:
        print(f"Error in executing original code {e}")
        return None

    # these should be defined in the original model code and present in the context
    get_init_inputs_fn = context.get("get_init_inputs")
    get_inputs_fn = context.get("get_inputs")
    Model = context.get("Model")
    return (Model, get_init_inputs_fn, get_inputs_fn)


def load_custom_model_with_tempfile(model_custom_src, entry_point="ModelNew"):
    """
    Writes the provided Python code string to a temporary .py file,
    dynamically imports the module so we can access the modified model class.

    Returns both a Model class and the temporary file. The temporary file must be
    deleted manually be the caller.

    This is a hack that is needed for triton code as compile / exec do not play well
    with the @triton.jit decorator.
    """

    # Create a temporary named file with a .py extension
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp_file:
        # Write the code string into the file
        tmp_file.write(model_custom_src)
        # Capture the path to the file
        tempfile_path = tmp_file.name
        temp_file = tmp_file

    # Create a module specification pointing to our temp file
    spec = importlib.util.spec_from_file_location("temp_module", tempfile_path)
    # Create a new module based on that spec
    temp_module = importlib.util.module_from_spec(spec)
    # Execute the code in the module's namespace
    spec.loader.exec_module(temp_module)

    ModelNew = getattr(temp_module, entry_point)

    # Return the object (class, function, etc.) that was defined in the code
    return ModelNew, temp_file


def load_custom_model(
    model_custom_src: str, context: dict, build_directory: str = None
) -> nn.Module:
    """
    Load class from custom NN.module pytorch code
    this is the code output by LLM with calls to custom cuda kernels
    """
    if build_directory:
        context["BUILD_DIRECTORY"] = build_directory
        # Add import at the start of the source code
        model_custom_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_directory}'\n"
        ) + model_custom_src

    try:
        compile(model_custom_src, "<string>", "exec")
        exec(model_custom_src, context)
        # DANGER: need to delete refernece from global namespace
    except SyntaxError as e:
        print(f"Syntax Error in custom generated code or Compilation Error {e}")
        return None

    ModelNew = context.get("ModelNew")
    return ModelNew


def _cleanup_cuda_extensions():
    """Helper function to cleanup compiled CUDA extensions"""
    # SIMON NOTE: is this necessary?
    import shutil

    torch_extensions_path = os.path.join(
        os.path.expanduser("~"), ".cache", "torch_extensions"
    )
    if os.path.exists(torch_extensions_path):
        shutil.rmtree(torch_extensions_path)


def graceful_eval_cleanup(
    curr_context: dict,
    device: torch.device,
    tempfile: tempfile.NamedTemporaryFile = None,
):
    """
    Clean up env, gpu cache, and compiled CUDA extensions after evaluation
    """  # delete ran-specific function definitions before next eval run
    del curr_context
    # Clear CUDA cache and reset GPU state
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

        # does this help?
        torch.cuda.reset_peak_memory_stats(device=device)

        torch.cuda.synchronize(
            device=device
        )  # Wait for all CUDA operations to complete
    if tempfile:
        tempfile.close()
        os.remove(tempfile.name)

    # _cleanup_cuda_extensions() # SIMON NOTE: is this necessary?


def build_compile_cache_legacy(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible

    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(
            f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}"
        )
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache(
    custom_model_src: str,
    verbose: bool = False,
    build_dir: os.PathLike = None,
) -> tuple[bool, str, str]:
    """
    Try to build the compiled cuda code for sample and store in the cache directory
    Should be able to run on CPUs to do this massively in parallel

    Don't limit ninja to set default number of workers, let it use all the cpu cores possible
    # try do this with a subprocess
    NOTE: currently stdout_buffer does not capture all the compiler warning and failure messages
    Returns:
        tuple[bool, str]: whether compilation is successful, stdout content as string
    """
    context = {}
    stdout_buffer = StringIO()

    if verbose:
        print("[Compilation] Pre-compile custom cuda binaries")

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        # sys.stdout.flush()

        # Capture stdout during compilation
        with redirect_stdout(stdout_buffer), redirect_stderr(stdout_buffer):
            load_custom_model(custom_model_src, context, build_dir)
            # sys.stdout.flush()

        if verbose:
            print(f"[Compilation] Compilation Successful, saved cache at: {build_dir}")
    except Exception as e:
        print(
            f"[Compilation] Failed to compile custom CUDA kernel. Unable to cache, \nError: {e}"
        )
        return False, stdout_buffer.getvalue(), str(e)

    return True, stdout_buffer.getvalue(), None


def build_compile_cache_with_capturing(
    custom_model_src: str, verbose: bool = False, build_dir: os.PathLike = None
) -> tuple[int, str, str]:
    """
    Write a temporary python file to compile the custom model on CPU
    Captures the return code, stdout, and stderr
    This works for capturing, build_compile_cache does not
    """
    if build_dir:
        # Add import at the start of the source code
        custom_model_src = (
            "import os\n" f"os.environ['TORCH_EXTENSIONS_DIR'] = '{build_dir}'\n"
        ) + custom_model_src

    kernel_hash = hash(custom_model_src)
    # tmp is a temp python file we write to for compilation
    tmp = os.path.join(build_dir, f"tmp_{kernel_hash}.py")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    with open(tmp, "w", encoding="utf-8") as f:
        f.write(custom_model_src)

    # Execute the temporary Python file and capture output
    process = subprocess.Popen(
        ["python", tmp], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    returncode = process.returncode

    # Clean up temporary file
    os.remove(tmp)

    if verbose:
        print("[CPU Precompile] return code: ", returncode)
        print("[CPU Precompile] stdout: \n", stdout.decode("utf-8"))
        print("[CPU Precompile] stderr: \n", stderr.decode("utf-8"))

    return returncode, stdout.decode("utf-8"), stderr.decode("utf-8")


def _process_input_tensor(input, device, backend="cuda", precision=torch.float32):
    """
    Helper function to move tensors to the correct device and apply backend-specific dtype casting.
    
    Args:
        input: Input tensor or non-tensor value
        device: Target CUDA device
        backend: Backend type (e.g., 'cuda', 'triton', 'cute')
        precision: torch.dtype 
    Returns:
        Processed tensor on correct device with correct dtype, or original value if not a tensor
    """

    # sometimes things like init inputs are floats (like in the case of labels / targets, classification losses, etc.) 
    if not isinstance(input, torch.Tensor):
        return input
    
    # cast to the desired percision dtype for activations
    input_tensor = input.to(dtype=precision)
    
    # Default for all other backends and float types
    return input_tensor.to(device=device)


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    measure_performance: bool = False,
    timing_method: str = "cuda_event", # see timing.py
    verbose: bool = False,
    build_dir: os.PathLike = None,
    device: Union[torch.device, int] = (
        torch.cuda.current_device() if torch.cuda.is_available() else None
    ),  # have to run on GPU
    backend: str = "cuda",  # can be 'cuda', 'triton', 'tilelang', or 'cute'
    precision: torch.dtype = torch.float32,

    # Guard against potential reward hacking [optional but ongoing enhancement]
    check_for_excessive_speedup: bool = True,
    excessive_speedup_threshold: float = 10, # flag if the kernel is more than <excessive_speedup_threshold>x faster than the reference
) -> KernelExecResult:
    """
    Evaluate the custom kernel against the original model

    NOTE: we are thinking about refactor this to be more modularized 
    and we can add more checks as our other ongiong PRs are working on

    num_correct_trials: number of trials to initialize different random inputs; correctness pass only if all trials pass
    num_perf_trials: run the evalutation many times to take the average
    device: GPU (cuda) device to run the evalutation on
    backend: str, one of 'cuda', 'triton', 'tilelang', or 'cute'
    precision: torch.dtype for computation (note: tilelang only supports fp16)
    timing_method: str, method to time kernel, see timing.py for more details 

    ONGOING EFFORT to refactor and modularize this, and adding more tests for eval.
    """
    # TODO: check device is busy
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Eval"
    
    if backend.lower() == "tilelang":
        assert precision == torch.float16 or precision == torch.bfloat16, "TileLang only supports fp16 or bfloat16"
    
    torch.set_printoptions(
        precision=4,  # Decimal places
        threshold=10,  # Total number of elements before truncating
        edgeitems=3,  # Number of elements at beginning and end of dimensions
        linewidth=80,  # Maximum width before wrapping
    )

    # set CUDA device
    torch.cuda.set_device(device)
    
    # Backends that use tempfile approach and need CUDA_VISIBLE_DEVICES
    # TileLang, Triton, and CuTe all use tempfile for proper module loading
    uses_tempfile = backend.lower() in ["triton", "tilelang", "cute"]
    
    metadata = {}  # for storing result metadata
    metadata["hardware"] = torch.cuda.get_device_name(device=device)
    metadata["device"] = str(device)  # for debugging

    if uses_tempfile:
        # need to set env var for triton/cute code to guarantee no wrong device shenanigans
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert (
                device.type == "cuda"
            ), "CUDA is not availible on device, cannot run Eval"
            device_num = device.index
        else:
            raise ValueError(
                f"device must be an int or torch.device, got {type(device)}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_num)
    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        original_model_src, context
    )
    set_seed(seed_num)  # set seed for reproducible input
    init_inputs = get_init_inputs()
    
    # Convert inputs to appropriate dtypes for GPU computation
    init_inputs = [_process_input_tensor(x, device, backend, precision) for x in init_inputs]
    
    with torch.no_grad():
        set_seed(seed_num)  # set seed for reproducible weights
        original_model = Model(*init_inputs)
        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    # this is where compilation happens
    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"  # compile with device side assertion
        tempfile = None
        # add hash for later to distinguish between multi-turn kernels
        
        backend_lower = backend.lower()
        if backend_lower in ["triton", "tilelang", "cute"]:
            # Use tempfile approach for triton, tilelang, and cute
            # These DSLs require proper module import for JIT decorators to work
            ModelNew, tempfile = load_custom_model_with_tempfile(
                custom_model_src, entry_point="ModelNew"
            )
        else:
            # Default CUDA backend
            ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.cuda.synchronize(device=device)  # not sure if this is too much
    except Exception as e:
        print(
            f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}"
        )
        # TODO: add metadata for compilation error (how to we get the compilation error message?)

        if "lock" in str(e) or "No such file or directory" in str(e):
            # this is a lock file error, likely due to concurrent compilation
            # this does not necessarily mean the compilation failed, but we should retry
            print(
                f"[Eval] Lock file error during compilation, Please retry. Error: {e}"
            )
            graceful_eval_cleanup(context, device, tempfile)
            return None
        else:
            metadata["compilation_error_name"] = get_error_name(e)
            metadata["compilation_error"] = e
            graceful_eval_cleanup(context, device, tempfile)
            return KernelExecResult(
                compiled=False, metadata=metadata
            )  # skip further steps

    # Check if ModelNew was successfully loaded (load_custom_model returns None on syntax errors)
    if ModelNew is None:
        print(
            "Failed to load custom model: Syntax error or ModelNew not found in generated code. Record as compilation failure."
        )
        metadata["compilation_error_name"] = "SyntaxError"
        metadata["compilation_error"] = "Syntax error in custom generated code or ModelNew not found"
        graceful_eval_cleanup(context, device, tempfile)
        return KernelExecResult(
            compiled=False, metadata=metadata
        )  # skip further steps

    # at this point we passed compilation
    try:
        with torch.no_grad():
            set_seed(seed_num)  # set seed for reproducible weights
            custom_model = ModelNew(*init_inputs)
            assert hasattr(custom_model, "forward")
            original_model = original_model.to(device=device, dtype=precision)
            custom_model = custom_model.to(device=device, dtype=precision)
            torch.cuda.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(
            f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}"
        )
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        graceful_eval_cleanup(context, device, tempfile)
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )  # skip further steps

    kernel_exec_result = None

    # Check Correctness
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        kernel_exec_result = run_and_check_correctness(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
            backend=backend,
            precision=precision,
        )
    except Exception as e:
        # TODO: add metadata for runtime error e.g. error in launching kernel, illegal memory access, ...
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        kernel_exec_result = KernelExecResult(
            compiled=True, correctness=False, metadata=metadata
        )

    # Measure Performance [Optional] | conditioned on compilation + correctness + no exception so far
    if measure_performance:
        try:
            if kernel_exec_result and kernel_exec_result.correctness:
                if verbose:
                    print("[Eval] Measuring Performance as Sample is Correct")

                torch.cuda.synchronize(device=device)
                set_seed(seed_num)
                inputs = get_inputs()
                # Convert inputs for performance measurement
                inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]
                
                model_new = custom_model.to(device=device, dtype=precision)
                torch.cuda.synchronize(device=device)

                # support multiple timing backend
                timing_fn = timing.get_timing_function(timing_method)
                elapsed_times = timing_fn(
                    model_new,
                    inputs,
                    num_trials=num_perf_trials,
                    verbose=verbose,
                    device=device,
                )
                runtime_stats = timing.get_timing_stats(elapsed_times, device=device)

                if verbose:
                    print(f"[Eval] Performance Stats: {runtime_stats}")
                kernel_exec_result.runtime = runtime_stats["mean"]
                kernel_exec_result.runtime_stats = runtime_stats

        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = e

    # To get base PyTorch time (eager, various compile modes)
    # please use timing.measure_ref_program_time()   


    ###############################################################
    # [Experimental] to be modularized
    # Condition: custom kernel ModelNew is correct and we are able to time it correctly with kernel_exec_result
    # We are working on preventing excessive speedup issues
    ##############################################################

    if measure_performance and check_for_excessive_speedup:  # experimental: hence able to shut off codepath if needed
    
        if verbose:
            print("[Eval] Additional checks to flag excessive speedup")

        torch.cuda.synchronize(device=device)
        set_seed(seed_num)
        inputs = get_inputs()
        # Convert inputs for performance measurement
        inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]
        
        model_new = custom_model.to(device=device, dtype=precision)
        torch.cuda.synchronize(device=device)

        # time PyTorch reference function
        # same timing_fn as specified from before
        timing_fn = timing.get_timing_function(timing_method)
        reference_elapsed_times = timing_fn(
            original_model,
            inputs, # ideally cloned for extra safety but handled already in correctness check
            num_trials=num_perf_trials,
            verbose=verbose,
            device=device,
        )
        reference_runtime_stats = timing.get_timing_stats(reference_elapsed_times, device=device)
        kernel_exec_result.ref_runtime = reference_runtime_stats["mean"]
        kernel_exec_result.ref_runtime_stats = reference_runtime_stats

        # Compute Effective Speedup
        effective_speedup = kernel_exec_result.ref_runtime / kernel_exec_result.runtime

        # TODO: integrate SoL estimation for each unique program on designated hardware
        # for now, we will use a heuristics such as 5-10x which is very hard to achieve

        if verbose:
            print(f"[Eval] Effective Speedup is {effective_speedup:.2f}x using timing method {timing_method}")

        if effective_speedup > excessive_speedup_threshold:
            kernel_exec_result.metadata["excessive_speedup"] = True
            
            print(f"[WARNING] Excessive speedup {effective_speedup:.2f}x over {excessive_speedup_threshold}x threshold detected")
            print(f"[WARNING] Double check your kernel carefully to ensure it is not reward hacking.")


    graceful_eval_cleanup(context, device, tempfile)
    return kernel_exec_result


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate=False,
    max_length=200,
):
    """
    max_length characters

    NOTE: I can't get torch truncate to work during exception handling so I have this for now
    """
    # Truncate exception message if too long
    exception_str = str(exception_msg)
    if truncate and len(exception_str) > max_length:
        exception_str = exception_str[: max_length - 3] + "..."

    if verbose:
        print(f"[Exception {exception_type}] {exception_str} ")
    metadata[exception_type] = exception_str

    return metadata


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose: bool =False,
    seed: int =42,
    device: Optional[torch.device] =None,
    backend: str ="cuda",
    precision: torch.dtype =torch.float32,
) -> KernelExecResult:
    """
    run the model and check correctness,
    assume model already loaded and compiled (loaded and compiled in the caller)
    this is all on GPU, requiring cuda device and transfer .cuda()

    num_correct_trials: run the evalutation multiple times with (ideally) different random inputs to ensure correctness
    backend: backend type for handling dtype conversions
    precision: torch.dtype
    """
    pass_count = 0

    # Generate num_correct_trials seeds deterministically from the initial seed
    torch.manual_seed(seed)
    correctness_trial_seeds = [
        torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)
    ]

    with torch.no_grad():

        for trial in range(num_correct_trials):

            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            # Convert inputs to appropriate dtypes for GPU computation
            inputs = [_process_input_tensor(x, device, backend, precision) for x in inputs]

            set_seed(trial_seed)
    
            model = original_model_instance.to(device=device, dtype=precision)

            set_seed(trial_seed)
     
            model_new = new_model_instance.to(device=device, dtype=precision)

            output = model(*inputs)
            torch.cuda.synchronize(device=device)
            # ensure all GPU operations are completed before checking results

            try:
                output_new = model_new(*inputs)
                torch.cuda.synchronize(device=device)
                if output.shape != output_new.shape:
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Output shape mismatch: Expected {output.shape}, got {output_new.shape}",
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    if verbose:
                        print(
                            f"[FAIL] trial {trial}: Output shape mismatch: Expected {output.shape}, got {output_new.shape}"
                        )
                    return KernelExecResult(
                        compiled=True, correctness=False, metadata=metadata
                    )

                # in torchbench, they use both precisions for atol and rtol
                # kernelbench v0 and v0.1 uses fp32, atol = rtol = 1e-02
                # now we will return the tolerance from get_tolerance_for_precision
                tolerance = get_tolerance_for_precision(precision)
                # check output value difference
                if not torch.allclose(
                    output, output_new, atol=tolerance, rtol=tolerance
                ):  # fail
                    max_diff = torch.max(torch.abs(output - output_new)).item()
                    avg_diff = torch.mean(torch.abs(output - output_new)).item()
                    metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                    metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                    metadata["correctness_issue"] = "Output mismatch"
                    if verbose:
                        print(f"[FAIL] trial {trial}: Output mismatch")
                else:  # pass
                    pass_count += 1
                    if verbose:
                        print(f"[PASS] trial {trial}: New Model matches Model")

            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")
                print("\n[Full Traceback]:")
                traceback.print_exc()
                print("\n")

                metadata = register_and_format_exception(
                    "runtime_error", e, metadata, truncate=True
                )
                metadata["runtime_error_name"] = get_error_name(e)
                # Also store the full traceback in metadata for debugging
                metadata["runtime_error_traceback"] = traceback.format_exc()
                return KernelExecResult(
                    compiled=True, correctness=False, metadata=metadata
                )
                # break

    if verbose:
        print(
            f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}"
        )

    # put all the useful info here!
    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    else:
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def check_metadata_serializable(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings
    """
    try:
        json.dumps(metadata)
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings
        metadata = {
            "eval_0": {
                k: (
                    str(v)
                    if not isinstance(
                        v, (dict, list, str, int, float, bool, type(None))
                    )
                    else v
                )
                for k, v in metadata["eval_0"].items()
            }
        }
        print(
            f"[WARNING] Metadata now converted to string: {metadata} to be JSON serializable"
        )

    return metadata


def check_metadata_serializable_all_types(metadata: dict):
    """
    Ensure metadata is JSON serializable,
    if not, convert non-serializable values to strings recursively
    """

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    try:
        json.dumps(metadata)
        return metadata
    except (TypeError, OverflowError) as e:
        print(f"[WARNING] Metadata is not JSON serializable, error: {str(e)}")
        # Convert non-serializable values to strings recursively
        converted_metadata = convert_to_serializable(metadata)
        print(
            f"[WARNING] Metadata now converted to be JSON serializable: {converted_metadata}"
        )
        return converted_metadata


# if __name__ == "__main__":
# fetch_kernel_from_database("kernelbench_prompt_v2_level_2", 1, 1, "http://localhost:9091")
# print(fetch_ref_arch_from_level_problem_id("2", 1, with_name=True))
# Note: fetch_baseline_time is available in kernelbench.timing module