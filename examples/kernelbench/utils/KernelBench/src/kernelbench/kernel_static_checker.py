"""
Kernel Static Checker - Pattern-based validation for GPU kernel code.

The goal is flag reward hacking patterns (both strictly prohibited and possible ones).
through statically examining the code.

In the future we can add 
- AST-based detections 
- LM as a judge checker

Warning: This list is by no means complete and nor this is not a replacement for runtime checks.
We welcome feedback and contributions as community find new ways of hacks.

- Bypass hacks (PyTorch wrapping, try-except fallback, inheritance bypass)
- Disallow some high-level torch operations (depends on the settings)
- Backend implementation requirements, that CUDA or DSL features must be used

Usage:
    result = validate_kernel_static(code, backend="cuda")
    will return a tuple (valid, errors, warnings) 
"""

import re
from typing import List, Tuple, Dict, Any, Optional, Callable, Union

def _strip_comments(code: str) -> str:
    """Remove # and // comments from code."""
    lines = []
    for line in code.split('\n'):
        if '#' in line:
            line = line[:line.index('#')]
        if '//' in line:
            line = line[:line.index('//')]
        lines.append(line)
    return '\n'.join(lines)


# =============================================================================
# BYPASS CHECKS - Strictly Prohibited 
# some of this is from Kevin RL Paper (arxiv:2507.11948)
# =============================================================================

# --- Try-Except Fallback ---
# Rationale: Models wrap incomplete CUDA in exception handlers that fall back to PyTorch.
# This allows them to pass tests without actually implementing the kernel.
TRY_EXCEPT_PATTERNS = [r"\btry\s*:", r"\bexcept\s*:", r"\bexcept\s+\w+"]

# --- Pass Statement / Inheritance Bypass ---
# Rationale: Model inherits from reference class and uses 'pass' to do nothing,
# effectively just calling the parent implementation.
PASS_PATTERN = r"\bpass\b"

def check_code_bypass(code: str) -> Tuple[bool, str]:
    """
    Check for code bypass patterns (strictly prohibited).
    1. Try-Except Fallback: Models wrap incomplete CUDA in exception handlers
       that fall back to PyTorch when custom code fails.
    2. Pass Statement: Models inherit from reference and use 'pass' to do nothing,
       effectively calling parent implementation.
        Uses word boundary for 'pass' to avoid matching 'passed', 'bypass', etc.
    """
    code = _strip_comments(code)
    
    # Check for try-except fallback
    for pattern in TRY_EXCEPT_PATTERNS:
        if re.search(pattern, code):
            return (True, "Contains try-except block (potential fallback bypass)")
    
    # Check for pass statement
    if re.search(PASS_PATTERN, code):
        return (True, "Contains 'pass' statement (inheritance bypass)")
    
    return (False, "")

# Since KernelBench problems uses PyTorch as a reference, there could be settigs where
# Model generated code
# 1. Replaces some (not all) ops with custom kernels, others are kept in Torch
# --> More practical from a performance perspective (ie. make better systems) as you want to use whatever makes the best system for your use case. 
# 2. All compuational ops must be replaced with custom kernels
# --> Could be helpful from an eval (model ability on transpile + optimization) / RL training perspective 
# Depends the setting you use, you can move the checks below (pytorch_wrap, torch_computation_ops) 
# from WARNING to STRICT

# --- PyTorch NN Module Wrapping ---
# Allows: nn.Module, nn.Parameter, nn.ParameterList, nn.ParameterDict, 
#         nn.ModuleList, nn.ModuleDict, nn.init (needed for model structure)
# Blocks: nn.Linear, nn.Conv2d, nn.ReLU, etc. (compute layers)
PYTORCH_DISALLOWED_NN_PATTERN = r'torch\.nn\.(?!(Module|parameter|Parameter|ParameterList|ParameterDict|ModuleList|ModuleDict|init)\b)'

def check_pytorch_wrap(code: str) -> Tuple[bool, str]:
    """
    Check for PyTorch nn module usage (nn.Linear, nn.Conv2d, etc.).
    
    Allows containers (nn.Module, nn.Parameter, nn.init) needed for model structure.
    Blocks compute layers (nn.Linear, nn.Conv2d, nn.ReLU, etc.).
    """
    code = _strip_comments(code)
    if re.search(PYTORCH_DISALLOWED_NN_PATTERN, code):
        return (True, "Uses torch.nn compute layer (only containers, Parameter, init allowed)")
    return (False, "")


# --- Torch Computation Operations ---
# Rationale: These are high-level PyTorch ops that conduct computation.
# Using them directly defeats the purpose of writing custom kernels.
# Includes both torch.* and F.* (torch.nn.functional) patterns.
TORCH_COMPUTATION_OPS = [
    # Matrix operations
    "torch.mm", "torch.bmm", "torch.matmul", "torch.einsum",
    # Convolutions
    "torch.conv1d", "torch.conv2d", "torch.conv3d", "torch.conv",
    "torch.conv_transpose1d", "torch.conv_transpose2d", "torch.conv_transpose3d",
    # Pooling
    "torch.avg_pool1d", "torch.avg_pool2d", "torch.avg_pool3d",
    "torch.max_pool1d", "torch.max_pool2d", "torch.max_pool3d",
    "torch.adaptive_avg_pool1d", "torch.adaptive_avg_pool2d", "torch.adaptive_avg_pool3d",
    "torch.adaptive_max_pool1d", "torch.adaptive_max_pool2d", "torch.adaptive_max_pool3d",
    # Activations
    "torch.relu", "torch.hardtanh", "torch.elu", "torch.selu",
    "torch.leaky_relu", "torch.gelu", "torch.softsign", "torch.softplus",
    "torch.softmax", "torch.log_softmax", "torch.tanh", "torch.sigmoid",
    "torch.hardsigmoid", "torch.silu", "torch.mish",
    # Normalization
    "torch.batch_norm", "torch.group_norm", "torch.layer_norm",
    "torch.instance_norm", "torch.rms_norm", "torch.normalize",
    # Linear & Loss
    "torch.linear", "torch.cross_entropy", "torch.kl_div", "torch.mse_loss",
    "torch.huber_loss", "torch.triplet_margin_loss", "torch.cosine_similarity",
    # Others
    "torch.logsumexp", "torch.clamp", "torch.dropout",
]

# F.* patterns (torch.nn.functional equivalents)
TORCH_FUNCTIONAL_PATTERNS = [
    r"torch\.nn\.functional\.\w+",       # torch.nn.functional.*
    r"\bnn\.functional\.\w+",            # nn.functional.*
    r"\bF\.(conv|linear|relu|gelu|softmax|batch_norm|layer_norm|dropout|max_pool|avg_pool)",
]

def check_torch_computation_ops(code: str) -> Tuple[bool, str]:
    """
    Check for high-level torch computation operations.
    
    Matches both torch.* ops (torch.matmul) and F.* ops (F.relu).
    This check is optional/taste-based. Configure as needed.
    """
    code = _strip_comments(code)
    
    # Check torch.* ops
    torch_pattern = r'\b(' + '|'.join(re.escape(f) for f in TORCH_COMPUTATION_OPS) + r')(?=\s*\(|\s|$)'
    match = re.search(torch_pattern, code)
    if match:
        return (True, f"Uses torch computation op: {match.group(0)}")
    
    # Check F.* / nn.functional ops
    for pattern in TORCH_FUNCTIONAL_PATTERNS:
        match = re.search(pattern, code)
        if match:
            return (True, f"Uses torch.nn.functional op: {match.group(0)}")
    
    return (False, "")

# =============================================================================
# Backend Specific Checks
# =============================================================================

# <========= CUDA CHECKS =========>
# Rationale: Valid CUDA kernels must have __global__ (kernel definition) and
# use load_inline or cpp_extension (PyTorch's inline compilation).
CUDA_COMPILE_PATTERNS = ["load_inline", "cpp_extension"]

def check_cuda_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid CUDA kernel implementation.
    
    Requirements:
    - Must have __global__ void kernel_name (kernel definition)
    - Must have load_inline or cpp_extension (PyTorch inline compilation)
    """
    code = _strip_comments(code)
    if "__global__" not in code:
        return (True, "Missing __global__ kernel definition")
    if not any(p in code for p in CUDA_COMPILE_PATTERNS):
        return (True, "Missing load_inline or cpp_extension for compilation")
    return (False, "")

# <========= TRITON CHECKS =========>
# Rationale: Triton kernels are compiled from @triton.jit decorated functions.
# They must use tl.* operations (tl.load, tl.store, etc.) for actual kernel work.
TRITON_JIT_PATTERN = r"@triton\.(jit|autotune)"
TRITON_OPS_PATTERN = r"\btl\.\w+"

def check_triton_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid Triton kernel implementation.
    
    Requirements:
    - Must have @triton.jit or @triton.autotune decorator
    - Must have tl.* operations (enforces actual Triton code, not wrapper)
    
    Note: Triton's compiler itself prevents PyTorch ops inside @triton.jit.
    """
    code = _strip_comments(code)
    if not re.search(TRITON_JIT_PATTERN, code):
        return (True, "Missing @triton.jit or @triton.autotune")
    if not re.search(TRITON_OPS_PATTERN, code):
        return (True, "No tl.* operations found in Triton kernel")
    return (False, "")


# <========= THUNDERKITTENS CHECKS =========>
# Rationale: ThunderKittens uses warp/warpgroup primitives and tile abstractions.
# Valid TK code must have namespace patterns and tile declarations.
TK_WARP_PATTERNS = [
    r"kittens::warp\b", r"kittens::warpgroup\b",
    r"::warpgroup::", r"::warp::", r"warpgroup::", r"warp::"
]
TK_TILE_PATTERN = r"(?:kittens::)?(?:st|rt)_\w+\s*<[^>]+>"

def check_tk_impl(code: str) -> Tuple[bool, str]:
    """
    Check for valid ThunderKittens kernel implementation.
    
    Requirements:
    - Must have warp/warpgroup namespace patterns (kittens::warp, etc.)
    - Must have tile declarations (st_bf<...>, rt_fl<...>, etc.)
    
    TODO: Add producer-consumer pattern check for complex kernels.
    """
    code = _strip_comments(code)
    if not any(re.search(p, code) for p in TK_WARP_PATTERNS):
        return (True, "Missing ThunderKittens warp/warpgroup patterns")
    if not re.search(TK_TILE_PATTERN, code):
        return (True, "Missing ThunderKittens tile declarations (st_*/rt_*)")
    return (False, "")


# <========= CUTE/CUTLASS CHECKS =========>
# CUTLASS uses cute:: namespace for tensor operations
# Check: https://github.com/NVIDIA/cutlass 
CUTE_PATTERNS = [
    r"cute::",           # cute:: namespace (CuTe library)
    r"cutlass::",        # cutlass:: namespace
    r"from cutlass",     # Python CUTLASS bindings
]

def check_cute_impl(code: str) -> Tuple[bool, str]:
    """Check for valid CUTLASS/CuTe kernel implementation."""
    code = _strip_comments(code)
    if not any(p in code for p in ["cute::", "cutlass::", "from cutlass"]):
        return (True, "Missing cute:: or cutlass:: namespace")
    return (False, "")


# <========= TILELANG CHECKS =========>
# TileLang uses TVM's T.prim_func decorator
# https://github.com/tile-ai/tilelang
TILELANG_PATTERNS = [
    r"@T\.prim_func",    # TVM primitive function decorator
    r"tvm\.build",       # TVM build call
    r"T\.grid",          # TileLang grid
]

def check_tilelang_impl(code: str) -> Tuple[bool, str]:
    """Check for valid TileLang kernel implementation."""
    code = _strip_comments(code)
    if not re.search(r"@T\.prim_func", code):
        return (True, "Missing @T.prim_func decorator")
    return (False, "")


# =============================================================================
# TIMING MANIPULATION CHECKS - Reward Hacking Patterns
# From adversarial hack PR and DeepReinforce blog
# =============================================================================

# <========= STREAM INJECTION CHECKS =========>
# Rationale: Models may use CUDA streams to defer computation, manipulate timing,
# or execute operations asynchronously to cheat benchmark measurements.
STREAM_PATTERNS = [
    r"torch\.cuda\.Stream\s*\(",       # torch.cuda.Stream()
    r"cuda\.Stream\s*\(",               # cuda.Stream() (if imported)
    r"with\s+torch\.cuda\.stream",      # with torch.cuda.stream(...)
    r"\.wait_stream\s*\(",              # .wait_stream() method
    r"\.record_stream\s*\(",            # .record_stream() method
]

def check_stream_injection(code: str) -> Tuple[bool, str]:
    """
    Check for CUDA stream injection patterns.
    
    Detects code that uses CUDA streams to potentially manipulate timing:
    1. Stream creation: torch.cuda.Stream(), cuda.Stream()
    2. Stream context managers: with torch.cuda.stream(...)
    3. Stream synchronization: .wait_stream(), .record_stream()
    
    Rationale: Streams can defer computation or change execution order,
    potentially affecting benchmark timing measurements.
    """
    code = _strip_comments(code)
    
    for pattern in STREAM_PATTERNS:
        if re.search(pattern, code):
            if "wait_stream" in pattern or "record_stream" in pattern:
                return (True, "Uses stream synchronization (potential timing manipulation)")
            elif "with" in pattern:
                return (True, "Uses stream context manager (potential timing manipulation)")
            else:
                return (True, "Uses CUDA streams (potential timing manipulation)")
    
    return (False, "")


# <========= THREAD INJECTION CHECKS =========>
# Rationale: Models may use threading to parallelize work or manipulate execution
# order in ways that could affect benchmark timing.
THREAD_PATTERNS = [
    r"threading\.Thread\s*\(",          # threading.Thread()
    r"import\s+threading",              # import threading
    r"from\s+threading\s+import",       # from threading import ...
    r"multiprocessing\.(Process|Pool|Manager|Queue|Pipe)",
    r"import\s+multiprocessing",        # import multiprocessing
    r"concurrent\.futures",             # concurrent.futures (thread pools)
    r"ThreadPoolExecutor",              # ThreadPoolExecutor
    r"ProcessPoolExecutor",             # ProcessPoolExecutor
]

def check_thread_injection(code: str) -> Tuple[bool, str]:
    """
    Check for thread/multiprocessing injection patterns.
    
    Detects code that uses threading or multiprocessing:
    1. Thread creation: threading.Thread()
    2. Threading imports: import threading
    3. Multiprocessing: Process, Pool, Manager, Queue, Pipe
    4. Concurrent futures: ThreadPoolExecutor, ProcessPoolExecutor
    
    Rationale: Threading can defer computation or change execution order,
    affecting timing measurements.
    
    Note: ProcessPoolExecutor might be legitimate in eval code but should
    not appear in kernel implementations.
    """
    code = _strip_comments(code)
    
    for pattern in THREAD_PATTERNS:
        if re.search(pattern, code):
            if "multiprocessing" in pattern:
                return (True, "Uses multiprocessing (potential timing manipulation)")
            elif "concurrent" in pattern or "Executor" in pattern:
                return (True, "Uses concurrent futures (potential timing manipulation)")
            else:
                return (True, "Uses threading (potential timing manipulation)")
    
    return (False, "")


# <========= LAZY EVALUATION CHECKS =========>
# Rationale: Models may create fake/lazy tensors that don't actually compute
# anything, passing correctness checks without real implementation.
LAZY_TENSOR_PATTERNS = [
    r"_make_subclass",                  # torch.Tensor._make_subclass (common lazy hack)
    r"class\s+\w+.*\(torch\.Tensor\)",  # Custom tensor subclasses
    r"class\s+\w+.*\(Tensor\)",         # Custom tensor subclasses (imported Tensor)
    r"torch\.Tensor\.__new__",          # Direct tensor construction (potential lazy)
]

def check_lazy_eval(code: str) -> Tuple[bool, str]:
    """
    Check for lazy tensor creation patterns.
    
    Detects patterns commonly used to create lazy/fake tensors:
    1. _make_subclass: Common way to create custom tensor subclasses
    2. Custom tensor subclasses: Classes inheriting from torch.Tensor
    3. Direct tensor construction: torch.Tensor.__new__ manipulation
    
    Rationale: Lazy tensors can pass correctness checks without actually
    computing anything, which is a form of reward hacking.
    """
    code = _strip_comments(code)
    
    for pattern in LAZY_TENSOR_PATTERNS:
        if re.search(pattern, code):
            if "_make_subclass" in pattern:
                return (True, "Uses _make_subclass (potential lazy tensor hack)")
            elif "class" in pattern:
                return (True, "Defines custom tensor subclass (potential lazy tensor hack)")
            else:
                return (True, "Uses direct tensor construction (potential lazy tensor hack)")
    
    return (False, "")


# <========= Timing Monkey Patch CHECKS =========>
# Rationale: Models may monkey-patch torch timing functions to fake benchmark results.
# This detects static patterns where timing functions are reassigned.
# especially when relying on timing markers like cuda.Event or time.perf_counter.
TIMING_EVENT_PATCH_PATTERNS = [
    r"torch\.cuda\.Event\.record\s*=",      # torch.cuda.Event.record = ...
    r"torch\.cuda\.Event\.elapsed_time\s*=", # torch.cuda.Event.elapsed_time = ...
    r"torch\.cuda\.synchronize\s*=",         # torch.cuda.synchronize = ...
    r"torch\.cuda\.Event\s*=",               # torch.cuda.Event = ... (replacing class)
    r"time\.perf_counter\s*=",               # time.perf_counter = ...
    r"time\.time\s*=",                       # time.time = ...
]

def check_timing_event_patch(code: str) -> Tuple[bool, str]:
    """
    Check for monkey patching of timing functions (static analysis).
    
    Detects code that reassigns torch or time functions used for benchmarking:
    1. torch.cuda.Event.record = ...
    2. torch.cuda.Event.elapsed_time = ...
    3. torch.cuda.synchronize = ...
    4. time.perf_counter = ...
    
    Rationale: Monkey patching timing functions can fake benchmark results,
    making slow code appear fast.
    """
    code = _strip_comments(code)
    
    for pattern in TIMING_EVENT_PATCH_PATTERNS:
        if re.search(pattern, code):
            return (True, "Reassigns timing function (monkey patch detected)")
    
    return (False, "")


# <========= PRECISION DOWNGRADE CHECKS =========>
# Rationale: Models may downgrade precision internally (FP32 -> FP16)
# to make kernels faster while still returning FP32 output. This is a reward
# hacking pattern where models cheat on timing while appearing correct.
# This might be a valid optimization, but is still worth flagging as a warning.
#
# We use explicit, high-confidence patterns that indicate intentional precision
# downgrading. These patterns have minimal false positives and clear semantic intent.

# Specific patterns that indicate FP32 -> FP16 precision downgrading
FP32_TO_FP16_PATTERNS = [
    # ========== CUDA / CUDA C++ ==========
    # 1.1 Explicit float -> half intrinsics (⭐ gold standard)
    # __float2half(f), __float2half_rn(f)
    r"__float2half(_rn)?\s*\(",
    
    # 1.2 Explicit C-style cast to __half
    # (__half)f
    r"\(\s*__half\s*\)\s*[\w\->\.]+",
    
    # 1.3 static_cast<half> / static_cast<__half>
    # static_cast<half>(f), static_cast<__half>(f)
    r"static_cast\s*<\s*(__half|half)\s*>\s*\(",
    
    # ========== Triton (Python) ==========
    # 2.1 Explicit tl.astype(..., tl.float16) (⭐ best signal)
    # tl.astype(x, tl.float16)
    r"tl\.astype\s*\(\s*[^,]+,\s*tl\.float16\s*\)",
    
    # ========== CUTLASS ==========
    # 3.1 NumericConverter float -> half (⭐ extremely reliable)
    # NumericConverter<half_t, float>
    r"NumericConverter\s*<\s*half_t\s*,\s*float\s*>",
    
    # 3.2 Epilogue output type = FP16
    # LinearCombination<half_t, ...>
    r"LinearCombination\s*<\s*half_t",
    
    # ========== CUTE (when used directly) ==========
    # 4.1 Explicit type conversion utility
    # type_convert<half_t>(x)
    r"type_convert\s*<\s*half_t\s*>\s*\(",
    
    # ========== ThunderKittens ==========
    # 5.1 Explicit half types in cast contexts
    # tk::half(x)
    r"tk::half\s*\(",
    
    # ========== Additional gaping patterns ==========
    # PyTorch explicit casts (common in inline CUDA contexts)
    r"\.half\s*\(",                              # .half() -> FP16
    r"\.float16\s*\(",                           # .float16() -> FP16
    r"\.to\s*\(\s*torch\.(float16|half)\b",      # .to(torch.float16)
    r"\.to\s*\(\s*dtype\s*=\s*torch\.(float16|half)\b",  # .to(dtype=torch.float16)
]

def check_precision_downgrade(code: str, precision: str = "fp32") -> Tuple[bool, str]:
    """
    Check for FP32 -> FP16 precision downgrading patterns in kernel code.
    
    Detects code that downgrades precision from FP32 to FP16 internally,
    which can make kernels faster but with reduced precision. This is a reward
    hacking pattern where models cheat on timing while appearing correct.
    
    Currently only checks for FP32 -> FP16 downgrades. Other precision combinations
    (e.g., FP16 -> BF16) are not flagged.
    
    Args:
        code: Kernel source code
        precision: Required precision - only checks when "fp32" or "float32"
    
    Returns:
        (True, error_message) if FP32 -> FP16 downgrade detected
        (False, "") if no downgrade detected
    
    Examples of detected patterns:
    - .half(), .float16()
    - .to(torch.float16), .to(torch.half)
    - dtype=torch.float16
    - __half, half2 (CUDA)
    - tl.float16 (Triton)
    """
    code = _strip_comments(code)
    precision = precision.lower()
    
    # Normalize precision to standard form
    precision_map = {"fp32": "fp32", "float32": "fp32", "fp16": "fp16", "bf16": "bf16", "bfloat16": "bf16"}
    precision = precision_map.get(precision, precision)
    
    # Only check for FP32 -> FP16 downgrades
    if precision != "fp32":
        return (False, "")
    
    # Check for FP16 patterns
    for pattern in FP32_TO_FP16_PATTERNS:
        if re.search(pattern, code):
            return (True, "Precision downgrade detected: required FP32 but code uses FP16")
    
    return (False, "")

# =============================================================================
# In the future, we can add a AST-based checker and a LM-as-a-judge checker
# =============================================================================


# =============================================================================
# REGISTRY & PRESETS
# =============================================================================

# Check functions can take either (code) or (code, precision) arguments
# Most checks take only code, but precision-dependent checks take both
CHECK_FUNCTIONS: Dict[str, Union[Callable[[str], Tuple[bool, str]], Callable[[str, str], Tuple[bool, str]]]] = {
    # Bypass checks (strict)
    "code_bypass": check_code_bypass,
    "pytorch_wrap": check_pytorch_wrap,
    "timing_event_patch": check_timing_event_patch,  # clearly malicious
    
    # Torch ops (depends on your setups)
    "torch_computation_ops": check_torch_computation_ops,
    
    # Timing manipulation checks (usually warnings)
    "stream_injection": check_stream_injection,
    "thread_injection": check_thread_injection,
    "lazy_eval": check_lazy_eval,
    "precision_downgrade": check_precision_downgrade,  # precision-dependent
    
    # Backend-specific implementation checks
    # should be strict
    "cuda_impl": check_cuda_impl,
    "triton_impl": check_triton_impl,
    "tk_impl": check_tk_impl,
    "cute_impl": check_cute_impl,
    "tilelang_impl": check_tilelang_impl,
}

# Checks that require additional parameters beyond just code
PRECISION_DEPENDENT_CHECKS = {"precision_downgrade"}

# Here are some presets for you to use
# You are welcome to adapt them to your settings
# These checks are NECESSARY for all kernels (strict = error)
STRICT_CHECKS = [
    "code_bypass",
    "timing_event_patch",
    "thread_injection",  
    "lazy_eval",         
]

# Backend-specific checks are added later at entry point
# per backend implementation check, usually strict
BACKEND_IMPL_CHECK = {
    "cuda": "cuda_impl",
    "triton": "triton_impl",
    "thunderkittens": "tk_impl",
    "cute": "cute_impl",
    "cutlass": "cute_impl",  # alias
    "tilelang": "tilelang_impl",
}

# These are optional checks (by user's decision) - flagged as warnings
# Move to STRICT_CHECKS if you want to enforce them
WARNING_CHECKS: List[str] = [
    # up to user to allow program to still have some torch computation ops
    "pytorch_wrap",
    "torch_computation_ops",  
    "stream_injection",       # could have legitimate uses (async ops), but should be careful!
    "precision_downgrade",    # precision downgrading - can be intentional but often a hack
]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def validate_kernel_static(
    code: str,
    backend: str = "cuda",
    precision: str = "fp16",
    forbidden: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate kernel code through statically inspecting the code
    We configure the checks against check groups that we have provided for common hacks.
    Note we do not guarantee that all checks are exhaustive. This is also only on the static level.
    
    Args:
        code: Kernel source code
        backend: "cuda", "triton", or "thunderkittens"
        precision: "fp16", "fp32", or "bf16" (for future precision checks)
        forbidden: Check categories that cause errors (default: STRICT_CHECKS)
        warnings: Check categories that cause warnings (default: WARNING_CHECKS)
        
    Returns:
        (valid, errors, warnings)
        valid: bool
        errors: List[str]
        warnings: List[str]
    """
    # Copy defaults to avoid mutating global lists
    forbidden_checks = list(forbidden) if forbidden is not None else list(STRICT_CHECKS)
    warning_checks = list(warnings) if warnings is not None else list(WARNING_CHECKS)
    
    # Add backend implementation check if specified
    if backend in BACKEND_IMPL_CHECK:
        impl_check = BACKEND_IMPL_CHECK[backend]
        if impl_check not in forbidden_checks:
            forbidden_checks.append(impl_check)
    
    # Aggregate results
    errors: List[str] = []
    warnings_list: List[str] = []
    
    for check_name in set(forbidden_checks + warning_checks):
        if check_name not in CHECK_FUNCTIONS:
            continue
        
        # Handle precision-dependent checks
        if check_name in PRECISION_DEPENDENT_CHECKS:
            has_issue, msg = CHECK_FUNCTIONS[check_name](code, precision)
        else:
            has_issue, msg = CHECK_FUNCTIONS[check_name](code)
        
        if has_issue:
            if check_name in forbidden_checks:
                errors.append(msg)
            else:
                warnings_list.append(msg)
    
    valid = len(errors) == 0 # valid if no errors
    return valid, errors, warnings_list
