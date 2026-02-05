"""LLM prompts, signatures, and constants for KernelBench."""

import dspy

# =============================================================================
# CONSTANTS
# =============================================================================

LLM = "openai/gpt-5"
TIMEOUT = 360

# =============================================================================
# PROMPTS
# =============================================================================

BACKGROUND = """
# KernelBench

KernelBench is a benchmark for generating optimized CUDA kernels. Given a PyTorch model (Model), you must produce ModelNew - a drop-in replacement that uses custom CUDA kernels via torch.utils.cpp_extension.load_inline.

Requirements:
- ModelNew must have the same interface as Model (same __init__ args, same forward signature)
- Output must be numerically identical to the PyTorch reference
- Goal: run faster than the PyTorch baseline while maintaining correctness

You may replace some operators with custom CUDA kernels and leave others as standard PyTorch ops.

## Refinement Guidance (for automatic refiner)
When fixing errors:
1. Compilation errors: Check includes, syntax, template parameters
2. Runtime errors: Check thread/block configuration, memory access
3. Correctness errors: Compare output shapes and values carefully
4. Performance: Profile bottlenecks, optimize memory access patterns
"""

KERNEL_GEN_PROMPT = """Write a CUDA kernel to replace the given PyTorch model for better performance.
Output a complete Python file with ModelNew using load_inline. Include all imports."""

OBJECTIVE = "Generate an LLM prompt that produces fast, correct CUDA kernels outperforming PyTorch baselines."


# =============================================================================
# SIGNATURES
# =============================================================================

class KernelGenSig(dspy.Signature):
    """Generate a CUDA kernel for a PyTorch model."""
    prompt: str = dspy.InputField(desc="Generation instructions")
    ref_arch: str = dspy.InputField(desc="PyTorch model to optimize")
    cuda_docs: str = dspy.InputField(desc="Relevant CUDA documentation")
    code: str = dspy.OutputField(desc="Complete CUDA kernel code")
