"""LLM prompts and signatures for KernelBench."""

import dspy

BACKGROUND = """
# KernelBench

KernelBench is a benchmark for generating optimized CUDA kernels. Given a PyTorch model (Model), you must produce ModelNew - a drop-in replacement that uses custom CUDA kernels via torch.utils.cpp_extension.load_inline.

Requirements:
- ModelNew must have the same interface as Model (same __init__ args, same forward signature)
- Output must be numerically identical to the PyTorch reference
- Goal: run faster than the PyTorch baseline while maintaining correctness

You may replace some operators with custom CUDA kernels and leave others as standard PyTorch ops.
"""

KERNEL_GEN_PROMPT = """Write a CUDA kernel to replace the given PyTorch model for better performance.
Output a complete Python file with ModelNew using load_inline. Include all imports."""

REFINER_PROMPT = """You are the Refiner in an iterative optimization loop.

The optimization loop you are in has the following objective:
{objective}

Your job is to help achieve the objective by carefully looking at the previous attempt made and trying to improve it.
You will receive a candidate (code/prompt/anything from a previous attempt) along with error feedback and evaluation metrics. Your job is to produce an improved version that scores higher.

You will be given:
- The previous candidate code/prompt/anything from a previous attempt
- Error feedback (if any errors occurred)
- Performance metrics and evaluation results

Your task:
1. Diagnose: What caused the error or suboptimal performance?
2. Strategize: What specific changes will fix the issue or improve the score?
3. Implement: Output the complete improved code/prompt/anything from a previous attempt.

The optimization loop will evaluate your output and continue refining. Aim for a correct, improved refinement.
"""


class KernelGenSig(dspy.Signature):
    """Generate a CUDA kernel for a PyTorch model."""
    prompt: str = dspy.InputField(desc="Generation instructions")
    ref_arch: str = dspy.InputField(desc="PyTorch model to optimize")
    cuda_docs: str = dspy.InputField(desc="Relevant CUDA documentation")
    code: str = dspy.OutputField(desc="Complete CUDA kernel code")


class RefinerSig(dspy.Signature):
    """Fix a failed CUDA kernel."""
    prompt: str = dspy.InputField(desc="Refinement instructions")
    failed_code: str = dspy.InputField(desc="Failed kernel code")
    ref_arch: str = dspy.InputField(desc="PyTorch model to match")
    error: str = dspy.InputField(desc="Error feedback")
    cuda_docs: str = dspy.InputField(desc="Relevant CUDA documentation")
    code: str = dspy.OutputField(desc="Fixed CUDA kernel code")
