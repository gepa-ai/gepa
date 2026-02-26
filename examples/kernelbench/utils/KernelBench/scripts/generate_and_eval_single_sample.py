import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import modal

from kernelbench.eval import eval_kernel_against_ref
from kernelbench.prompt_constructor_toml import get_prompt_for_backend, get_custom_prompt
from kernelbench.utils import (
    create_inference_server_from_presets,
    extract_first_code,
    query_server,
    set_gpu_arch,
)
from kernelbench.eval import get_torch_dtype_from_string
"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging

Example usage:
python3 scripts/generate_and_eval_single_sample.py dataset_src=huggingface level=1 problem_id=1 eval_mode=local server_type=google model_name=gemini/gemini-2.5-flash max_tokens=8192 temperature=0.0
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)


class EvalConfig(Config):
    def __init__(self):

        self.dataset_src = REQUIRED  # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED
        # NOTE: this is the logical index (problem id the problem_name)\
        self.problem_id = REQUIRED

        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "local" 
        # only support local for now
        # see scripts/eval_from_generations_modal.py for modal evaluation
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ["Ada"]
        self.precision = "fp32" # options ["fp32", "fp16", "bf16"]

        # Inference config
        self.server_type = REQUIRED
        self.model_name = REQUIRED
        self.max_tokens = None
        self.temperature = None
        
        # Reasoning model specific parameters
        self.is_reasoning_model = False  # set to True for o1, o3, Gemini 2.5 thinking, etc.
        self.reasoning_effort = None  # for o1/o3: "low", "medium", "high"
        self.budget_tokens = 0  # for Claude extended thinking mode

        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False

        self.log = False
        self.log_prompt = False
        self.log_generated_kernel = False
        self.log_eval_result = False

        self.backend = "cuda"
        self.timing_method = "cuda_event"  # see timing.py

        # Prompt construction
        self.prompt_option = "one_shot"  # choices: zero_shot, one_shot, few_shot
        self.include_hardware_info = False
        self.hardware_gpu_name = None
        self.custom_prompt_key = None

        self.check_kernel = True  # [experimental] optional static checker catching potential hacking patterns

    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    Keep it simple: Generate and evaluate a single sample
    Note: will shorten code logic to make this as simple as possible
    """
    from kernelbench.utils import SERVER_PRESETS
    
    if config.server_type and config.server_type in SERVER_PRESETS:
        preset = SERVER_PRESETS[config.server_type]
        if config.model_name is None or config.model_name == "None":
            config.model_name = preset.get("model_name", "None")
        if config.max_tokens is None or config.max_tokens == "None":
            config.max_tokens = preset.get("max_tokens", "None")
        if config.temperature is None or config.temperature == "None":
            config.temperature = preset.get("temperature", "None")
    
    # Convert string boolean to actual boolean for reasoning model flag
    if isinstance(config.is_reasoning_model, str):
        config.is_reasoning_model = config.is_reasoning_model.lower() in ['true', '1', 'yes']
    
    print(f"Starting Eval with config: {config}")

    # Configurations - Unified dataset loading (works for both HF and local)
    from kernelbench.dataset import construct_kernelbench_dataset
    
    dataset = construct_kernelbench_dataset(
        level=config.level,
        source=config.dataset_src,
        dataset_name=config.dataset_name,
    )

    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)  # otherwise build for all architectures

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)

    # Problem Checks
    num_problems = len(dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(
        f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}"
    )

    # Fetch problem - unified interface, no branching needed
    problem = dataset.get_problem_by_id(config.problem_id)
    ref_arch_src = problem.code
    problem_name = problem.name

    # 2. Generate Sample
    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        time_generation=True,
        is_reasoning_model=config.is_reasoning_model,
        reasoning_effort=config.reasoning_effort,
        budget_tokens=config.budget_tokens,
    )

    # Prompt Construction (Note: could be shortened in future PR)
    custom_prompt_key = getattr(config, "custom_prompt_key", None)
    if isinstance(custom_prompt_key, str):
        trimmed = custom_prompt_key.strip()
        if trimmed.lower() in {"", "none"}:
            custom_prompt_key = None
        else:
            custom_prompt_key = trimmed
    config.custom_prompt_key = custom_prompt_key

    # Use appropriate prompt constructor based on backend
    prompt_option = str(config.prompt_option).lower()
    valid_prompt_options = {"zero_shot", "one_shot", "few_shot"}
    include_hardware = config.include_hardware_info
    if isinstance(include_hardware, str):
        include_hardware = include_hardware.lower() in ["true", "1", "yes"]
    config.include_hardware_info = include_hardware

    supported_backends = {"cuda", "triton", "tilelang", "cute", "thunderkittens"}
    backend = config.backend.lower()
    if backend not in supported_backends:
        raise ValueError(
            f"Unsupported backend: {config.backend}. Must be one of {sorted(supported_backends)}."
        )

    if backend == "tilelang":
        config.precision = "fp16" # tilelang only operates with fp16
        config.hardware_gpu_name = config.hardware_gpu_name or getattr(config, "gpu", None)
    
    if backend == "thunderkittens":
        config.precision = "bf16"

    if not custom_prompt_key:
        if prompt_option not in valid_prompt_options:
            raise ValueError(
                f"Invalid prompt_option '{config.prompt_option}'. "
                f"Must be one of {sorted(valid_prompt_options)}."
            )
        if include_hardware and not config.hardware_gpu_name:
            raise ValueError(
                "include_hardware_info is True but hardware_gpu_name is not provided."
            )

    if custom_prompt_key:
        custom_prompt = get_custom_prompt(
            custom_prompt_key,
            ref_arch_src=ref_arch_src,
            backend=backend,
            option=prompt_option,
            precision=config.precision,
            include_hardware=include_hardware,
            gpu_name=config.hardware_gpu_name,
        )
    else:
        custom_prompt = get_prompt_for_backend(
            ref_arch_src,
            backend,
            option=prompt_option,
            precision=config.precision,
            include_hardware=include_hardware,
            gpu_name=config.hardware_gpu_name,
        )
    
    os.makedirs(config.logdir, exist_ok=True)

    if config.log_prompt:
        with open(os.path.join(config.logdir, f"prompt_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
            f.write(custom_prompt)

    # Query server with constructed prompt
    custom_kernel = inference_server(custom_prompt)
    custom_kernel = extract_first_code(custom_kernel, ["python", "cpp"])

    # check LLM is able to generate custom kernel code
    assert (
        custom_kernel is not None
    ), f"Custom {config.backend} kernel code generation failed"

    # Optional: static code checker for kernel code using regex matching
    # NOTE: by no means is this checker complete, but it could help catch some potential hacks
    if config.check_kernel:
        from kernelbench.kernel_static_checker import validate_kernel_static
        static_check_status, errors, warnings = validate_kernel_static(
            custom_kernel,
            backend=config.backend,
            precision=config.precision,
        )
        assert static_check_status, f"Static check failed for level {config.level} problem {config.problem_id}. Errors: {errors}. Warnings: {warnings}"
        if warnings:
            print(f"Static check warnings for level {config.level} problem {config.problem_id}: {warnings}")

    # this should be optional
    if config.log:
        with open(os.path.join(config.logdir, f"generated_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_kernel)

    # 3. Evaluate Kernel
    # NOTE: no need to wrap around process here as only a single sample
    # see batch eval for examples of process isolation
    kernel_exec_result = eval_kernel_against_ref(
        ref_arch_src,
        custom_kernel,
        verbose=config.verbose,
        measure_performance=True,
        timing_method=config.timing_method,
        num_correct_trials=5,
        num_perf_trials=100,
        backend=config.backend,
        precision=get_torch_dtype_from_string(config.precision),
    )

    print(
        f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}"
    )

    if config.log:
        with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "a",) as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(str(kernel_exec_result))


if __name__ == "__main__":
    main()