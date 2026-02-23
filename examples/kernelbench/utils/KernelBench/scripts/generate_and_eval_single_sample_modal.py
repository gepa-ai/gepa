'''
Example Usage:
uv run python scripts/generate_and_eval_single_sample_modal.py dataset_src=huggingface level=1 problem_id=1 eval_mode=modal gpu=L40S 
    server_type=gemini model_name=gemini-2.5-flash max_tokens=4096 temperature=0.0
'''

import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import modal

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.utils import extract_first_code, query_server, set_gpu_arch, create_inference_server_from_presets

app = modal.App("eval_single_sample")

"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}

class EvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"


        # Problem Specification
        self.level = REQUIRED
        # NOTE: this is the logical index (problem id the problem_name)\
        self.problem_id = REQUIRED

        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "modal"
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu = "L40S"
        self.gpu_arch = ['Ada']
        self.precision = "fp32" # options ["fp32", "fp16", "bf16"]

        # Inference config
        self.server_type = None
        self.model_name = None
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
        # Prompt generation settings
        self.prompt_option = "one_shot"  # zero_shot, one_shot, few_shot
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

cuda_version = "13.0.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

SRC_DIR = os.path.join(REPO_TOP_DIR, "src")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git",
                "gcc-10",
                "g++-10",
                "clang" # note i skip a step
                )

    .uv_sync(uv_project_dir=REPO_TOP_DIR, extras=["gpu"])
    .run_commands("git clone -b main https://github.com/HazyResearch/ThunderKittens.git /root/ThunderKittens")
    .env({
        "THUNDERKITTENS_ROOT": "/root/ThunderKittens",
        "PYTHONPATH": "/root:/root/src"
    })
    .add_local_dir(SRC_DIR, remote_path="/root/src")  # must be last
)

@app.cls(image=image)
class EvalFunc:

    @modal.method()
    def eval_single_sample_modal(self, ref_arch_src, custom_kernel, verbose, gpu_arch, backend, precision, timing_method):
        # 3. Evaluate Kernel
        # NOTE: no need to wrap around process here as only a single sample
        # see batch eval for examples of process isolation
        from kernelbench.eval import eval_kernel_against_ref
        from kernelbench.eval import get_torch_dtype_from_string
        # Use utility function to set the GPU architecture in the modal environment
        from kernelbench.utils import set_gpu_arch as modal_set_gpu_arch
        modal_set_gpu_arch(gpu_arch)
        return eval_kernel_against_ref(
            ref_arch_src, custom_kernel, verbose=verbose, measure_performance=True, 
            timing_method=timing_method,
            num_correct_trials=5, num_perf_trials=100, backend=backend, precision=get_torch_dtype_from_string(precision)
        )

@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    
    """
    Keep it simple: Generate and evaluate a single sample
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
    dataset = construct_kernelbench_dataset(
        level=config.level,
        source=config.dataset_src,
        dataset_name=config.dataset_name,
    )

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)
        
    # Problem Checks
    num_problems = len(dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}")

    # Fetch problem - unified interface, no branching needed
    problem = dataset.get_problem_by_id(config.problem_id)
    ref_arch_src = problem.code
    problem_name = problem.name
    
    
    # 2. Generate Sample
    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens,
                                                        verbose=config.verbose, 
                                                        time_generation=True,
                                                        is_reasoning_model=config.is_reasoning_model,
                                                        reasoning_effort=config.reasoning_effort,
                                                        budget_tokens=config.budget_tokens)
    

    custom_prompt_key = getattr(config, "custom_prompt_key", None)
    if isinstance(custom_prompt_key, str):
        trimmed = custom_prompt_key.strip()
        if trimmed.lower() in {"", "none"}:
            custom_prompt_key = None
        else:
            custom_prompt_key = trimmed
    config.custom_prompt_key = custom_prompt_key

    # Checks if user has inputted a valid argument for how many examples they want to give as context to the model
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

    #tilelang only supports fp16 or bf16
    if backend == "tilelang":
        config.precision = "fp16"
        config.hardware_gpu_name = config.hardware_gpu_name or getattr(config, "gpu", None)
    
    # thunderkittens can use bf16 or fp16 by default, also set default GPU to H100
    if backend == "thunderkittens":
        config.precision = "bf16"
        config.gpu = "H100"

    if not custom_prompt_key:
        if prompt_option not in valid_prompt_options:
            raise ValueError(
                f"Invalid prompt_option '{config.prompt_option}'. Must be one of {sorted(valid_prompt_options)}."
            )
        if include_hardware and not config.hardware_gpu_name:
            raise ValueError(
                "include_hardware_info is True but hardware_gpu_name is not provided."
            )

    # Lazy import prompt constructor
    from kernelbench.prompt_constructor_toml import get_prompt_for_backend, get_custom_prompt

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
        
    if config.log_prompt:
        with open(os.path.join(config.logdir, f"prompt_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
            f.write(custom_prompt)

    # Query server with constructed prompt
    custom_kernel = inference_server(custom_prompt)
    custom_kernel = extract_first_code(custom_kernel, ["python", "cpp"])
    # check LLM is able to generate custom kernel code
    assert custom_kernel is not None, f"Custom {config.backend} kernel code generation failed"
    
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

    with app.run():
        kernel_exec_result = EvalFunc.with_options(gpu=config.gpu)().eval_single_sample_modal.remote(
            ref_arch_src, custom_kernel, config.verbose, gpu_arch_mapping[config.gpu], config.backend, config.precision, config.timing_method
        )
        
        print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")
        
        if config.log:
            with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "a") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(str(kernel_exec_result))

if __name__ == "__main__":
    main()