import shutil
import torch
import pydra
from pydra import REQUIRED, Config
import os
import modal

from kernelbench import eval as kernel_eval
from kernelbench import utils as kernel_utils
from kernelbench.timing import measure_ref_program_time
from kernelbench.utils import read_file
from kernelbench.kernel_static_checker import validate_kernel_static

# Modal setup
app = modal.App("run_and_check")
gpu_arch_mapping = {
    "L40S": ["Ada"],
    "H100": ["Hopper"],
    "H200": ["Hopper"],
    "A100": ["Ampere"],
    "A100-80GB": ["Ampere"],
    "L4": ["Ada"],
    "T4": ["Turing"],
    "A10G": ["Ampere"]
}

REPO_TOP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

cuda_version = "13.0.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

SRC_DIR = os.path.join(REPO_TOP_PATH, "src")
SCRIPTS_DIR = os.path.join(REPO_TOP_PATH, "scripts")
KERNELBENCH_DIR = os.path.join(REPO_TOP_PATH, "KernelBench")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install("git", "gcc-10", "g++-10", "clang")
    .uv_sync(uv_project_dir=REPO_TOP_PATH, extras=["gpu"])
    .run_commands("git clone -b main https://github.com/HazyResearch/ThunderKittens.git /root/ThunderKittens")
    .env({
        "THUNDERKITTENS_ROOT": "/root/ThunderKittens",
        "PYTHONPATH": "/root:/root/src:/root/scripts"
    })
    .add_local_dir(SRC_DIR, remote_path="/root/src")
    .add_local_dir(SCRIPTS_DIR, remote_path="/root/scripts")
    .add_local_dir(KERNELBENCH_DIR, remote_path="/root/KernelBench")  # must be last
)

"""
Run a pair of KernelBench format (problem, solution) to check if solution is correct and compute speedup

You will need two files
1. Reference: PyTorch reference (module Model) implementation with init and input shapes
2. Solution: PyTorch solution (module ModelNew) with inline CUDA Code
Please see examples in src/kernelbench/prompts

The Reference could be either
1. a local file: specify the path to the file
2. a kernelbench problem: specify level and problem id

====================================================
Usage:
1. PyTorch reference is a local file (local eval)
python3 scripts/run_and_check.py ref_origin=local ref_arch_src_path=src/kernelbench/prompts/model_ex_add.py kernel_src_path=src/kernelbench/prompts/model_new_ex_add.py eval_mode=local
python3 scripts/run_and_check.py ref_origin=local ref_arch_src_path=src/kernelbench/prompts/few_shot/model_ex_tiled_matmul.py kernel_src_path=src/kernelbench/prompts/few_shot/model_new_ex_tiled_matmul.py eval_mode=local


2. PyTorch reference is a kernelbench problem (local eval)
python3 scripts/run_and_check.py ref_origin=kernelbench level=<level> problem_id=<problem_id> kernel_src_path=<path to model-generated kernel> eval_mode=local

3. PyTorch reference is a local file (modal eval on cloud GPU)
python3 scripts/run_and_check.py ref_origin=local ref_arch_src_path=src/kernelbench/prompts/model_ex_add.py kernel_src_path=src/kernelbench/prompts/model_new_ex_add.py eval_mode=modal gpu=H100

4. PyTorch reference is a kernelbench problem (modal eval on cloud GPU)
python3 scripts/run_and_check.py ref_origin=kernelbench level=<level> problem_id=<problem_id> kernel_src_path=<path to model-generated kernel> eval_mode=modal gpu=L40S
====================================================

"""

torch.set_printoptions(precision=4, threshold=10)

class ScriptConfig(Config):
    def __init__(self):

        # Problem and Solution definition
        # Input src origin definition
        self.ref_origin = REQUIRED # either local or kernelbench
        # ref_origin is local, specify local file path
        self.ref_arch_src_path = ""
        # ref_origin is kernelbench, specify level and problem id
        self.dataset_src = "huggingface" # either huggingface or local
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = ""
        self.problem_id = ""
        # Solution src definition
        self.kernel_src_path = ""

        # Evaluation mode
        self.eval_mode = "local"  # either "local" or "modal"
        self.gpu = "L40S"  # GPU type for modal (L40S, H100, H200, A100, etc.)

        # KernelBench Eval specific
        # number of trials to run for correctness
        self.num_correct_trials = 5
        # number of trials to run for performance
        self.num_perf_trials = 100
        # timeout for each trial
        self.timeout = 300
        # verbose logging
        self.verbose = False
        self.measure_performance = True
        self.timing_method = "cuda_event"  # see timing.py
        self.build_dir_prefix = "" # if you want to specify a custom build directory
        self.clear_cache = False # TODO

        # Replace with your NVIDIA GPU architecture, e.g. ["Hopper"]
        self.gpu_arch = ["Ada"]
        self.precision = "fp32"
        self.backend = "cuda"

        self.check_kernel = True  # [experimental] optional static checker catching potential hacking patterns

    def __repr__(self):
        return f"ScriptConfig({self.to_dict()})"

def evaluate_single_sample_src(ref_arch_src: str, kernel_src: str, configs: dict, device: torch.device) -> kernel_eval.KernelExecResult:
    """
    Evaluate a single sample source code against a reference source code
    """

    kernel_hash = str(hash(kernel_src))
    build_dir = os.path.join(configs["build_dir_prefix"], "test_build", kernel_hash)
    
    if configs["clear_cache"]: # fresh kernel build
        print(f"[INFO] Clearing cache for build directory: {build_dir}")
        shutil.rmtree(build_dir, ignore_errors=True)
    
    num_correct_trials = configs["num_correct_trials"]
    num_perf_trials = configs["num_perf_trials"]    
    verbose = configs["verbose"]
    measure_performance = configs["measure_performance"]
    timing_method = configs["timing_method"]
    backend = configs["backend"]
    precision = kernel_eval.get_torch_dtype_from_string(configs["precision"])
    
    try:
        eval_result = kernel_eval.eval_kernel_against_ref(
        original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=measure_performance,
            timing_method=timing_method,
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=build_dir,
            device=device,
            backend=backend,
            precision=precision
        )
        return eval_result
    except Exception as e:
        print(f"[WARNING] Last level catch: Some issue evaluating for kernel: {e} ")
        if "CUDA error" in str(e): 
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {"cuda_error": f"CUDA Error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        }
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result
        else:
            metadata = {"other_error": f"error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        }
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False,
                                                metadata=metadata)
            return eval_result


# Modal evaluation class
@app.cls(image=image, scaledown_window=5)
class EvalFunc:

    @modal.method()
    def evaluate_single_sample_src_modal(self, ref_arch_src: str, kernel_src: str, configs: dict, gpu_arch: list):
        """Evaluate a single sample source code against a reference source code on Modal"""
        from kernelbench.utils import set_gpu_arch
        from kernelbench.eval import eval_kernel_against_ref, get_torch_dtype_from_string

        set_gpu_arch(gpu_arch)
        device = torch.device("cuda:0")

        num_correct_trials = configs["num_correct_trials"]
        num_perf_trials = configs["num_perf_trials"]
        verbose = configs["verbose"]
        measure_performance = configs["measure_performance"]
        timing_method = configs["timing_method"]
        backend = configs["backend"]
        precision = kernel_eval.get_torch_dtype_from_string(configs["precision"])

        eval_result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=measure_performance,
            timing_method=timing_method,
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            device=device,
            backend=backend,
            precision=precision
        )
        return eval_result

    @modal.method()
    def measure_program_time_modal(
        self,
        ref_arch_src: str,
        num_trials: int,
        use_torch_compile: bool,
        torch_compile_backend: str,
        torch_compile_options: str,
        gpu_arch: list,
        precision: str,
    ):
        """Measure the execution time of a reference program on Modal"""
        from kernelbench.timing import measure_ref_program_time
        from kernelbench.utils import set_gpu_arch

        set_gpu_arch(gpu_arch)
        device = torch.device("cuda:0")

        return measure_ref_program_time(
            ref_arch_name="Reference Program",
            ref_arch_src=ref_arch_src,
            num_trials=num_trials,
            use_torch_compile=use_torch_compile,
            torch_compile_backend=torch_compile_backend,
            torch_compile_options=torch_compile_options,
            verbose=False,
            device=device,
            precision=precision,
        )


@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):

    print("Running with config", config)

    # Fetch reference and kernel code

    assert config.ref_origin == "local" or config.ref_origin == "kernelbench", "ref_origin must be either local or kernelbench"
    assert config.kernel_src_path != "", "kernel_src_path is required"  
    
    if config.ref_origin == "local":
        assert config.ref_arch_src_path != "", "ref_arch_src_path is required"
        ref_arch_src = read_file(config.ref_arch_src_path)
        print(f"Loaded reference from local file: {config.ref_arch_src_path}")
    elif config.ref_origin == "kernelbench":
        from kernelbench.dataset import construct_kernelbench_dataset
        
        assert config.level != "", "level is required"
        assert config.problem_id != "", "problem_id is required"
        
        # Unified interface - same code for huggingface and local!
        dataset = construct_kernelbench_dataset(
            level=int(config.level),
            source=config.dataset_src,
            dataset_name=config.dataset_name,
        )
        problem = dataset.get_problem_by_id(int(config.problem_id))
        ref_arch_src = problem.code
        
        print(f"Fetched problem {problem.problem_id} from KernelBench level {problem.level}: {problem.name}")
    
    kernel_src = read_file(config.kernel_src_path)

    # Optional: static code checker for kernel code using regex matching
    # NOTE: by no means is this checker complete, but it could help catch some potential hacks
    if config.check_kernel:
        static_check_status, errors, warnings = validate_kernel_static(
            kernel_src,
            backend=config.backend,
            precision=config.precision,
        )
        assert static_check_status, f"Static check failed. Errors: {errors}. Warnings: {warnings}"
        if warnings:
            print(f"[WARN] Static check warnings: {warnings}")

    # Start Evaluation
    assert config.eval_mode in ["local", "modal"], "eval_mode must be either 'local' or 'modal'"

    if config.eval_mode == "local":
        # Local evaluation (existing code path)
        device = torch.device("cuda:0")
        kernel_utils.set_gpu_arch(config.gpu_arch)

        print("[INFO] Evaluating kernel against reference code (LOCAL)")
        # Evaluate kernel against reference code
        kernel_eval_result = evaluate_single_sample_src(
            ref_arch_src=ref_arch_src,
            kernel_src=kernel_src,
            configs=config.to_dict(),
            device=device
        )
        kernel_exec_time = kernel_eval_result.runtime

        # Measure baseline time
        print("[INFO] Measuring reference program time")
        # Default using PyTorch Eager here
        ref_time_eager_result = measure_ref_program_time(ref_arch_name="Reference Program",
                                                    ref_arch_src=ref_arch_src,
                                                    num_trials=config.num_perf_trials,
                                                    use_torch_compile=False,
                                                    timing_method=config.timing_method,
                                                    device=device,
                                                    verbose=False,
                                                    precision=config.precision,
                                                    )
        ref_exec_eager_time = ref_time_eager_result.get("mean", None)

        # Measure Torch Compile time
        ref_time_compile_result = measure_ref_program_time(ref_arch_name="Reference Program",
                                                    ref_arch_src=ref_arch_src,
                                                    num_trials=config.num_perf_trials,
                                                    use_torch_compile=True,
                                                    timing_method=config.timing_method,
                                                    device=device,
                                                    verbose=False,
                                                    precision=config.precision,
                                                    )
        ref_exec_compile_time = ref_time_compile_result.get("mean", None)

    elif config.eval_mode == "modal":
        # Modal evaluation (remote execution)
        gpu_arch = gpu_arch_mapping.get(config.gpu, config.gpu_arch)
        print(f"[INFO] Using GPU: {config.gpu} with architecture: {gpu_arch}")

        with app.run():
            print("[INFO] Evaluating kernel against reference code (MODAL)")
            # Evaluate kernel against reference code
            kernel_eval_result = EvalFunc.with_options(
                gpu=config.gpu
            )().evaluate_single_sample_src_modal.remote(
                ref_arch_src=ref_arch_src,
                kernel_src=kernel_src,
                configs=config.to_dict(),
                gpu_arch=gpu_arch
            )
            kernel_exec_time = kernel_eval_result.runtime

            # Measure baseline time
            print("[INFO] Measuring reference program time (PyTorch Eager)")
            ref_time_eager_result = EvalFunc.with_options(
                gpu=config.gpu
            )().measure_program_time_modal.remote(
                ref_arch_src=ref_arch_src,
                num_trials=config.num_perf_trials,
                use_torch_compile=False,
                torch_compile_backend=None,
                torch_compile_options=None,
                gpu_arch=gpu_arch,
                precision=config.precision,
            )
            ref_exec_eager_time = ref_time_eager_result.get("mean", None)

            # Measure Torch Compile time
            print("[INFO] Measuring reference program time (torch.compile)")
            ref_time_compile_result = EvalFunc.with_options(
                gpu=config.gpu
            )().measure_program_time_modal.remote(
                ref_arch_src=ref_arch_src,
                num_trials=config.num_perf_trials,
                use_torch_compile=True,
                torch_compile_backend="inductor",
                torch_compile_options="default",
                gpu_arch=gpu_arch,
                precision=config.precision,
            )
            ref_exec_compile_time = ref_time_compile_result.get("mean", None)

    print("="*40)
    print(f"[Eval] Kernel eval result: {kernel_eval_result}")
    print("-"*40)
    print(f"[Timing] PyTorch Reference Eager exec time: {ref_exec_eager_time} ms")
    print(f"[Timing] PyTorch Reference torch.compile time: {ref_exec_compile_time} ms")
    print(f"[Timing] Custom Kernel exec time: {kernel_exec_time} ms")
    print("-"*40)   
    
    if kernel_eval_result.correctness:
        print(f"[Speedup] Speedup over eager: {ref_exec_eager_time / kernel_exec_time:.2f}x")
        print(f"[Speedup] Speedup over torch.compile: {ref_exec_compile_time / kernel_exec_time:.2f}x")
    else:
        print("[Speedup] Speedup Not Available as Kernel did not pass correctness")

    print("="*40)


if __name__ == "__main__":
    main()