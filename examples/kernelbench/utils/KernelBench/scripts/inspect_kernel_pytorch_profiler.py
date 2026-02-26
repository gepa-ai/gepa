import torch
from torch.profiler import profile, record_function, ProfilerActivity
import logging
import os
import io


"""
For analysis
Inspect the operator and kernel breakdown of model-generated kernel to a particular problem
Using PyTorch Profiler
"""

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

device = "cuda:0"


from kernelbench.utils import read_file
from kernelbench.eval import (
    load_custom_model,
    load_original_model_and_inputs,
    set_seed,
)


def get_torch_profiler_info(ref_arch_src: str, 
                            kernel_src: str, 
                            build_dir: str, 
                            device: torch.device, 
                            num_trials: int = 100,
                            table_row_limit: int = 10,
                            seed_num: int = 42)->str:
    """
    Get the profiler info for a particular kernel
    Given a KernelBench solution to a problem, we want to profile the kernel

    ref_arch_src: str, the source code of the reference architecture; we use this to get the inputs
    kernel_src: str, the source code of the kernel; this will be compiled and used to get the model
    build_dir: str, the directory to build the custom kernel
    device: torch.device, the device to run the profiler on
    num_trials: int, the number of trials to run for Torch profiling
    table_row_limit: int, the number of rows to display in the profiler table
    seed_num: int to initiliaze on device random seed


    Notes about profiling:
        - We do not set p.toggle_collection_dynamic explicitly, 
        - We only collect CUDA activity (ProfilerActivity.CUDA), as we are only interested in the kernel
        
    """

    assert torch.cuda.is_available(), "CUDA is not available, cannot run Torch Profiler"

    context = {}
    _, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )

    set_seed(seed_num)
    inputs = get_inputs()
    init_inputs = get_init_inputs()
    inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x
        for x in inputs
    ]
    init_inputs = [
        x.cuda(device=device) if isinstance(x, torch.Tensor) else x
        for x in init_inputs
    ]
    
    ModelNew = load_custom_model(kernel_src, context, build_dir)
    # construct the new model with init inputs
    model = ModelNew(*init_inputs)
    assert hasattr(model, "forward")
    torch.cuda.synchronize(device=device)

    model = model.cuda(device=device)


    with torch.no_grad():
        profiling_scheduler = torch.profiler.schedule(
            skip_first=2,
            wait=2,
            warmup=3,
            active=num_trials,
        )

        with profile(
            activities=[ProfilerActivity.CUDA],
            schedule=profiling_scheduler,
        ) as prof:
            for _ in range(num_trials):
            
                output = model(*inputs)
                prof.step()

        profiler_output = prof.key_averages().table(sort_by='cuda_time_total', 
                                                    row_limit=table_row_limit)
        
    return profiler_output
    
def __main__():
    # run_profile(dataset, problem_id, num_trials=10)

    ref_arch_src_path = os.path.join(REPO_ROOT, "src/kernelbench/prompts/few_shot/model_ex_mnist2.py")
    kernel_src_path = os.path.join(REPO_ROOT, "src/kernelbench/prompts/few_shot/model_new_ex_mnist2.py")

    ref_arch_src = read_file(ref_arch_src_path)
    kernel_src = read_file(kernel_src_path)

    profile_result = get_torch_profiler_info(
        ref_arch_src,
        kernel_src,
        build_dir="build",
        device="cuda:0",
        num_trials=20,
        seed_num=42,
        table_row_limit=10
    )
    
    print(profile_result)
    print(f"Profiler result could be parsed as a string of length {len(profile_result)}")

if __name__ == "__main__":
    __main__()