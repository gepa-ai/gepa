from dataclasses import dataclass
import random
import time

from tqdm import tqdm

import shutil
from kernelbench.eval import build_compile_cache
from kernelbench import utils as utils
import torch
import os
import multiprocessing as mp

"""
Compile and Cache

This module contains the logic for compiling and caching the kernels
on CPU in parallel so you can speedup the evaluation process

The cache build directory must match the ones you use during evaluation phase
""" 

@dataclass
class WorkArgs:
    problem_id: int
    sample_id: int
    device: torch.device

def compile_single_sample(work_args: WorkArgs, config: dict) -> tuple[bool, str]:

    problem_id = work_args.problem_id 
    sample_id = work_args.sample_id
    verbose = config["verbose"]
    
    utils.set_gpu_arch(config["gpu_arch"])

    build_dir = os.path.join(config["kernel_eval_build_dir"], config["run_name"], str(problem_id), str(sample_id))

    run_dir = os.path.join(config["runs_dir"], config["run_name"])
    kernel_src_path = os.path.join(run_dir, f"level_{config['level']}_problem_{problem_id}_sample_{sample_id}_kernel.py")

    if not os.path.exists(kernel_src_path):
        print(f"[ERROR] Kernel source file not found for Problem ID: {problem_id}, Sample ID: {sample_id}")
        return False, "Kernel source file not found"

    with open(kernel_src_path, "r") as f:
        kernel_src = f.read()

    try:
        compiled_and_cached, stdout_content, error_msg = build_compile_cache(custom_model_src=kernel_src,
                                                       verbose=verbose, 
                                                       build_dir=build_dir)

        return compiled_and_cached, stdout_content, error_msg
    except Exception as e:
        print(f"[WARNING] Last level catch on {sample_id}: Some issue while compiling and attempting to cache for kernel: {e} ")
        return None, str(e), str(e)
    
def remove_cache_dir(config, problem_id, sample_id):
    """
    Remove the cached folder for sample compilation so it can start a clean build next time
    useful for time out, failed build, etc.
    """
    cache_dir = os.path.join(config['kernel_eval_build_dir'], config["run_name"], f"{problem_id}", f"{sample_id}")
    print(f"cache_dir to remove: {cache_dir}")
    if os.path.exists(cache_dir):
        try:
            # Add error handling and retry with force
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"\n[INFO] Removed cached folder for Problem ID: {problem_id}, Sample ID: {sample_id}")
        except Exception as e:
            print(f"\n[WARNING] Failed to remove cache directory {cache_dir}: {str(e)}")

def batch_compile(total_work: list[tuple[int, int]], config: dict):
    """
    Batch compile cache across CPUs, assume config has num_cpu_workers
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    assert "num_cpu_workers" in config, "num_cpu_workers must be specified in config for batch compile"
    try:
        with mp.Pool(config["num_cpu_workers"]) as pool:
            # Create work args for each task
            work_args = [
                (WorkArgs(problem_id=p_id, sample_id=s_idx, device=None), config)
                for p_id, s_idx in total_work
            ]


            # Launch all tasks in parallel and track start times
            async_results = []
            start_times = {}
            for i, work_arg in enumerate(work_args):
                # this is type AsyncResult
                async_result = pool.apply_async(compile_single_sample, args=work_arg)
                async_results.append(async_result)
                start_times[id(async_result)] = time.time()
            
            results = []
            pending_tasks = list(enumerate(async_results))
            
            with tqdm(total=len(work_args), desc="Compile & Cache Progress") as pbar:
                while pending_tasks:
                    remaining_tasks = []
                    for i, async_result in pending_tasks:
                        try:
                            problem_id, sample_id = total_work[i] # curr code of interest
                            if async_result.ready():
                                try:
                                    compiled, stdout_content, error_msg = async_result.get(timeout=1)  # Short timeout for completed tasks
                                    
                                    print(f"[Status] Compilation {compiled} for problem {problem_id} sample {sample_id}")
                                    results.append((i, compiled))

                                    if not compiled:
                                        # Remove the cached folder for this timed out sample so it can start a clean build next time                                        problem_id, sample_id = total_work[i]
                                        remove_cache_dir(config, problem_id, sample_id)
                                        
                                    pbar.update(1)
                                except Exception as e:
                                    problem_id, sample_id = total_work[i]
                                    with open("error_log.txt", "a") as f:
                                        f.write(f"\n[ERROR] Task failed for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}")
                                    print(f"\n[ERROR] Task failed for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}")
                                    remove_cache_dir(config, problem_id, sample_id)
                                    results.append((i, None))
                                    pbar.update(1)
                            else:
                                # Check if the task has exceeded timeout
                                if time.time() - start_times[id(async_result)] > config["timeout"]:
                                    problem_id, sample_id = total_work[i]
                                    print(f"\n[TIME OUT] Task timed out for Problem ID: {problem_id}, Sample ID: {sample_id}")
                                    
                                    problem_id, sample_id = total_work[i]
                                    remove_cache_dir(config, problem_id, sample_id)

                                    # if we were to retry!
                                    # Start a new task for the same work
                                    print(f"Retrying for Problem ID: {problem_id}, Sample ID: {sample_id}")
                                    new_async_result = pool.apply_async(compile_single_sample, args=work_args[i])
                                    start_times[id(new_async_result)] = time.time()
                                    remaining_tasks.append((i, new_async_result))
                                else:
                                    # keep going 
                                    remaining_tasks.append((i, async_result))

                        except Exception as e:
                            problem_id, sample_id = total_work[i]
                            print(f"\n[ERROR] Unexpected error for Problem ID: {problem_id}, Sample ID: {sample_id}: {str(e)}")
                            
                            remove_cache_dir(config, problem_id, sample_id)
                            
                            results.append((i, None))
                            
                            pbar.update(1)
                    
                    pending_tasks = remaining_tasks
                    time.sleep(0.1)  # Prevent busy waiting
            
            # Sort results back to original order
            sorted_results = [r for _, r in sorted(results, key=lambda x: x[0])]
            return sorted_results

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Terminating workers...")
        pool.terminate()
        raise
    finally:
        if 'pool' in locals():
            pool.close()
