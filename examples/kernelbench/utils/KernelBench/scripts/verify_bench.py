"""
This script tests the correctness of models in KernelBench by generating random inputs 
and random initialization. It compares the output of the original model against itself.
It ensures that the test is well-formed and there are no sources of non-determinism in the test.

Usage: python test_bench.py
"""

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import os
import importlib.util

"""
Test all the reference architectures compiles 
and reproduce the same results when run against itself
"""

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def check_correctness(
    Model, NewModel, get_inputs, get_init_inputs, seed=1012, atol=1e-02, rtol=1e-02
):
    # run the model and check correctness
    with torch.no_grad():
        set_seed(seed)
        inputs = get_inputs()
        inputs = [x.cuda() if isinstance(x, torch.Tensor) else x for x in inputs]

        set_seed(seed)
        init_inputs = get_init_inputs()
        init_inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x for x in init_inputs
        ]

        set_seed(seed)
        model = Model(*init_inputs).cuda()

        set_seed(seed)
        model_new = NewModel(*init_inputs).cuda()

        output = model(*inputs)
        output_new = model_new(*inputs)

        if output.shape != output_new.shape:
            return False
        if not torch.allclose(output, output_new, atol=atol, rtol=rtol):
            return False
    return True


def run(Model, NewModel, get_inputs, get_init_inputs, seed=1012):
    return check_correctness(Model, NewModel, get_inputs, get_init_inputs, seed)


from kernelbench.dataset import construct_kernelbench_dataset

def run_all(level):
    print(f"Running Level {level}")
    dataset = construct_kernelbench_dataset(level)
    total = 0
    passed = 0
    fail_tests = []
    
    for problem in dataset:
        total += 1
        module_name = problem.name.replace(".py", "")
        try:
            problem_path = getattr(problem, "path", None)
            if not problem_path:
                raise ValueError(
                    f"Problem '{module_name}' does not have a local file path; "
                    "verify_bench.py only supports local datasets."
                )
            # Dynamically import the module
            spec = importlib.util.spec_from_file_location(
                module_name, problem_path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            # Get the required attributes from the module
            Model = getattr(module, "Model")
            get_inputs = getattr(module, "get_inputs")
            get_init_inputs = getattr(module, "get_init_inputs")
            assert run(Model, Model, get_inputs, get_init_inputs)
            passed += 1
        except Exception as e:
            print(f"Failed {module_name}: {e}")
            fail_tests.append(module_name)
    print(f"Level {level}: {passed}/{total} passed")
    if len(fail_tests) > 0:
        print(f"Failed tests: {fail_tests}")


if __name__ == "__main__":
    run_all(1)
    run_all(2)
    run_all(3)
