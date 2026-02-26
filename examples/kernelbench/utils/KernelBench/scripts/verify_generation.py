import sys, os
import kernelbench.utils as utils
import time
from kernelbench.prompt_constructor_toml import get_prompt_for_backend, get_custom_prompt
"""
For testing infernece and quickly iterate on prompts 
Uses functions in prompt_constructor
"""

def inference_with_prompt(arch_path, inference_server: callable = None, log_to_local: bool = False) -> str:
    """
    Returns the generated custom CUDA code (kernel to evaluate)

    if log_to_local, save the prompt and the generated code to ./scratch/
    """
    # read in an architecture file, copy it to ./scratch/model.py
    arch = utils.read_file(arch_path)

    if log_to_local:
        # Ensure the ./scratch directory exists
        os.makedirs("./scratch", exist_ok=True)

        # Write the architecture to ./scratch/model.py
        with open("./scratch/model.py", "w") as f:
            f.write(arch)

    custom_backend_prompt = get_prompt_for_backend(
        ref_arch_src=arch,
        backend="cuda",
        option="one_shot",
        precision="fp16",
        include_hardware=False,
        gpu_name="H100"
    )

    if log_to_local:    
        with open(f"./scratch/prompt.py", "w") as f:
            f.write(custom_backend_prompt)

    # query LLM
    custom_cuda = inference_server(custom_backend_prompt)

    custom_cuda = utils.extract_first_code(custom_cuda, ["python", "cpp"])
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"
    print(
        "[Verification] Torch module with Custom CUDA code **GENERATED** successfully"
    )

    if log_to_local:
        with open(f"./scratch/model_new.py", "w") as f:
            f.write(custom_cuda)

    return custom_cuda


def sanity_check_inference(inference_server: callable):
    """
    Simple fucntion to intiiate call to server just to check we can call API endpoint
    """
    start_time = time.time()
    lm_response = inference_server("What does CUDA stand for?")
    end_time = time.time()
    print(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
    print(lm_response) 
    return lm_response


if __name__ == "__main__":

    inference_provider_preset = "deepseek"
    inference_server = utils.create_inference_server_from_presets(server_type=inference_provider_preset,
                                                        greedy_sample=True,
                                                        verbose=True, 
                                                        time_generation=True)
    
    # sanity_check_inference(inference_server)

    if len(sys.argv) > 1:
        arch_path = sys.argv[1]
    else:
        # run from KernelBench top level directory
        # most basic problem
        arch_path = "./KernelBench/level1/1_Square_matrix_multiplication_.py"
        # representative of long problem, might require longer max tokens to not get cut of
        # arch_path = "./KernelBench/level3/45_UNetSoftmax.py" 
    
    inference_with_prompt(arch_path, inference_server, log_to_local=True)
