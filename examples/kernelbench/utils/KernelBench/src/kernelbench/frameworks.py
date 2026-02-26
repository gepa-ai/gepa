########################
# Frameworks
# Support Test-Time Frameworks for LLM Inference
# instead of single model calls
# See how we added Archon support as an example
########################

import multiprocessing
import subprocess
import re
import random
import tempfile
from pathlib import Path
import re
import math
import os
import json
from tqdm import tqdm

# API clients
from archon.completions import Archon

# python-dotenv reads key-value pairs from a .env file and sets them as environment variables
from dotenv import load_dotenv
load_dotenv()

import numpy as np
from contextlib import contextmanager
from collections import defaultdict
import time
import shutil
import concurrent
from functools import cache
import hashlib

from concurrent.futures import ProcessPoolExecutor, as_completed

# Define API key access
TOGETHER_KEY = os.environ.get("TOGETHER_API_KEY")
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
SGLANG_KEY = os.environ.get("SGLANG_API_KEY")  # for Local Deployment
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
SAMBANOVA_API_KEY = os.environ.get("SAMBANOVA_API_KEY")



########################################################
# Inference Time Frameworks
########################################################

def query_framework_server(
    prompt: str | list[dict],  # string if normal prompt, list of dicts if chat prompt,
    system_prompt: str = "You are a helpful assistant",  # only used for chat prompts
    temperature: float = 0.0,
    top_p: float = 1.0, # nucleus sampling
    top_k: int = 50, 
    max_tokens: int = 128,  # max output tokens to generate
    num_completions: int = 1,
    server_port: int = 30000,  # only for local server hosted on SGLang
    server_address: str = "localhost",
    server_type: str = "sglang",
    model_name: str = "default",  # specify model type
    framework_config_path: str = None,
):
    """
    Query various sort of LLM inference Frameworks
    - Archon: https://arxiv.org/abs/2409.15254
    - Might add future supports for more test-time frameworks
    """
    # Select model and client based on arguments
    match server_type:
        case "archon":
            archon_config_path = framework_config_path
            assert archon_config_path is not None, "Archon config path is required"
            assert os.path.exists(archon_config_path), f"Archon config path {archon_config_path} does not exist"
            client = Archon(json.load(open(archon_config_path)))
            model = model_name
            print(f"Querying Archon model {model} with config {archon_config_path}")
            response = client.generate(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            outputs = response
            if len(outputs) == 1:
                return outputs[0]
            else:
                return outputs
        case _:
            raise NotImplementedError

# a list of presets for API server configs
SERVER_PRESETS = {
    "archon": {
        "archon_config_path": "archon_configs/gpt-4-turbo.json",
        "model_name": "individual_gpt-4-turbo",
    },
}

def create_inference_framework_server_from_presets(framework_type: str = None, 
                                         greedy_sample: bool = False,   
                                         verbose: bool = False,
                                         time_generation: bool = False,
                                         **kwargs,
                                         ) -> callable:
    """
    Return a callable function that queries LLM with given settings
    """
    def _query_llm(prompt: str | list[dict]):
        server_args = SERVER_PRESETS[framework_type].copy()

        if kwargs:
            server_args.update(kwargs)
        if greedy_sample:
            server_args["temperature"] = 0.0
            server_args["top_p"] = 1.0
            server_args["top_k"] = 1
        if verbose:
            print(f"Querying server {framework_type} with args: {server_args}")
        
        if time_generation:
            start_time = time.time()
            response = query_framework_server(
                prompt, framework_type=framework_type, **server_args
            )
            end_time = time.time()
            print(f"[Timing] Inference took {end_time - start_time:.2f} seconds")
            return response
        else:
            return query_framework_server(
                prompt, framework_type=framework_type, **server_args
            )
    
    return _query_llm