################################################################################
# Helpers for Analysis
################################################################################
import numpy as np

from functools import cache
from transformers import AutoTokenizer
import utils
import re


def pass_at_k(n, c, k):
    """
    A numerically stable script for calculating an unbiased estimate of pass@k
    Referenced from HumanEval: https://arxiv.org/abs/2107.03374
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def get_token_count(text: str, tokenizer: AutoTokenizer) -> int:
    assert isinstance(text, str), "can only tokenize strings but got {}".format(type(text))
    return len(tokenizer.encode(text))


def extract_all_cuda_sources(file_content: str) -> list[str]:
    """
    Extract all CUDA sources wrapped in triple quotes.
    
    Returns:
        list[str]: List of all extracted CUDA source code blocks
    """
    pattern = r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*"""(.*?)"""'
    matches = re.findall(pattern, file_content, re.DOTALL)
    return [match.strip() for match in matches]


def get_cuda_tokens(kernel_src: str, tokenizer: AutoTokenizer) -> int:
    """
    Count number of all CUDA tokens in the kernel
    """
    all_cuda_code = extract_all_cuda_sources(kernel_src)
    num_cuda_tokens = sum(get_token_count(code, tokenizer) for code in all_cuda_code)
    return num_cuda_tokens

