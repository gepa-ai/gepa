"""
Unit tests for precision downgrade detection in FP32 kernels.

Tests the check_precision_downgrade function to ensure it correctly detects
various patterns that indicate precision downgrading (FP32 -> FP16),
which is a reward hacking pattern.

Note: The current implementation only checks for FP32 -> FP16 downgrades,
not BF16 downgrades.

Run with pytest:
    pytest src/kernelbench/unit_tests/test_precision.py -v
    or
    uv run pytest src/kernelbench/unit_tests/test_precision.py -v
"""

import os
import sys
import pytest

# Add src directory to path for imports (consistent with other test files)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kernelbench.kernel_static_checker import check_precision_downgrade


# ============================================================================
# Test Cases for FP32 -> FP16 Precision Downgrades (PyTorch patterns)
# ============================================================================

def test_fp32_half_method():
    """Test detection of .half() method call downgrading FP32 to FP16."""
    code = """
    def forward(self, x):
        x = x.half()
        return x * 2
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect .half() method call"
    assert "FP16" in message


def test_fp32_float16_method():
    """Test detection of .float16() method call."""
    code = """
    x = input_tensor.float16()
    return x
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect .float16() method call"
    assert "FP16" in message


def test_fp32_to_torch_half():
    """Test detection of .to(torch.half) pattern."""
    code = """
    def forward(self, x):
        x = x.to(torch.half)
        return x
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect .to(torch.half)"
    assert "FP16" in message


def test_fp32_to_torch_float16():
    """Test detection of .to(torch.float16) pattern."""
    code = """
    x = x.to(torch.float16)
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect .to(torch.float16)"
    assert "FP16" in message


def test_fp32_to_dtype_float16():
    """Test detection of .to(dtype=torch.float16) pattern."""
    code = """
    def forward(self, x):
        return x.to(dtype=torch.float16)
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect .to(dtype=torch.float16)"
    assert "FP16" in message


def test_fp32_to_dtype_half():
    """Test detection of .to(dtype=torch.half) pattern."""
    code = """
    x.to(dtype=torch.half)
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect .to(dtype=torch.half)"
    assert "FP16" in message


# ============================================================================
# Test Cases for FP32 -> FP16 Precision Downgrades (CUDA patterns)
# ============================================================================

def test_fp32_cuda_float2half():
    """Test detection of CUDA __float2half intrinsic."""
    code = """
    __global__ void kernel(float* input, __half* output) {
        output[0] = __float2half(input[0]);
    }
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect CUDA __float2half"
    assert "FP16" in message


def test_fp32_cuda_float2half_rn():
    """Test detection of CUDA __float2half_rn intrinsic."""
    code = """
    __half result = __float2half_rn(value);
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect CUDA __float2half_rn"
    assert "FP16" in message


def test_fp32_cuda_cast_to_half():
    """Test detection of CUDA C-style cast to __half."""
    code = """
    __half h = (__half)float_value;
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect CUDA cast to __half"
    assert "FP16" in message


def test_fp32_cuda_static_cast_half():
    """Test detection of CUDA static_cast<half>."""
    code = """
    __half h = static_cast<__half>(float_value);
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect CUDA static_cast<__half>"
    assert "FP16" in message


# ============================================================================
# Test Cases for FP32 -> FP16 Precision Downgrades (Triton patterns)
# ============================================================================

def test_fp32_triton_astype_float16():
    """Test detection of Triton tl.astype(..., tl.float16)."""
    code = """
    @triton.jit
    def kernel(X, Y):
        x = tl.load(X)
        x_fp16 = tl.astype(x, tl.float16)
        tl.store(Y, x_fp16)
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect Triton tl.astype(..., tl.float16)"
    assert "FP16" in message


# ============================================================================
# Test Cases for FP32 -> FP16 Precision Downgrades (CUTLASS patterns)
# ============================================================================

def test_fp32_cutlass_numeric_converter():
    """Test detection of CUTLASS NumericConverter<half_t, float>."""
    code = """
    NumericConverter<half_t, float> converter;
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect CUTLASS NumericConverter<half_t, float>"
    assert "FP16" in message


def test_fp32_cutlass_linear_combination():
    """Test detection of CUTLASS LinearCombination<half_t, ...>."""
    code = """
    LinearCombination<half_t, 128> epilogue;
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect CUTLASS LinearCombination<half_t>"
    assert "FP16" in message


def test_fp32_cute_type_convert():
    """Test detection of CuTe type_convert<half_t>."""
    code = """
    auto result = type_convert<half_t>(input);
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect CuTe type_convert<half_t>"
    assert "FP16" in message


# ============================================================================
# Test Cases for FP32 -> FP16 Precision Downgrades (ThunderKittens patterns)
# ============================================================================

def test_fp32_thunderkittens_half():
    """Test detection of ThunderKittens tk::half()."""
    code = """
    using namespace tk;
    auto h = tk::half(value);
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect ThunderKittens tk::half()"
    assert "FP16" in message


# ============================================================================
# Test Cases for Valid FP32 Code (Should NOT be detected)
# ============================================================================

def test_fp32_legitimate_code_no_downgrade():
    """Test that legitimate FP32 code is not flagged."""
    code = """
    def forward(self, x):
        # Legitimate FP32 operations
        x = x * 2.0
        y = torch.matmul(x, x.t())
        return y.float()  # This is fine, float() returns float32
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is False, "Should not detect downgrade in legitimate FP32 code"
    assert message == ""


def test_fp32_with_comments_mentioning_half():
    """Test that comments mentioning half precision don't trigger false positives."""
    code = """
    def forward(self, x):
        # Note: This should use FP32, not FP16
        # Don't use .half() here!
        x = x * 2.0
        return x
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is False, "Comments should not trigger detection"


def test_fp32_string_literals():
    """Test that string literals mentioning precision don't trigger false positives."""
    code = """
    def forward(self, x):
        error_msg = "dtype=torch.float16 is not allowed"
        print(error_msg)
        return x
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is False, "String literals should not trigger detection"


def test_fp32_function_names_containing_half():
    """Test that function names containing 'half' don't trigger false positives."""
    code = """
    def compute_half_tensor(self, x):
        # Function name contains 'half' but doesn't downgrade precision
        return x * 0.5
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is False, "Function names should not trigger detection"


# ============================================================================
# Test Cases for Edge Cases
# ============================================================================

def test_fp32_case_insensitive_precision():
    """Test that precision parameter is case-insensitive."""
    code = """
    x = x.half()
    """
    detected1, _ = check_precision_downgrade(code, precision="FP32")
    detected2, _ = check_precision_downgrade(code, precision="fp32")
    detected3, _ = check_precision_downgrade(code, precision="Fp32")
    assert detected1 == detected2 == detected3 == True, "Should handle case-insensitive precision"


def test_fp32_alternative_precision_names():
    """Test that alternative precision names are normalized correctly."""
    code = """
    x = x.half()
    """
    detected1, _ = check_precision_downgrade(code, precision="float32")
    detected2, _ = check_precision_downgrade(code, precision="fp32")
    assert detected1 == detected2 == True, "Should normalize float32 to fp32"


def test_fp32_non_fp32_precision_skips_check():
    """Test that non-FP32 precision skips the check (implementation only checks FP32)."""
    code = """
    x = x.half()
    """
    detected, message = check_precision_downgrade(code, precision="fp16")
    assert detected is False, "Should skip check for non-FP32 precision"
    assert message == ""


def test_fp32_unknown_precision_skips_check():
    """Test that unknown precision skips the check."""
    code = """
    x = x.half()
    """
    detected, message = check_precision_downgrade(code, precision="int8")
    assert detected is False, "Should skip check for unknown precision"
    assert message == ""


def test_fp32_empty_code():
    """Test that empty code doesn't trigger detection."""
    code = ""
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is False, "Empty code should not trigger detection"


def test_fp32_whitespace_only():
    """Test that whitespace-only code doesn't trigger detection."""
    code = "   \n\n\t  \n  "
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is False, "Whitespace-only code should not trigger detection"


# ============================================================================
# Test Cases for Real-World Scenarios
# ============================================================================

def test_fp32_matmul_with_hidden_downgrade():
    """Test detection in a realistic matmul kernel that downgrades internally."""
    code = """
    @triton.jit
    def matmul_kernel(A, B, C, M, N, K):
        # Compute matrix multiplication
        a = tl.load(A)
        b = tl.load(B)
        
        # Sneaky precision downgrade
        a = tl.astype(a, tl.float16)
        b = tl.astype(b, tl.float16)
        
        c = tl.dot(a, b)
        tl.store(C, c)
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect hidden precision downgrade in matmul"
    assert "FP16" in message


def test_fp32_cuda_kernel_with_half():
    """Test detection in a realistic CUDA kernel."""
    code = """
    __global__ void add_kernel(float* a, float* b, float* c, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            __half a_half = __float2half(a[idx]);
            __half b_half = __float2half(b[idx]);
            c[idx] = __half2float(a_half + b_half);
        }
    }
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect CUDA kernel using __float2half"
    assert "FP16" in message


def test_fp32_complex_code_with_downgrade():
    """Test detection in complex code with multiple operations."""
    code = """
    def forward(self, x, y):
        # Some preprocessing
        x = x * 2.0
        y = y + 1.0
        
        # Main computation with downgrade
        x = x.to(dtype=torch.float16)
        result = torch.matmul(x, y)
        
        # Post-processing
        result = result * 3.0
        return result
    """
    detected, message = check_precision_downgrade(code, precision="fp32")
    assert detected is True, "Should detect downgrade in complex code"
    assert "FP16" in message


# ============================================================================
# Note on BF16 Tests
# ============================================================================
# The current implementation only checks for FP32 -> FP16 downgrades.
# BF16 downgrade detection is not yet implemented. These tests document
# expected behavior when BF16 support is added in the future.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
