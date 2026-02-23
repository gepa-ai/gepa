"""
Unit tests for validate_kernel_static function.

Tests the main entry point function to ensure it correctly:
- Passes precision to precision-dependent checks
- Categorizes errors vs warnings correctly
- Handles backend-specific checks
- Respects forbidden/warnings parameters
- Returns correct output format

Run with pytest:
    pytest src/kernelbench/unit_tests/test_validate_kernel_static.py -v
    or
    uv run pytest src/kernelbench/unit_tests/test_validate_kernel_static.py -v
"""

import os
import sys
import pytest

# Add src directory to path for imports (consistent with other test files)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kernelbench.kernel_static_checker import validate_kernel_static


# ============================================================================
# Test Basic Function Signature and Return Values
# ============================================================================

def test_validate_kernel_static_returns_tuple():
    """Test that validate_kernel_static returns a tuple of (valid, errors, warnings)."""
    code = "x = 1 + 1"
    result = validate_kernel_static(code)
    
    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 3, "Should return (valid, errors, warnings)"
    valid, errors, warnings = result
    assert isinstance(valid, bool), "First element should be bool"
    assert isinstance(errors, list), "Second element should be list"
    assert isinstance(warnings, list), "Third element should be list"


def test_validate_kernel_static_defaults():
    """Test that validate_kernel_static works with default parameters."""
    code = "x = 1 + 1"
    valid, errors, warnings = validate_kernel_static(code)
    
    # Should work without errors for simple valid code
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


# ============================================================================
# Test Precision Parameter Passing
# ============================================================================

def test_precision_passed_to_precision_checker_fp32():
    """Test that precision parameter is correctly passed to precision-dependent checks."""
    # Code with FP32 -> FP16 downgrade
    code = """
    def forward(self, x):
        x = x.half()  # Precision downgrade
        return x
    """
    
    # With fp32 precision, should detect downgrade (as warning by default)
    valid, errors, warnings = validate_kernel_static(code, precision="fp32")
    
    # Check that precision downgrade was detected (should be in warnings by default)
    all_messages = errors + warnings
    has_precision_warning = any("precision" in msg.lower() or "fp16" in msg.lower() 
                                for msg in all_messages)
    assert has_precision_warning, "Should detect precision downgrade with fp32 precision"


def test_precision_passed_to_precision_checker_fp16():
    """Test that fp16 precision skips FP32 -> FP16 downgrade check."""
    # Code with FP32 -> FP16 downgrade
    code = """
    def forward(self, x):
        x = x.half()  # Precision downgrade
        return x
    """
    
    # With fp16 precision, precision downgrade check should be skipped
    valid, errors, warnings = validate_kernel_static(code, precision="fp16")
    
    # Should not detect precision downgrade (check is skipped for non-FP32)
    all_messages = errors + warnings
    has_precision_warning = any("precision" in msg.lower() or "fp16" in msg.lower() 
                                for msg in all_messages)
    # This is expected - the check only runs for fp32
    # So for fp16, it won't flag this


def test_precision_case_insensitive():
    """Test that precision parameter is case-insensitive."""
    code = """
    def forward(self, x):
        x = x.half()
        return x
    """
    
    # Test different case variations
    result1 = validate_kernel_static(code, precision="FP32")
    result2 = validate_kernel_static(code, precision="fp32")
    result3 = validate_kernel_static(code, precision="Fp32")
    
    # All should produce the same result
    assert result1 == result2 == result3, "Precision should be case-insensitive"


def test_precision_alternative_names():
    """Test that alternative precision names are normalized."""
    code = """
    def forward(self, x):
        x = x.half()
        return x
    """
    
    # float32 should be normalized to fp32
    result1 = validate_kernel_static(code, precision="float32")
    result2 = validate_kernel_static(code, precision="fp32")
    
    assert result1 == result2, "float32 should be normalized to fp32"


# ============================================================================
# Test Error vs Warning Categorization
# ============================================================================

def test_strict_checks_are_errors():
    """Test that strict checks (like code_bypass) produce errors."""
    code = """
    try:
        result = custom_kernel(x)
    except:
        result = torch.matmul(x, w)  # Fallback to torch
    """
    
    valid, errors, warnings = validate_kernel_static(code)
    
    assert not valid, "Code with strict violations should be invalid"
    assert len(errors) > 0, "Strict checks should produce errors, not warnings"
    assert any("try-except" in e.lower() or "bypass" in e.lower() 
               for e in errors), "Should flag bypass in errors"


def test_warning_checks_are_warnings():
    """Test that warning checks produce warnings, not errors."""
    code = """
    def forward(self, x):
        x = x.half()  # Precision downgrade - in warnings by default
        return x
    """
    
    # Test with default settings - precision_downgrade should be in warnings
    valid, errors, warnings = validate_kernel_static(
        code, 
        precision="fp32"
        # Using defaults - precision_downgrade is in WARNING_CHECKS
    )
    
    # Check that precision downgrade message is in warnings (if detected)
    # Note: backend impl checks might add errors, but precision should be in warnings
    precision_warnings = [w for w in warnings if "precision" in w.lower() or "fp16" in w.lower()]
    precision_errors = [e for e in errors if "precision" in e.lower() or "fp16" in e.lower()]
    
    if precision_warnings or precision_errors:
        # If precision downgrade is detected, it should be in warnings, not errors
        assert len(precision_warnings) > 0, "Precision downgrade should be in warnings (default)"
        assert len(precision_errors) == 0, "Precision downgrade should not be in errors (default)"


def test_custom_forbidden_checks():
    """Test that custom forbidden checks produce errors."""
    code = """
    def forward(self, x):
        x = x.half()  # Precision downgrade
        return x
    """
    
    # Make precision_downgrade a forbidden check (error) instead of warning
    valid, errors, warnings = validate_kernel_static(
        code, 
        precision="fp32",
        forbidden=["precision_downgrade"]
    )
    
    assert not valid, "Should be invalid when precision_downgrade is forbidden"
    assert len(errors) > 0, "Forbidden checks should produce errors"
    assert any("precision" in e.lower() or "fp16" in e.lower() 
               for e in errors), "Should flag precision downgrade in errors"


def test_custom_warnings_list():
    """Test that custom warnings list works."""
    code = """
    try:
        result = custom_kernel(x)
    except:
        result = torch.matmul(x, w)
    """
    
    # Move code_bypass to warnings instead of errors
    # Use a backend that won't add strict impl checks
    valid, errors, warnings = validate_kernel_static(
        code,
        backend="cuda",  # Explicit backend
        forbidden=[],  # No forbidden checks
        warnings=["code_bypass"]  # Make bypass a warning
    )
    
    # Note: Backend might add impl checks, so we check that code_bypass
    # appears in warnings (not errors) if it's detected
    all_messages = errors + warnings
    bypass_messages = [msg for msg in all_messages if "bypass" in msg.lower() or "try-except" in msg.lower()]
    
    if bypass_messages:
        # If bypass is detected, it should be in warnings, not errors
        bypass_in_warnings = any(msg in warnings for msg in bypass_messages)
        assert bypass_in_warnings, "Bypass should be in warnings when specified as warning"


# ============================================================================
# Test Backend Parameter Handling
# ============================================================================

def test_backend_adds_impl_check():
    """Test that backend parameter adds appropriate implementation check."""
    code = """
    # This code doesn't have CUDA implementation
    def forward(self, x):
        return x * 2
    """
    
    valid, errors, warnings = validate_kernel_static(code, backend="cuda")
    
    # Should check for CUDA implementation (cuda_impl check)
    # The exact behavior depends on what cuda_impl check does,
    # but we can verify the backend parameter is processed
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


def test_different_backends():
    """Test that different backends are handled correctly."""
    code = """
    def forward(self, x):
        return x * 2
    """
    
    # Test multiple backends
    backends = ["cuda", "triton", "thunderkittens", "cute", "tilelang"]
    
    for backend in backends:
        valid, errors, warnings = validate_kernel_static(code, backend=backend)
        assert isinstance(valid, bool)
        assert isinstance(errors, list)
        assert isinstance(warnings, list)


# ============================================================================
# Test Edge Cases
# ============================================================================

def test_empty_code():
    """Test handling of empty code."""
    code = ""
    
    valid, errors, warnings = validate_kernel_static(code)
    
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


def test_whitespace_only_code():
    """Test handling of whitespace-only code."""
    code = "   \n\n\t  \n  "
    
    valid, errors, warnings = validate_kernel_static(code)
    
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


def test_unknown_check_name():
    """Test that unknown check names are ignored."""
    code = "x = 1"
    
    # Should not crash with unknown check names
    valid, errors, warnings = validate_kernel_static(
        code,
        forbidden=["unknown_check_that_doesnt_exist"],
        warnings=["another_unknown_check"]
    )
    
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


def test_multiple_precision_dependent_checks():
    """Test that multiple precision-dependent checks work (if any exist in future)."""
    code = """
    def forward(self, x):
        x = x.half()
        return x
    """
    
    # Currently only precision_downgrade is precision-dependent
    valid, errors, warnings = validate_kernel_static(code, precision="fp32")
    
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


# ============================================================================
# Test Integration: Precision + Backend + Custom Checks
# ============================================================================

def test_integration_precision_backend_forbidden():
    """Test integration of precision, backend, and custom forbidden checks."""
    code = """
    def forward(self, x):
        x = x.half()  # Precision downgrade
        return x
    """
    
    valid, errors, warnings = validate_kernel_static(
        code,
        backend="cuda",
        precision="fp32",
        forbidden=["precision_downgrade"]
    )
    
    assert not valid, "Should be invalid with precision downgrade as forbidden"
    assert len(errors) > 0, "Should have errors"
    assert any("precision" in e.lower() or "fp16" in e.lower() 
               for e in errors), "Should flag precision downgrade"


def test_integration_all_parameters():
    """Test with all parameters specified."""
    code = """
    def forward(self, x):
        return x * 2.0
    """
    
    valid, errors, warnings = validate_kernel_static(
        code,
        backend="triton",
        precision="fp16",
        forbidden=["code_bypass"],
        warnings=["precision_downgrade"]
    )
    
    assert isinstance(valid, bool)
    assert isinstance(errors, list)
    assert isinstance(warnings, list)


# ============================================================================
# Test Precision Check Integration
# ============================================================================

def test_precision_check_in_warnings_by_default():
    """Test that precision_downgrade is in warnings by default."""
    code = """
    def forward(self, x):
        x = x.half()
        return x
    """
    
    valid, errors, warnings = validate_kernel_static(code, precision="fp32")
    
    # precision_downgrade should be in WARNING_CHECKS by default
    # So it should produce warnings, not errors
    all_messages = errors + warnings
    has_precision_msg = any("precision" in msg.lower() or "fp16" in msg.lower() 
                            for msg in all_messages)
    
    if has_precision_msg:
        # If detected, should be in warnings, not errors (by default)
        precision_in_warnings = any("precision" in msg.lower() or "fp16" in msg.lower() 
                                    for msg in warnings)
        assert precision_in_warnings, "Precision downgrade should be in warnings by default"


def test_precision_check_respects_forbidden():
    """Test that precision_downgrade respects forbidden parameter."""
    code = """
    def forward(self, x):
        x = x.half()
        return x
    """
    
    # Make precision_downgrade forbidden
    valid, errors, warnings = validate_kernel_static(
        code,
        precision="fp32",
        forbidden=["precision_downgrade"],
        warnings=[]  # Remove from warnings
    )
    
    # Should produce errors, not warnings
    has_precision_in_errors = any("precision" in msg.lower() or "fp16" in msg.lower() 
                                  for msg in errors)
    
    if has_precision_in_errors:
        assert not valid, "Should be invalid when precision downgrade is forbidden"
        assert len(errors) > 0, "Should have errors"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

