"""
Evaluator for the Real-Time Adaptive Signal Processing Algorithm

Adapted for EvolveAdapter: accepts a single data_instance (signal pair) and returns EvaluationResult.

This evaluator implements the multi-objective optimization function:
J(θ) = α₁·S(θ) + α₂·L_recent(θ) + α₃·L_avg(θ) + α₄·R(θ)

Where:
- S(θ): Slope change penalty - counts directional reversals
- L_recent(θ): Instantaneous lag error - |y[n] - x[n]|
- L_avg(θ): Average tracking error over window
- R(θ): False reversal penalty - mismatched trend changes
- α₁=0.3, α₂=α₃=0.2, α₄=0.3: Weighting coefficients
"""

import concurrent.futures
import importlib.util
import time
import traceback

import numpy as np
from openevolve.evaluation_result import EvaluationResult  # type: ignore
from scipy.stats import pearsonr


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """
    Run a function with a timeout using concurrent.futures
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")


def safe_float(value):
    """Convert a value to float safely"""
    try:
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def calculate_slope_changes(signal_data):
    """
    Calculate slope change penalty S(θ) - counts directional reversals

    Args:
        signal_data: 1D array of signal values

    Returns:
        Number of slope changes (directional reversals)
    """
    if len(signal_data) < 3:
        return 0

    # Calculate differences
    diffs = np.diff(signal_data)

    # Count sign changes in consecutive differences
    sign_changes = 0
    for i in range(1, len(diffs)):
        if np.sign(diffs[i]) != np.sign(diffs[i - 1]) and diffs[i - 1] != 0:
            sign_changes += 1

    return sign_changes


def calculate_lag_error(filtered_signal, original_signal, window_size):
    """
    Calculate instantaneous lag error L_recent(θ) = |y[n] - x[n]|

    Args:
        filtered_signal: Output of the filter
        original_signal: Original input signal
        window_size: Size of the processing window

    Returns:
        Instantaneous lag error at the most recent sample
    """
    if len(filtered_signal) == 0:
        return 1.0  # Maximum penalty

    # Account for processing delay
    delay = window_size - 1
    if len(original_signal) <= delay:
        return 1.0

    # Compare the last filtered sample with the corresponding original sample
    recent_filtered = filtered_signal[-1]
    recent_original = original_signal[delay + len(filtered_signal) - 1]

    return abs(recent_filtered - recent_original)


def calculate_average_tracking_error(filtered_signal, original_signal, window_size):
    """
    Calculate average tracking error L_avg(θ) over the window

    Args:
        filtered_signal: Output of the filter
        original_signal: Original input signal
        window_size: Size of the processing window

    Returns:
        Average absolute error over the processed samples
    """
    if len(filtered_signal) == 0:
        return 1.0  # Maximum penalty

    # Account for processing delay
    delay = window_size - 1
    if len(original_signal) <= delay:
        return 1.0

    # Align signals
    aligned_original = original_signal[delay : delay + len(filtered_signal)]

    # Ensure same length
    min_length = min(len(filtered_signal), len(aligned_original))
    if min_length == 0:
        return 1.0

    filtered_aligned = filtered_signal[:min_length]
    original_aligned = aligned_original[:min_length]

    # Calculate mean absolute error
    return np.mean(np.abs(filtered_aligned - original_aligned))


def calculate_false_reversal_penalty(filtered_signal, clean_signal, window_size):
    """
    Calculate false reversal penalty R(θ) - mismatched trend changes

    Args:
        filtered_signal: Output of the filter
        clean_signal: Ground truth clean signal
        window_size: Size of the processing window

    Returns:
        Penalty for trend changes that don't match the clean signal
    """
    if len(filtered_signal) < 3 or len(clean_signal) < 3:
        return 0

    # Account for processing delay
    delay = window_size - 1
    if len(clean_signal) <= delay:
        return 1.0

    # Align signals
    aligned_clean = clean_signal[delay : delay + len(filtered_signal)]
    min_length = min(len(filtered_signal), len(aligned_clean))

    if min_length < 3:
        return 0

    filtered_aligned = filtered_signal[:min_length]
    clean_aligned = aligned_clean[:min_length]

    # Calculate trend changes for both signals
    filtered_diffs = np.diff(filtered_aligned)
    clean_diffs = np.diff(clean_aligned)

    # Count mismatched trend changes
    false_reversals = 0
    for i in range(1, len(filtered_diffs)):
        # Check if there's a trend change in filtered signal
        filtered_change = np.sign(filtered_diffs[i]) != np.sign(filtered_diffs[i - 1]) and filtered_diffs[i - 1] != 0

        # Check if there's a corresponding trend change in clean signal
        clean_change = np.sign(clean_diffs[i]) != np.sign(clean_diffs[i - 1]) and clean_diffs[i - 1] != 0

        # Count as false reversal if filtered has change but clean doesn't
        if filtered_change and not clean_change:
            false_reversals += 1

    return false_reversals


def calculate_composite_score(S, L_recent, L_avg, R, alpha=[0.3, 0.2, 0.2, 0.3]):
    """
    Calculate the composite metric J(θ) = α₁·S(θ) + α₂·L_recent(θ) + α₃·L_avg(θ) + α₄·R(θ)

    All metrics are normalized and converted to penalties (higher = worse)
    The final score is converted to a maximization problem (higher = better)
    """
    # Normalize slope changes (typical range 0-100)
    S_norm = min(S / 50.0, 2.0)

    # Lag errors are already in reasonable range (0-10 typically)
    L_recent_norm = min(L_recent, 2.0)
    L_avg_norm = min(L_avg, 2.0)

    # Normalize false reversals (typical range 0-50)
    R_norm = min(R / 25.0, 2.0)

    # Calculate weighted penalty
    penalty = alpha[0] * S_norm + alpha[1] * L_recent_norm + alpha[2] * L_avg_norm + alpha[3] * R_norm

    # Convert to maximization score (higher is better)
    score = 1.0 / (1.0 + penalty)

    return score


def generate_test_signals(num_signals=5):
    """
    Generate multiple test signals with different characteristics.
    Used for creating the training dataset.
    """
    test_signals = []

    for i in range(num_signals):
        np.random.seed(42 + i)  # Different seed for each signal
        length = 500 + i * 100  # Varying lengths
        noise_level = 0.2 + i * 0.1  # Varying noise levels

        t = np.linspace(0, 10, length)

        # Different signal characteristics
        if i == 0:
            # Smooth sinusoidal with trend
            clean = 2 * np.sin(2 * np.pi * 0.5 * t) + 0.1 * t
        elif i == 1:
            # Multiple frequency components
            clean = np.sin(2 * np.pi * 0.5 * t) + 0.5 * np.sin(2 * np.pi * 2 * t) + 0.2 * np.sin(2 * np.pi * 5 * t)
        elif i == 2:
            # Non-stationary with changing frequency
            clean = np.sin(2 * np.pi * (0.5 + 0.2 * t) * t)
        elif i == 3:
            # Step changes
            clean = np.concatenate(
                [
                    np.ones(length // 3),
                    2 * np.ones(length // 3),
                    0.5 * np.ones(length - 2 * (length // 3)),
                ]
            )
        else:
            # Random walk with trend
            clean = np.cumsum(np.random.randn(length) * 0.1) + 0.05 * t

        # Add noise
        noise = np.random.normal(0, noise_level, length)
        noisy = clean + noise

        test_signals.append((noisy, clean))

    return test_signals


# CHANGE: Modified to accept data_instance and return EvaluationResult
def evaluate(program_path: str, data_instance: tuple | dict) -> EvaluationResult:
    """
    Evaluate the signal processing algorithm on a single signal pair.

    Args:
        program_path: Path to the program file to evaluate
        data_instance: Either a tuple (noisy_signal, clean_signal) or dict with keys:
            - "noisy_signal": 1D numpy array of noisy signal
            - "clean_signal": 1D numpy array of clean (ground truth) signal
            - "window_size": Optional, defaults to 20
            - "noise_level": Optional, for metadata

    Returns:
        EvaluationResult for this data instance
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        if spec is None or spec.loader is None:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": 1.0}, artifacts={"error": "Failed to load program module"}
            )
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if required function exists
        if not hasattr(program, "run_signal_processing"):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": 1.0},
                artifacts={"error": "Missing run_signal_processing function"},
            )

        # Extract signal pair from data_instance
        if isinstance(data_instance, tuple):
            noisy_signal, clean_signal = data_instance
            window_size = 20  # Default
        elif isinstance(data_instance, dict):
            noisy_signal = data_instance.get("noisy_signal")
            clean_signal = data_instance.get("clean_signal")
            window_size = data_instance.get("window_size", 20)
        else:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": 1.0},
                artifacts={"error": f"Invalid data_instance type: {type(data_instance)}"},
            )

        # Convert to numpy arrays if needed
        noisy_signal = np.array(noisy_signal)
        clean_signal = np.array(clean_signal)

        # Run the algorithm with timeout
        start_time = time.time()

        # Call the program's main function
        result = run_with_timeout(
            program.run_signal_processing,
            kwargs={
                "signal_length": len(noisy_signal),
                "noise_level": 0.3,  # Can be extracted from data_instance if provided
                "window_size": window_size,
            },
            timeout_seconds=10,
        )

        execution_time = time.time() - start_time

        # Validate result format
        if not isinstance(result, dict):
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": 1.0},
                artifacts={"error": "Invalid result format: expected dict"},
            )

        if "filtered_signal" not in result:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": 1.0}, artifacts={"error": "Missing filtered_signal in result"}
            )

        filtered_signal = result["filtered_signal"]

        if len(filtered_signal) == 0:
            return EvaluationResult(
                metrics={"combined_score": 0.0, "error": 1.0}, artifacts={"error": "Empty filtered signal"}
            )

        # Convert to numpy arrays
        filtered_signal = np.array(filtered_signal)

        # Calculate all penalty components
        S = calculate_slope_changes(filtered_signal)
        L_recent = calculate_lag_error(filtered_signal, noisy_signal, window_size)
        L_avg = calculate_average_tracking_error(filtered_signal, noisy_signal, window_size)
        R = calculate_false_reversal_penalty(filtered_signal, clean_signal, window_size)

        # Calculate composite score
        composite_score = calculate_composite_score(S, L_recent, L_avg, R)

        # Additional quality metrics
        correlation = 0.0
        noise_reduction = 0.0

        try:
            # Calculate correlation with clean signal
            delay = window_size - 1
            aligned_clean = clean_signal[delay : delay + len(filtered_signal)]
            min_length = min(len(filtered_signal), len(aligned_clean))

            if min_length > 1:
                corr_result = pearsonr(filtered_signal[:min_length], aligned_clean[:min_length])
                correlation = corr_result[0] if not np.isnan(corr_result[0]) else 0.0

            # Calculate noise reduction
            aligned_noisy = noisy_signal[delay : delay + len(filtered_signal)]
            aligned_noisy = aligned_noisy[:min_length]
            aligned_clean = aligned_clean[:min_length]

            if min_length > 0:
                noise_before = np.var(aligned_noisy - aligned_clean)
                noise_after = np.var(filtered_signal[:min_length] - aligned_clean)
                noise_reduction = (noise_before - noise_after) / noise_before if noise_before > 0 else 0
                noise_reduction = max(0, noise_reduction)  # Ensure non-negative

        except Exception:
            # Continue with correlation=0, noise_reduction=0 if calculation fails
            pass

        # Calculate derived scores
        smoothness_score = 1.0 / (1.0 + S / 20.0)  # Higher is better
        responsiveness_score = 1.0 / (1.0 + L_recent)  # Higher is better
        accuracy_score = max(0, correlation)  # 0-1, higher is better
        efficiency_score = min(1.0, 1.0 / max(0.001, execution_time))  # Speed bonus

        # Overall score combining multiple factors (used as combined_score)
        overall_score = (
            0.4 * composite_score  # Primary metric
            + 0.2 * smoothness_score  # Smoothness
            + 0.2 * accuracy_score  # Correlation with clean signal
            + 0.1 * noise_reduction  # Noise reduction capability
            + 0.1 * 1.0  # Reliability (single signal, so 1.0 if successful)
        )

        # Create artifacts
        artifacts = {
            "slope_changes": int(S),
            "lag_error": float(L_recent),
            "avg_error": float(L_avg),
            "false_reversals": int(R),
            "correlation": float(correlation),
            "noise_reduction": float(noise_reduction),
            "execution_time": float(execution_time),
            "signal_length": len(filtered_signal),
            "window_size": window_size,
        }

        return EvaluationResult(
            metrics={
                "composite_score": safe_float(composite_score),
                "combined_score": safe_float(overall_score),  # Primary selection metric
                "slope_changes": safe_float(S),
                "lag_error": safe_float(L_recent),
                "avg_error": safe_float(L_avg),
                "false_reversals": safe_float(R),
                "correlation": safe_float(correlation),
                "noise_reduction": safe_float(noise_reduction),
                "smoothness_score": safe_float(smoothness_score),
                "responsiveness_score": safe_float(responsiveness_score),
                "accuracy_score": safe_float(accuracy_score),
                "efficiency_score": safe_float(efficiency_score),
            },
            artifacts=artifacts,
        )

    except TimeoutError:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": 1.0}, artifacts={"error": "Timeout during evaluation"}
        )
    except Exception as e:
        return EvaluationResult(
            metrics={"combined_score": 0.0, "error": 1.0},
            artifacts={"error": str(e), "traceback": traceback.format_exc()},
        )


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path: str, data_instance: tuple | dict) -> EvaluationResult:
    """
    Stage 1 evaluation: Quick validation that the program runs without errors.

    Args:
        program_path: Path to the program file to evaluate
        data_instance: Signal pair (noisy_signal, clean_signal) or dict

    Returns:
        EvaluationResult for this data instance
    """
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        if spec is None or spec.loader is None:
            return EvaluationResult(
                metrics={"runs_successfully": 0.0, "combined_score": 0.0, "error": 1.0},
                artifacts={"error": "Failed to load program module"},
            )
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if required function exists
        if not hasattr(program, "run_signal_processing"):
            return EvaluationResult(
                metrics={"runs_successfully": 0.0, "combined_score": 0.0, "error": 1.0},
                artifacts={"error": "Missing run_signal_processing function"},
            )

        # Extract signal from data_instance
        if isinstance(data_instance, tuple):
            noisy_signal, _ = data_instance
            window_size = 10  # Smaller for stage 1
        elif isinstance(data_instance, dict):
            noisy_signal = data_instance.get("noisy_signal")
            window_size = data_instance.get("window_size", 10)
        else:
            return EvaluationResult(
                metrics={"runs_successfully": 0.0, "combined_score": 0.0, "error": 1.0},
                artifacts={"error": f"Invalid data_instance type: {type(data_instance)}"},
            )

        # Quick test with smaller signal
        try:
            result = run_with_timeout(
                program.run_signal_processing,
                kwargs={
                    "signal_length": min(len(noisy_signal), 100),  # Use smaller for stage 1
                    "noise_level": 0.3,
                    "window_size": window_size,
                },
                timeout_seconds=5,
            )

            if isinstance(result, dict) and "filtered_signal" in result:
                filtered_signal = result["filtered_signal"]
                if len(filtered_signal) > 0:
                    # Quick quality check
                    composite_score = 0.5  # Baseline score for working programs

                    # Bonus for reasonable output length
                    expected_length = min(len(noisy_signal), 100) - window_size + 1
                    if len(filtered_signal) == expected_length:
                        composite_score += 0.2

                    return EvaluationResult(
                        metrics={
                            "runs_successfully": 1.0,
                            "combined_score": composite_score,
                        },
                        artifacts={"output_length": len(filtered_signal)},
                    )
                else:
                    return EvaluationResult(
                        metrics={"runs_successfully": 0.5, "combined_score": 0.0},
                        artifacts={"error": "Empty filtered signal"},
                    )
            else:
                return EvaluationResult(
                    metrics={"runs_successfully": 0.3, "combined_score": 0.0},
                    artifacts={"error": "Invalid result format"},
                )

        except TimeoutError:
            return EvaluationResult(
                metrics={"runs_successfully": 0.0, "combined_score": 0.0, "error": 1.0},
                artifacts={"error": "Timeout in stage 1"},
            )
        except Exception as e:
            return EvaluationResult(
                metrics={"runs_successfully": 0.0, "combined_score": 0.0, "error": 1.0},
                artifacts={"error": f"Stage 1 error: {e!s}"},
            )

    except Exception as e:
        return EvaluationResult(
            metrics={"runs_successfully": 0.0, "combined_score": 0.0, "error": 1.0},
            artifacts={"error": f"Stage 1 failed: {e!s}"},
        )


def evaluate_stage2(program_path: str, data_instance: tuple | dict) -> EvaluationResult:
    """
    Stage 2 evaluation: Full evaluation with the signal pair.

    Args:
        program_path: Path to the program file to evaluate
        data_instance: Signal pair (noisy_signal, clean_signal) or dict

    Returns:
        EvaluationResult for this data instance
    """
    # Full evaluation as in the main evaluate function
    return evaluate(program_path, data_instance)
