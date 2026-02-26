import numpy as np

def geometric_mean_speed_ratio_correct_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    prod = np.prod(speed_up)
    n_correct = np.sum(is_correct) # Count number of correct samples

    return prod ** (1 / n_correct) if n_correct > 0 else 0

def geometric_mean_speed_ratio_correct_and_faster_only(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int) -> float:
    """
    Geometric mean of the speed ratio for correct samples that have speedup > 1
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    speed_up = np.array([x for x in speed_up if x > 1])
    prod = np.prod(speed_up)
    n_correct_and_faster = len(speed_up)

    return prod ** (1 / n_correct_and_faster) if n_correct_and_faster > 0 else 0

def fastp(is_correct: np.ndarray, baseline_speed: np.ndarray, actual_speed: np.ndarray, n: int, p: float) -> float:
    """
    Rate of samples within a threshold p
    """
    filtered_baseline_speed = np.array([x for i, x in enumerate(baseline_speed) if is_correct[i]])
    filtered_actual_speed = np.array([x for i, x in enumerate(actual_speed) if is_correct[i]])
    speed_up = filtered_baseline_speed / filtered_actual_speed
    fast_p_score = np.sum(speed_up > p)
    return fast_p_score / n if n > 0 else 0