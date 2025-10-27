"""
Automatic stopping criterion for optimization based on multiple convergence signals.

The StopGovernor monitors:
- Pareto hypervolume gain rate
- Best candidate improvement
- Frontier stability
- QD grid novelty
- Cost efficiency (ROI)
- Statistical significance

It stops when all signals indicate plateau, with hysteresis to avoid premature stopping.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpochMetrics:
    """Metrics collected during an optimization epoch."""

    round_num: int
    hypervolume: float
    new_evaluations: int
    best_quality: float
    best_cost: float  # negative tokens
    frontier_ids: set[str]
    qd_filled_cells: int
    qd_total_cells: int
    qd_novelty_rate: float  # fraction landing in new cells
    total_tokens_spent: int


@dataclass
class StopGovernorConfig:
    """Configuration for automatic stopping."""

    # EWMA smoothing parameter
    alpha: float = 0.3

    # Hysteresis: require H consecutive epochs below threshold
    hysteresis_window: int = 5  # Increased from 3 for more patience

    # Overall stop threshold (0-1, lower = easier to stop)
    stop_threshold: float = 0.15  # Lowered from 0.2 to be more conservative

    # Thresholds for each signal (minimum useful improvement)
    tau_hv: float = 1e-5  # Lowered from 1e-4 for harder tasks
    tau_quality: float = 1e-3  # 0.1% absolute improvement per epoch
    tau_quality_relative: float = 0.01  # OR 1% relative improvement per epoch (whichever is less strict)
    tau_cost: float = 5.0  # Lowered from 10.0
    tau_qd_novelty: float = 0.03  # Lowered from 0.05 (3% instead of 5%)
    tau_roi: float = 1e-6  # Lowered from 1e-5

    # Stability thresholds
    min_jaccard_for_stable: float = 0.85  # Increased from 0.8 (require more stability)
    max_epsilon_churn: float = 0.1  # Epsilon-dominance churn

    # Signal weights (for max aggregation)
    weight_hv: float = 1.0
    weight_quality: float = 1.0
    weight_cost: float = 0.6  # Reduced from 0.8 (care less about cost)
    weight_qd: float = 0.7  # Increased from 0.6 (care more about exploration)
    weight_roi: float = 0.5  # Reduced from 0.7

    # Stability penalty exponent
    stability_penalty_beta: float = 0.5  # Reduced from 0.7 (less harsh penalty)

    # Hard caps (always enforced)
    max_no_improvement_epochs: int = 12  # Doubled from 6 for more patience


class StopGovernor:
    """
    Monitors optimization progress and decides when to stop.

    Uses multiple signals (hypervolume, quality, stability, novelty, ROI)
    combined with EWMA smoothing and hysteresis to avoid premature stopping.
    """

    def __init__(self, config: StopGovernorConfig | None = None):
        self.config = config or StopGovernorConfig()

        # History tracking
        self.epochs: list[EpochMetrics] = []
        self.prev_metrics: EpochMetrics | None = None

        # EWMA state
        self.ewma_hv_rate: float = 0.0
        self.ewma_quality_delta: float = 0.0
        self.ewma_cost_delta: float = 0.0
        self.ewma_roi: float = 0.0

        # Hysteresis counter
        self.epochs_below_threshold: int = 0
        self.epochs_no_improvement: int = 0

        # Last best values
        self.last_best_quality: float = 0.0
        self.last_best_cost: float = float("-inf")

    def update(self, metrics: EpochMetrics) -> None:
        """Record new epoch metrics and update EWMA state."""
        self.epochs.append(metrics)

        if self.prev_metrics is not None:
            # Compute deltas
            delta_hv = metrics.hypervolume - self.prev_metrics.hypervolume
            hv_rate = delta_hv / max(1, metrics.new_evaluations)

            delta_quality = metrics.best_quality - self.prev_metrics.best_quality
            delta_cost = metrics.best_cost - self.prev_metrics.best_cost  # Higher = better (less negative)

            roi = delta_hv / max(1, metrics.total_tokens_spent - self.prev_metrics.total_tokens_spent)

            # Update EWMA
            alpha = self.config.alpha
            self.ewma_hv_rate = alpha * hv_rate + (1 - alpha) * self.ewma_hv_rate
            self.ewma_quality_delta = alpha * delta_quality + (1 - alpha) * self.ewma_quality_delta
            self.ewma_cost_delta = alpha * delta_cost + (1 - alpha) * self.ewma_cost_delta
            self.ewma_roi = alpha * roi + (1 - alpha) * self.ewma_roi

            # Track improvement
            if delta_quality > self.config.tau_quality:
                self.epochs_no_improvement = 0
            else:
                self.epochs_no_improvement += 1

        else:
            # First epoch - initialize
            self.ewma_hv_rate = 0.0
            self.ewma_quality_delta = 0.0
            self.ewma_cost_delta = 0.0
            self.ewma_roi = 0.0

        self.prev_metrics = metrics
        self.last_best_quality = metrics.best_quality
        self.last_best_cost = metrics.best_cost

    def compute_signals(self) -> dict[str, float]:
        """Compute normalized 0-1 signals for each stopping criterion."""
        if len(self.epochs) < 2:
            # Not enough data yet
            return {
                "s_hv": 1.0,
                "s_quality": 1.0,
                "s_cost": 1.0,
                "s_qd": 1.0,
                "s_roi": 1.0,
                "s_stability": 0.0,
                "jaccard": 0.0,
            }

        curr = self.epochs[-1]
        prev = self.epochs[-2]

        # Signal 1: HV rate (normalized by threshold)
        s_hv = min(1.0, self.ewma_hv_rate / self.config.tau_hv) if self.config.tau_hv > 0 else 1.0

        # Signal 2: Quality improvement (use whichever threshold is more lenient)
        if self.config.tau_quality > 0 and self.config.tau_quality_relative > 0:
            # Absolute improvement signal
            absolute_signal = self.ewma_quality_delta / self.config.tau_quality
            # Relative improvement signal (avoid division by zero)
            relative_signal = (
                self.ewma_quality_delta / max(0.01, self.last_best_quality)
            ) / self.config.tau_quality_relative
            # Use the more lenient of the two (higher signal = more improvement detected)
            s_quality = min(1.0, max(absolute_signal, relative_signal))
        elif self.config.tau_quality > 0:
            s_quality = min(1.0, self.ewma_quality_delta / self.config.tau_quality)
        else:
            s_quality = 1.0

        # Signal 3: Cost improvement (tokens saved)
        s_cost = min(1.0, self.ewma_cost_delta / self.config.tau_cost) if self.config.tau_cost > 0 else 1.0

        # Signal 4: QD novelty
        s_qd = min(1.0, curr.qd_novelty_rate / self.config.tau_qd_novelty) if self.config.tau_qd_novelty > 0 else 1.0

        # Signal 5: ROI
        s_roi = min(1.0, self.ewma_roi / self.config.tau_roi) if self.config.tau_roi > 0 else 1.0

        # Signal 6: Frontier stability (inverse - high stability = low score)
        jaccard = self._compute_jaccard(prev.frontier_ids, curr.frontier_ids)
        s_stability = jaccard if jaccard > self.config.min_jaccard_for_stable else 0.0

        return {
            "s_hv": s_hv,
            "s_quality": s_quality,
            "s_cost": s_cost,
            "s_qd": s_qd,
            "s_roi": s_roi,
            "s_stability": s_stability,
            "jaccard": jaccard,
        }

    def compute_stop_score(self) -> tuple[float, dict[str, float]]:
        """
        Compute overall stop score (0-1) and individual signals.

        Returns:
            (stop_score, signals_dict)

        stop_score interpretation:
            1.0 = strong improvement, keep going
            0.0 = complete plateau, should stop
        """
        signals = self.compute_signals()

        # Conservative OR-style: if ANY signal is strong, keep going
        max_signal = max(
            self.config.weight_hv * signals["s_hv"],
            self.config.weight_quality * signals["s_quality"],
            self.config.weight_cost * signals["s_cost"],
            self.config.weight_qd * signals["s_qd"],
            self.config.weight_roi * signals["s_roi"],
        )

        # Penalize by stability (if frontier is very stable and max_signal is low, reduce score)
        stability_penalty = (1.0 - signals["s_stability"]) ** self.config.stability_penalty_beta

        stop_score = max_signal * stability_penalty

        signals["stop_score"] = stop_score
        signals["max_signal"] = max_signal
        signals["stability_penalty"] = stability_penalty

        return stop_score, signals

    def should_stop(self) -> tuple[bool, dict[str, any]]:
        """
        Determine if optimization should stop.

        Returns:
            (should_stop, debug_info)
        """
        if len(self.epochs) < 2:
            return False, {"reason": "insufficient_epochs", "epochs": len(self.epochs)}

        stop_score, signals = self.compute_stop_score()

        # Check hysteresis
        if stop_score < self.config.stop_threshold:
            self.epochs_below_threshold += 1
        else:
            self.epochs_below_threshold = 0

        # Hard cap: no improvement for too long
        hard_stop = self.epochs_no_improvement >= self.config.max_no_improvement_epochs

        # Hysteresis stop: below threshold for H consecutive epochs
        hysteresis_stop = self.epochs_below_threshold >= self.config.hysteresis_window

        should_stop = hard_stop or hysteresis_stop

        debug_info = {
            "stop_score": stop_score,
            "signals": signals,
            "epochs_below_threshold": self.epochs_below_threshold,
            "epochs_no_improvement": self.epochs_no_improvement,
            "hysteresis_window": self.config.hysteresis_window,
            "threshold": self.config.stop_threshold,
            "reason": None,
        }

        if should_stop:
            if hard_stop:
                debug_info["reason"] = f"no_improvement_for_{self.epochs_no_improvement}_epochs"
            else:
                debug_info["reason"] = (
                    f"score_below_{self.config.stop_threshold}_for_{self.epochs_below_threshold}_epochs"
                )

        return should_stop, debug_info

    def _compute_jaccard(self, set1: set[str], set2: set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def reset(self) -> None:
        """Reset governor state."""
        self.epochs.clear()
        self.prev_metrics = None
        self.ewma_hv_rate = 0.0
        self.ewma_quality_delta = 0.0
        self.ewma_cost_delta = 0.0
        self.ewma_roi = 0.0
        self.epochs_below_threshold = 0
        self.epochs_no_improvement = 0


def compute_hypervolume_2d(
    points: list[tuple[float, float]],
    reference: tuple[float, float] = (0.0, 0.0),
) -> float:
    """
    Compute 2D hypervolume for quality/cost Pareto frontier.

    Args:
        points: List of (quality, neg_cost) tuples (both higher is better)
        reference: Reference point (lower-left corner)

    Returns:
        Hypervolume dominated by the points
    """
    if not points:
        return 0.0

    # Filter dominated points and sort by quality (descending)
    pareto = []
    for q, c in points:
        dominated = False
        for pq, pc in pareto:
            if pq >= q and pc >= c and (pq > q or pc > c):
                dominated = True
                break
        if not dominated:
            # Remove points dominated by this one
            pareto = [(pq, pc) for pq, pc in pareto if not (q >= pq and c >= pc and (q > pq or c > pc))]
            pareto.append((q, c))

    # Sort by quality descending
    pareto.sort(reverse=True)

    # Compute hypervolume using step function
    ref_q, ref_c = reference
    hv = 0.0
    prev_c = ref_c

    for q, c in pareto:
        if q > ref_q and c > prev_c:
            hv += (q - ref_q) * (c - prev_c)
            prev_c = c

    return hv
