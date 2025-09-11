"""
Utility functions for graceful stopping of GEPA runs.
"""

import os
import signal
import time
from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class StopperProtocol(Protocol):
    """
    Protocol for stop condition objects.

    A stopper is a callable object that returns True when the optimization should stop.
    """

    def __call__(self, gepa_state) -> bool:
        """
        Check if the optimization should stop.

        Args:
            gepa_state: The current GEPA state containing optimization information

        Returns:
            True if the optimization should stop, False otherwise.
        """
        ...


class TimeoutStopCondition(StopperProtocol):
    # stop callback that stops after a specified timeout

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    def __call__(self, gepa_state) -> bool:
        # return true if timeout has been reached
        return time.time() - self.start_time > self.timeout_seconds


class FileStopper(StopperProtocol):
    # stop callback that stops when a specific file exists

    def __init__(self, stop_file_path: str):
        self.stop_file_path = stop_file_path

    def __call__(self, gepa_state) -> bool:
        # returns true if stop file exists
        return os.path.exists(self.stop_file_path)

    def create_stop_file(self):
        # create the stop file to signal stopping
        with open(self.stop_file_path, "w") as f:
            f.write(f"Stop requested at {time.time()}\n")

    def remove_stop_file(self):
        # remove the stop file
        if os.path.exists(self.stop_file_path):
            os.remove(self.stop_file_path)




class ScoreThresholdStopper(StopperProtocol):
    # stop callback that stops when a score threshold is reached

    def __init__(self, threshold: float, score_getter: Callable[[], float]):
        self.threshold = threshold
        self.score_getter = score_getter

    def __call__(self, gepa_state) -> bool:
        # return true if score threshold is reached
        try:
            current_score = self.score_getter()
            return current_score >= self.threshold
        except Exception:
            return False


class NoImprovementStopper(StopperProtocol):
    # stop callback that stops after a specified number of iterations without improvement

    def __init__(self, max_iterations_without_improvement: int):
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.best_score = float("-inf")
        self.iterations_without_improvement = 0

    def __call__(self, gepa_state) -> bool:
        # return true if max iterations without improvement reached
        try:
            current_score = max(gepa_state.program_full_scores_val_set) if gepa_state.program_full_scores_val_set else 0.0
            if current_score > self.best_score:
                self.best_score = current_score
                self.iterations_without_improvement = 0
            else:
                self.iterations_without_improvement += 1

            return self.iterations_without_improvement >= self.max_iterations_without_improvement
        except Exception:
            return False

    def reset(self):
        """Reset the counter (useful when manually improving the score)."""
        self.iterations_without_improvement = 0


class SignalStopper(StopperProtocol):
    # stop callback that stops when a signal is received

    def __init__(self, signals=None):
        self.signals = signals or [signal.SIGINT, signal.SIGTERM]
        self._stop_requested = False
        self._original_handlers = {}
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self._stop_requested = True

        # Store original handlers and set new ones
        for sig in self.signals:
            try:
                self._original_handlers[sig] = signal.signal(sig, signal_handler)
            except (OSError, ValueError):
                # Signal not available on this platform
                pass

    def __call__(self, gepa_state) -> bool:
        # return true if a signal was received
        return self._stop_requested

    def cleanup(self):
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError):
                pass


class MaxMetricCallsStopper(StopperProtocol):
    # stop callback that stops after a maximum number of metric calls

    def __init__(self, max_metric_calls: int):
        self.max_metric_calls = max_metric_calls

    def __call__(self, gepa_state) -> bool:
        # return true if max metric calls reached
        return gepa_state.total_num_evals >= self.max_metric_calls


class CompositeStopper(StopperProtocol):
    # stop callback that combines multiple stopping conditions

    def __init__(self, *stoppers: Callable[[Any], bool], mode: str = "any"):
        # initialize composite stopper

        self.stoppers = stoppers
        self.mode = mode

    def __call__(self, gepa_state) -> bool:
        # return true if stopping condition is met
        if self.mode == "any":
            return any(stopper(gepa_state) for stopper in self.stoppers)
        elif self.mode == "all":
            return all(stopper(gepa_state) for stopper in self.stoppers)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


