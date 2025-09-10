"""
Utility functions for graceful stopping of GEPA runs.
"""

import os
import time
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class StopperProtocol(Protocol):
    """
    Protocol for stop condition objects.
    
    A stopper is a callable object that returns True when the optimization should stop.
    """

    def __call__(self) -> bool:
        """
        Check if the optimization should stop.
        
        Returns:
            True if the optimization should stop, False otherwise.
        """
        ...


class TimeoutStopCondition:
    # stop callback that stops after a specified timeout

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    def __call__(self) -> bool:
        # return true if timeout has been reached
        return time.time() - self.start_time > self.timeout_seconds


class FileStopper:
    # stop callback that stops when a specific file exists

    def __init__(self, stop_file_path: str):
        self.stop_file_path = stop_file_path

    def __call__(self) -> bool:
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


class IterationStopper:
    # stop callback that stops after a specified number of iterations

    def __init__(self, max_iterations: int):
        self.max_iterations = max_iterations
        self.current_iterations = 0

    def __call__(self) -> bool:
        # return true if max iterations reached
        return self.current_iterations >= self.max_iterations

    def increment(self):
        # increment the iteration counter
        self.current_iterations += 1


class ScoreThresholdStopper:
    # stop callback that stops when a score threshold is reached

    def __init__(self, threshold: float, score_getter: Callable[[], float]):
        self.threshold = threshold
        self.score_getter = score_getter

    def __call__(self) -> bool:
        # return true if score threshold is reached
        try:
            current_score = self.score_getter()
            return current_score >= self.threshold
        except Exception:
            return False


class CompositeStopper:
    # stop callback that combines multiple stopping conditions

    def __init__(self, *stoppers: Callable[[], bool], mode: str = "any"):
        # initialize composite stopper

        self.stoppers = stoppers
        self.mode = mode

    def __call__(self) -> bool:
        # return true if stopping condition is met
        if self.mode == "any":
            return any(stopper() for stopper in self.stoppers)
        elif self.mode == "all":
            return all(stopper() for stopper in self.stoppers)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


def create_timeout_stopcondition(timeout_seconds: float) -> Callable[[], bool]:
    # create a timeout-based stop callback
    return TimeoutStopCondition(timeout_seconds)


def create_file_stopper(stop_file_path: str) -> tuple[Callable[[], bool], FileStopper]:
    # creates a file-based stop callback
    file_stopper = FileStopper(stop_file_path)
    return file_stopper, file_stopper


def create_iteration_stopper(max_iterations: int) -> tuple[Callable[[], bool], IterationStopper]:
    # creates an iteration-based stop callback
    iteration_stopper = IterationStopper(max_iterations)
    return iteration_stopper, iteration_stopper


def create_score_threshold_stopper(threshold: float, score_getter: Callable[[], float]) -> Callable[[], bool]:
    # creates a score threshold-based stop callback
    return ScoreThresholdStopper(threshold, score_getter)


def create_composite_stopper(*stoppers: Callable[[], bool], mode: str = "any") -> Callable[[], bool]:
    # creates a composite stop callback that combines multiple conditions
    return CompositeStopper(*stoppers, mode=mode)
