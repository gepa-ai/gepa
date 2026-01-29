"""Fitness evaluator for gepa_blog blackbox optimization."""

from typing import Any
import numpy as np
import json
from pathlib import Path

from gepa.optimize_anything import SideInfo
from gepa.utils.code_execution import execute_code as _execute_code, ExecutionMode
from examples.polynomial.evalset.problems import problems, problem_configs


def _truncate(text: str, limit: int = 4000) -> str:
    """Truncate text to avoid token limits."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n...[truncated]...\n" + text[-half:]



class FitnessEvaluator:
    """Fitness evaluator for GEPA blackbox optimization."""

    def __init__(
        self,
        problem_index: int,
        timeout: int = 300,
        evaluation_budget: int = 100,
        log_dir: str = None,
        seed: int = 0,
    ):
        self.problem_index = problem_index
        self.timeout = timeout
        self.evaluation_budget = evaluation_budget
        self.log_dir = Path(log_dir) if log_dir else None
        self.seed = seed

        # State tracking for warm-start (minimization: lower is better)
        self.evaluation_history = []
        self.best_score = float("inf")
        self.best_x = None

    def evaluate(self, candidate: dict[str, str], **kwargs) -> tuple[float, Any, SideInfo]:
        """Evaluate code candidate on a single problem."""
        code = candidate["code"]
        function = problems[self.problem_index]
        problem_config = problem_configs[self.problem_index]

        # Track state for this candidate
        eval_count = 0
        best_candidate_score = float("inf")
        errors = []

        def objective_function(x):
            nonlocal eval_count, best_candidate_score
            if eval_count >= self.evaluation_budget:
                raise ValueError(f"Evaluation budget exceeded: {eval_count} >= {self.evaluation_budget}")
            eval_count += 1

            score = function.do_evaluate(np.array(x))

            if score < best_candidate_score:
                best_candidate_score = score
            if score < self.best_score:
                self.best_score = score
                self.best_x = np.array(x).copy()

            self.evaluation_history.append({
                "score": score,
                "best_score": self.best_score,
            })
            return score

        # Execute code
        result = _execute_code(
            code=code,
            timeout=self.timeout,
            mode=ExecutionMode.IN_PROCESS,
            entry_point="solve",
            entry_point_kwargs={
                "objective_function": objective_function,
                "config": {"bounds": function.bounds, "dim": function.dim, "budget": self.evaluation_budget},
                "prev_best_x": self.best_x,
            },
            seed=self.seed,
        )

        x = result.variables.get("__return__")
        stdout = _truncate(result.stdout)
        stderr = _truncate(result.stderr)

        if result.error:
            errors.append(result.error)
        if result.traceback and result.traceback not in (result.error or ""):
            errors.append(result.traceback)
        if x is None or not isinstance(x, np.ndarray):
            errors.append("Code did not return a valid numpy array.")
        if eval_count == 0:
            errors.append("No objective_function calls were made.")

        # Use best score found, or inf if none
        score = best_candidate_score if best_candidate_score < float("inf") else float("inf")
        print(f"Best score from {eval_count} calls: {score}")

        side_info = {
            "score": score,
            "Input": problem_config["name"],
            "Prints": stdout,
            "Logs": stderr,
            "Error": "\n".join(errors) if errors else "",
        }

        output = {
            **side_info,
            "code": code,
            "X": " ".join(map(str, x.ravel())) if x is not None else "not found",
        }

        self.save()
        gepa_score = -score if score < float("inf") else -1e9
        return (gepa_score, output, side_info)

    def save(self, verbose: bool = False):
        """Save evaluation history to JSON."""
        if not self.log_dir:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        filename = self.log_dir / f"evaluation_history.json"
        try:
            with open(filename, "w") as f:
                json.dump(self.evaluation_history, f, indent=2, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
            if verbose:
                print(f"Saved to {filename}")
        except Exception as e:
            print(f"Warning: Failed to save: {e}")

