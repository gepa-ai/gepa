from typing import Any
import numpy as np
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
import signal
import hashlib
import json
from pathlib import Path
import os
import time

from gepa.optimize_anything import SideInfo
from gepa.core.adapter import DataInst
from experiments.polynomial.evalset import Rastrigin
from experiments.polynomial.problems import problems, problem_configs

# Try to import psutil for killing child processes
try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class TimeLimitError(Exception):
    pass


def _json_default(obj):
    """Handle numpy types for JSON serialization."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _alarm_handler(signum, frame):
    raise TimeLimitError("Time Limit Exceeded")


def execute_code(code_string, global_vars=None, timeout=5, seed=None):
    """Execute code with timeout support.

    Note: signal.alarm() only works for the current process/thread.
    If the code spawns subprocesses, they won't be killed automatically.
    We try to kill child processes using psutil if available.
    """
    if seed is not None:
        import random

        random.seed(seed)
        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            pass
        try:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    f_out = io.StringIO()
    f_err = io.StringIO()

    if global_vars is None:
        context = {"__name__": "__main__"}
    else:
        context = global_vars.copy()
        context["__name__"] = "__main__"

    error = ""
    start_time = time.time()
    current_pid = os.getpid()

    # Set up signal handler for timeout
    old_handler = None
    if timeout is not None and timeout > 0 and hasattr(signal, "SIGALRM"):
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        # Use setitimer for more precision if available, otherwise fall back to alarm
        try:
            signal.setitimer(signal.ITIMER_REAL, timeout)
        except AttributeError:
            signal.alarm(int(timeout))

    try:
        with redirect_stdout(f_out), redirect_stderr(f_err):
            exec(code_string, context)
    except TimeLimitError:
        error = f"TimeLimitError: Code execution exceeded {timeout} seconds."
        # Try to kill any child processes if psutil is available
        if HAS_PSUTIL:
            try:
                parent = psutil.Process(current_pid)
                # Kill all children recursively
                for child in parent.children(recursive=True):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
            except Exception:
                pass  # Ignore errors in cleanup
    except Exception as e:
        error = str(e) + "\n" + traceback.format_exc()
    finally:
        # Disable the alarm and restore previous handler
        if timeout is not None and timeout > 0 and hasattr(signal, "SIGALRM"):
            try:
                signal.setitimer(signal.ITIMER_REAL, 0)
            except AttributeError:
                signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

        # Double-check: if we've exceeded timeout but no exception was raised, add error
        elapsed = time.time() - start_time
        if timeout is not None and timeout > 0 and elapsed > timeout and not error:
            error = f"TimeLimitError: Code execution exceeded {timeout} seconds (elapsed: {elapsed:.2f}s)."
            # Try to kill children even if signal didn't fire
            if HAS_PSUTIL:
                try:
                    parent = psutil.Process(current_pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                except Exception:
                    pass

    return {
        "output": f_out.getvalue(),
        "logs": f_err.getvalue(),
        "results": context,
        "error": error,
    }


class FitnessEvaluator:
    """Class-based fitness evaluator for GEPA blackbox optimization.

    Encapsulates all state for candidate caching, evaluation tracking, and logging.
    """

    def __init__(
        self,
        timeout: int = 300,
        evaluation_budget: int = 100,
        log_dir: str = None,
        base_seed: int = 0,
    ):
        """Initialize the fitness evaluator.

        Args:
            timeout: Timeout in seconds for code execution
            evaluation_budget: Maximum number of evaluation calls allowed per candidate
            log_dir: Directory to save trajectory files
            base_seed: Base seed for reproducibility
        """
        self.timeout = timeout
        self.evaluation_budget = evaluation_budget
        self.log_dir = Path(log_dir) if log_dir else None
        self.base_seed = base_seed

        # State tracking (replaces globals)
        self.candidate_cache = {}  # code_hash -> candidate data
        self.code_hash_to_id = {}  # code_hash -> sequential candidate_id
        self.next_candidate_id = 0  # Counter for sequential candidate naming
        self.evaluation_history = []  # list of eval calls for the single problem
        self.best_score = float("-inf")  # best maximization score (higher is better)
        self.best_x = None  # best x found so far across all candidates. we will feed this into the LLM-generated codes to resume search from there.
        self.problem_index = None  # Captured during first evaluation

    def evaluate(self, candidate: dict[str, str], example: DataInst, **kwargs) -> tuple[float, Any, SideInfo]:
        """Evaluate code candidate on a single problem.

        Args:
            candidate: Dict with "code"
            example: Single dspy.Example with problem info

        Returns:
            Tuple of (score, output, feedback_dict)
        """
        code = candidate["code"]

        # Use a hash of the code for caching and seeding.
        code_hash = hashlib.md5(code.encode()).hexdigest()

        # For seeding, use a portion of the hash.
        seed_offset = int(code_hash[:7], 16)

        example_dict = example.toDict()
        problem_index = int(example["problem_index"])

        # Capture problem index for logging if not already set
        if self.problem_index is None:
            self.problem_index = problem_index

        # Check cache first (using hash to detect duplicate code)
        if code_hash in self.candidate_cache:
            candidate_id = self.code_hash_to_id[code_hash]
            print(f"Candidate {candidate_id} already executed. Using cached result.")
            return self.candidate_cache[code_hash]

        # Assign sequential candidate ID for new candidates
        candidate_id = self.next_candidate_id
        self.next_candidate_id += 1
        self.code_hash_to_id[code_hash] = candidate_id

        print(f"\n--- Processing Candidate {candidate_id} for problem_index={problem_index} ---")

        # Execute and evaluate the candidate
        result = self._evaluate_candidate(code, candidate_id, code_hash, example_dict, problem_index, seed_offset)
        return result

    def _evaluate_candidate(
        self,
        code: str,
        candidate_id: int,
        code_hash: str,
        example_dict: dict,
        problem_index: int,
        seed_offset: int,
    ) -> tuple[float, Any, SideInfo]:
        """Evaluate a single candidate on a problem."""
        problem_description = example_dict["problem_description"]
        score = -99999

        # problems list contains already-instantiated problem objects
        function = problems[problem_index]
        problem_config = problem_configs[problem_index]  # For metadata like name

        print(f"\n{'=' * 70}")
        print(f"Evaluating candidate {candidate_id} for problem_index={problem_index}")
        print(f"{'=' * 70}")

        # Track evaluation calls for this candidate
        evaluation_count = [0]
        previous_best_score = self.best_score
        best_score_during_optimization = [-99999]
        best_x_during_optimization = [None]
        all_scores_during_optimization = []  # Track all scores for mean/EMA computation

        def objective_function(x):
            if evaluation_count[0] > self.evaluation_budget:
                raise ValueError(f"Evaluation budget exceeded: {evaluation_count[0]} > {self.evaluation_budget}")
            evaluation_count[0] += 1
            # Convert to maximization problem for GEPA
            current_score = -function.do_evaluate(np.array(x))

            # Track all scores for this candidate
            all_scores_during_optimization.append(current_score)

            # Track best maximization score and x found by this candidate
            if current_score > best_score_during_optimization[0]:
                best_score_during_optimization[0] = current_score

            # Update global best maximization score
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_x = np.array(x).copy()

            # Log evaluation call
            self.evaluation_history.append(
                {
                    "score": current_score,
                    "best_score": self.best_score,
                    "candidate_number": candidate_id,
                    "best_x": self.best_x,
                }
            )
            self.save()

            return current_score

        # Execute the code
        execution_data = execute_code(
            code,
            {
                "dim": function.dim,
                "objective_function": objective_function,
                "total_evaluation_budgets": self.evaluation_budget,
                "bounds": function.bounds,
                "seed": self.base_seed + seed_offset,
                "prev_best_x": self.best_x.tolist() if self.best_x is not None else None,
            },
            timeout=self.timeout,
            seed=self.base_seed + seed_offset,
        )

        code_results = execution_data["results"]
        code_prints = execution_data["output"]
        code_logs = execution_data["logs"]
        code_error = execution_data["error"]
        # Truncate to avoid token limits in LLM prompts (keep first 2000 + last 2000 if > 4000 chars)
        code_prints = (
            code_prints[:2000] + "\n...[truncated]...\n" + code_prints[-2000:]
            if len(code_prints) > 4000
            else code_prints
        )
        code_logs = (
            code_logs[:2000] + "\n...[truncated]...\n" + code_logs[-2000:] if len(code_logs) > 4000 else code_logs
        )

        # Initialize score
        score = -99999
        candidate_x = None

        # Check for errors in the execution context
        if "x" not in code_results.keys() or code_results["x"] is None:
            code_error += "x not found in the global variables. "

        if code_error and ("TimeLimitError" in code_error or "timeout" in code_error.lower()):
            code_error += "Timeout or execution error occurred. "

        # ALWAYS use the best score found during optimization if any calls were made
        if best_score_during_optimization[0] > -99999:
            score = best_score_during_optimization[0]
            candidate_x = best_x_during_optimization[0]

            if evaluation_count[0] > self.evaluation_budget:
                code_error += f" Evaluation budget exceeded: {evaluation_count[0]} > {self.evaluation_budget}. "
                print(f"Budget exceeded ({evaluation_count[0]}), using best score found within budget.")

            code_error += f"Using best score found during {evaluation_count[0]} optimization calls: {best_score_during_optimization[0]:.6f}. "
            print(f"Using best score from optimization: {best_score_during_optimization[0]:.6f}")

            if score > self.best_score:
                self.best_score = score
        else:
            # If no objective_function calls were made, it's an error
            code_error += "No objective_function calls were made. Evaluation failed. "
            print("No evaluations made. Score set to -99999.")
            if code_error.strip():
                print(f"  Error: {code_error}")
            if code_prints.strip():
                print(f"  Output: {code_prints[:500]}")  # Truncate long output

        # Compute score metrics for this candidate
        mean_score = None
        ema_score_fixed = None  # alpha = 0.1
        ema_score_adaptive = None  # alpha = 2/(n+1)

        if all_scores_during_optimization:
            mean_score = float(np.mean(all_scores_during_optimization))

            # Compute EMA with fixed alpha = 0.1
            alpha_fixed = 0.1
            ema_fixed = all_scores_during_optimization[0]
            for s in all_scores_during_optimization[1:]:
                ema_fixed = alpha_fixed * s + (1 - alpha_fixed) * ema_fixed
            ema_score_fixed = float(ema_fixed)

            # Compute EMA with adaptive alpha = 2/(n+1)
            n = len(all_scores_during_optimization)
            alpha_adaptive = 2.0 / (n + 1)
            ema_adaptive = all_scores_during_optimization[0]
            for s in all_scores_during_optimization[1:]:
                ema_adaptive = alpha_adaptive * s + (1 - alpha_adaptive) * ema_adaptive
            ema_score_adaptive = float(ema_adaptive)

        # Build side_info and output
        side_info = {
            "scores": {
                "max_score": score,
                "mean_score": mean_score,
                "ema_score_fixed": ema_score_fixed,
                "ema_score_adaptive": ema_score_adaptive,
                "score_improvement_from_previous_best": score - previous_best_score
                if previous_best_score is not None
                else 0,
            },
            "Input": problem_description,
            "X": candidate_x,
            "Prints": code_prints,
            "Logs": code_logs,
            "Error": code_error,
        }

        output = side_info.copy()
        output["code"] = code
        output["problem_name"] = problem_config["name"]
        if candidate_x is not None:
            # Handle dict, list, or numpy array and convert to space-separated string
            vals = candidate_x.values() if isinstance(candidate_x, dict) else candidate_x
            output["X"] = " ".join(map(str, np.atleast_1d(vals).ravel()))
        else:
            output["X"] = "x not found"

        result = (score, output, side_info)
        self.candidate_cache[code_hash] = result
        self.save()

        return result

    def save(self, verbose: bool = False):
        """Save evaluation history to JSON files."""
        if self.log_dir is None or self.problem_index is None:
            if verbose and self.log_dir is None:
                print("No log directory set, cannot save trajectory")
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Save evaluation call history for the single problem
        filename = self.log_dir / f"evaluation_call_history_problem_{self.problem_index}.json"
        try:
            temp_filename = filename.with_suffix(".json.tmp")
            with open(temp_filename, "w", encoding="utf-8") as f:
                json.dump(self.evaluation_history, f, indent=2, default=_json_default)
            temp_filename.replace(filename)
            if verbose:
                print(f"Saved evaluation call history for problem_{self.problem_index} to {filename}")
        except Exception as e:
            print(f"Warning: Failed to save evaluation history: {e}")

    def save_trajectory(self, verbose: bool = True):
        """Backwards compatibility for save_trajectory."""
        self.save(verbose=verbose)


# Backwards compatibility: factory function that returns the evaluate method
def create_fitness_function(timeout=300, evaluation_budget=100, log_dir=None, base_seed=0):
    """Create fitness function that evaluates code with optional refinement.

    Args:
        timeout: Timeout in seconds for code execution
        evaluation_budget: Maximum number of evaluation calls allowed per candidate
        log_dir: Directory to save trajectory files
        base_seed: Base seed for reproducibility

    Returns:
        Fitness function compatible with GEPA
    """
    evaluator = FitnessEvaluator(
        timeout=timeout,
        evaluation_budget=evaluation_budget,
        log_dir=log_dir,
        base_seed=base_seed,
    )
    # Attach save method to the function for backwards compatibility
    evaluator.evaluate.save = evaluator.save
    evaluator.evaluate.save_trajectory = evaluator.save_trajectory
    evaluator.evaluate._evaluator = evaluator
    return evaluator.evaluate


if __name__ == "__main__":
    code_to_run = """
import optuna
import numpy as np  # <--- 1. Import numpy
from examples.bbox_opt.evalset import Rastrigin

def create_objective(problem):
    print("Bounds: ", problem.bounds)
    def objective(trial):
        x = []
        for i in range(problem.dim):
            val = trial.suggest_float(
                f"x{i}", problem.bounds[i][0], problem.bounds[i][1]
            )
            x.append(val)
        
        # <--- 2. CONVERT TO NUMPY ARRAY BEFORE EVALUATING --->
        x_array = np.array(x)
        # Convert to maximization for testing consistency
        result = -problem.do_evaluate(x_array)
        
        return result

    return objective

def main():
    problem = Rastrigin(dim)
    print(dim)
    objective = create_objective(problem)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    # Store results in global variables so 'execute_code_string' can capture them
    global x, y
    x = study.best_trial.params
    x = np.array(x)
    y = study.best_trial.value
    
    print("Best x: ", x)
    print("Best y: ", y)

if __name__ == "__main__":
    main()
"""

    # Run the corrected code
    execution_data = execute_code(code_to_run, {"dim": 8}, timeout=300)

    print("code_results: ", execution_data["results"]["x"])

    print("--------------------------------")
    print("code_prints: ", execution_data["output"][:500])

    print("--------------------------------")
    print("code_logs: ", execution_data["logs"][:500])
    print("--------------------------------")

    if execution_data["error"]:
        print("code_error: ", execution_data["error"])
    else:
        print("code_error: None")

    problem = Rastrigin(dim=8)
    print(problem.do_evaluate(np.array(execution_data["results"]["x"])))
