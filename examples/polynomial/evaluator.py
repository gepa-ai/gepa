from typing import Any, Sequence
import numpy as np
import traceback

from gepa.optimize_anything import SideInfo
from gepa.utils.code_execution import execute_code as _execute_code, ExecutionMode, get_code_hash
from examples.polynomial.evalset import problems

# Track unique code variants with IDs
code_registry = {}  # hash -> id
code_counter = [0]  # Use list to allow mutation


def get_code_id(code: str) -> tuple[int, bool]:
    """
    Get or create an ID for a code variant.

    Returns:
        tuple: (code_id, is_new) where is_new=True if this is a new variant
    """
    code_hash = get_code_hash(code)

    is_new = code_hash not in code_registry
    if is_new:
        code_counter[0] += 1
        code_registry[code_hash] = code_counter[0]

    return code_registry[code_hash], is_new


def execute_code(code_string, global_vars=None, timeout=5):
    """Execute code with timeout support using shared code execution utility.
    
    Returns dict with keys matching the original API for backwards compatibility:
        - output: stdout
        - logs: stderr
        - results: execution context variables
        - error: error message (includes traceback)
    """
    result = _execute_code(
        code=code_string,
        timeout=timeout,
        mode=ExecutionMode.IN_PROCESS,
        global_vars=global_vars,
    )

    # Combine error and traceback for backwards compatibility
    error = result.error
    if result.traceback and result.traceback not in error:
        error = f"{error}\n{result.traceback}" if error else result.traceback

    return {
        "output": result.stdout,
        "logs": result.stderr,
        "results": result.variables,
        "error": error,
    }


class Evaluator:
    def __init__(self):
        self.problem = None
        self.global_evaluation_calls = 0
        self.local_evaluation_calls = 0

    def set_problem(self, problem):
        self.problem = problem
        self.local_evaluation_calls = 0

    def evaluate(self, x) -> float:
        self.global_evaluation_calls += 1
        self.local_evaluation_calls += 1
        return self.problem.do_evaluate(np.array(x))

    def get_num_evaluation_calls(self) -> tuple[int, int]:
        return self.global_evaluation_calls, self.local_evaluation_calls


def create_fitness_function(timeout=30):
    """
    Create fitness function that evaluates code with optional refinement.

    Args:
        timeout: Timeout in seconds for code execution

    Returns:
        Fitness function compatible with GEPA
    """

    evaluator = Evaluator()

    def fitness_fn(
        candidate: dict[str, str], batch: Sequence[Any], **kwargs
    ) -> list[tuple[float, Any, SideInfo]]:
        """
        Evaluate code candidate on batch of problems to minimize the polynomial.

        Args:
            candidate: Dict with "code"
            batch: Sequence of dspy.Example objects with problem description

        Returns:
            List of (score, output, feedback_dict) tuples
        """
        code = candidate["code"]
        code_id, _ = get_code_id(code)

        results = []

        for example in batch:
            example_dict = example.toDict()
            problem_name = example["problem_name"]
            problem_description = example_dict["problem_description"]
            y_dist = None
            score = -99999
            x = "x is not found in the global variables"

            function = problems[problem_name]

            print(f"\n{'=' * 70}")
            print(f"Evaluating code #{code_id} for {problem_name}")
            print(f"{'=' * 70}")

            evaluator.set_problem(function)

            execution_data = execute_code(
                code,
                {"dim": function.dim, "evaluator": evaluator},
                timeout=timeout,
            )

            code_results = execution_data["results"]
            code_prints = execution_data["output"]
            code_logs = execution_data["logs"]
            code_error = execution_data["error"]
            global_evaluation_calls, local_evaluation_calls = (
                evaluator.get_num_evaluation_calls()
            )
            print("global_evaluation_calls: ", global_evaluation_calls)
            print("local_evaluation_calls: ", local_evaluation_calls)

            if "x" not in code_results.keys() or code_results["x"] is None:
                code_error += "x not found in the global variables"
                print("code_error: ", code_error)
            else:
                x = code_results["x"]
                print("x: ", x)
                try:
                    x_array = np.array(x)
                    score = -function.do_evaluate(x_array)
                    if np.isnan(score):
                        code_error += "Score is nan. Returning -99999."
                        score = -99999
                        print("code_error: ", code_error)
                    print("Score: ", score)

                    true_minimum = function.min_loc
                    x_dist = np.linalg.norm(x_array - true_minimum)
                    y_dist = np.abs(function.fmin + score)
                    print(f"x Distance from true minimum: {x_dist}")
                    print(f"y Distance from true minimum: {y_dist}")
                except Exception as e:
                    full_traceback = traceback.format_exc()
                    code_error += f"\nError evaluating the code: {str(e)}\nTraceback:\n{full_traceback}"
                    print("Error evaluating the code: ", code_error)
                    traceback.print_exc()

            side_info = {
                "scores": {
                    "score": score,
                },
                "Input": {
                    "problem_description": problem_description,
                },
                "code_side_info": {
                    "X": x,
                    "Prints": {code_prints},
                    "Logs": {code_logs},
                    "Error": {code_error},
                    "Total evaluation calls so far": global_evaluation_calls,
                    "Num evaluation calls for this candidate": local_evaluation_calls,
                },
            }

            output = side_info.copy()
            output["code"] = code
            output["problem_name"] = problem_name
            output["y_dist"] = y_dist
            if type(x) is dict:
                x = [x[i] for i in x.keys()]
            elif type(x) is list:
                x = x
            else:
                x = [x]
            output["code_side_info"]["X"] = " ".join([str(i) for i in x])

            results.append((score, output, side_info))

        return results

    return fitness_fn


if __name__ == "__main__":
    code_to_run = """
import optuna
import numpy as np  # <--- 1. Import numpy
from examples.polynomial.evalset import problems, Rastrigin

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
        result = evaluator.evaluate(x_array)
        
        return result

    return objective

def main():
    problem = Rastrigin(dim)
    print(dim)
    objective = create_objective(problem)
    study = optuna.create_study(direction="minimize")
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
    problem_name = "Rastrigin"
    problem = problems[problem_name]
    evaluator = Evaluator(problem_name)
    execution_data = execute_code(
        code_to_run, {"dim": problem.dim, "evaluator": evaluator}, timeout=300
    )

    print("*****num_evaluation_calls: ", evaluator.get_num_evaluation_calls())

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
