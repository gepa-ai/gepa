from typing import Any, Sequence

import numpy as np

from examples.polynomial.evalset import problems
from gepa.optimize_anything import SideInfo
from gepa.utils.code_execution import execute_code as _execute_code, ExecutionMode


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
    Create fitness function that evaluates code candidates for GEPA.
    """
    evaluator = Evaluator()

    def fitness_fn(candidate: dict[str, str], batch: Sequence[Any], **kwargs) -> list[tuple[float, Any, SideInfo]]:
        code = candidate["code"]
        results = []

        for example in batch:
            example_dict = example.toDict()
            problem_name = example["problem_name"]
            problem_description = example_dict["problem_description"]
            score = -99999
            x = None
            y_dist = None

            function = problems[problem_name]
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
            global_calls, local_calls = evaluator.get_num_evaluation_calls()

            if "x" not in code_results or code_results["x"] is None:
                code_error += "\n'x' not found in global variables after execution."
            else:
                x = code_results["x"]
                try:
                    x_array = np.array(x)
                    score = -function.do_evaluate(x_array)
                    if np.isnan(score):
                        score = -99999
                        code_error += "\nEvaluation resulted in NaN."

                    y_dist = np.abs(function.fmin + score)
                except Exception as e:
                    code_error += f"\nError evaluating candidate result: {e}"

            side_info = {
                "scores": {"score": score},
                "Input": {"problem_description": problem_description},
                "code_side_info": {
                    "X": str(x),
                    "Prints": code_prints,
                    "Logs": code_logs,
                    "Error": code_error,
                    "total_calls": global_calls,
                    "candidate_calls": local_calls,
                },
            }

            output = side_info.copy()
            output.update(
                {
                    "code": code,
                    "problem_name": problem_name,
                    "y_dist": y_dist,
                }
            )

            results.append((score, output, side_info))

        return results

    return fitness_fn
