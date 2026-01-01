import io
import signal
import traceback
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Sequence

import numpy as np

from examples.polynomial.evalset import problems
from gepa.optimize_anything import SideInfo


class TimeLimitError(Exception):
    pass


def _alarm_handler(signum, frame):
    raise TimeLimitError("Time Limit Exceeded")


def execute_code(code_string, global_vars=None, timeout=5):
    f_out = io.StringIO()
    f_err = io.StringIO()

    if global_vars is None:
        context = {"__name__": "__main__"}
    else:
        context = global_vars.copy()
        context["__name__"] = "__main__"

    error = ""
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(timeout)

    try:
        with redirect_stdout(f_out), redirect_stderr(f_err):
            exec(code_string, context)
    except TimeLimitError:
        error = f"TimeLimitError: Code execution exceeded {timeout} seconds."
    except Exception as e:
        error = str(e) + "\n" + traceback.format_exc()
    finally:
        signal.alarm(0)

    return {
        "output": f_out.getvalue(),
        "logs": f_err.getvalue(),
        "results": context,
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
