import importlib

# Functions used in the nonparametric/parametric comparison (Table 2)
problem_configs = [
    {"name": "Ackley", "dim": 11, "int": None, "res": None},
    {"name": "Ackley", "dim": 3, "int": None, "res": 1},
    {"name": "Adjiman", "dim": 2, "int": None, "res": None},
    {"name": "Alpine02", "dim": 2, "int": [0], "res": None},
    {"name": "CarromTable", "dim": 2, "int": [0], "res": None},
    {"name": "Csendes", "dim": 2, "int": None, "res": None},
    {"name": "DeflectedCorrugatedSpring", "dim": 4, "int": None, "res": None},
    {"name": "DeflectedCorrugatedSpring", "dim": 7, "int": None, "res": None},
    {"name": "Easom", "dim": 2, "int": None, "res": None},
    {"name": "Easom", "dim": 4, "int": None, "res": None},
    {"name": "Easom", "dim": 5, "int": None, "res": None},
    {"name": "Hartmann3", "dim": 3, "int": [0], "res": None},
    {"name": "Hartmann6", "dim": 6, "int": None, "res": 10},  # Already run
    {"name": "HelicalValley", "dim": 3, "int": None, "res": None},
    {"name": "LennardJones6", "dim": 6, "int": None, "res": None},  # Already run
    {"name": "McCourt01", "dim": 7, "int": None, "res": 10},
    {"name": "McCourt03", "dim": 9, "int": None, "res": None},  # Already run
    {"name": "McCourt06", "dim": 5, "int": None, "res": None},
    {"name": "McCourt07", "dim": 6, "int": None, "res": 12},
    {"name": "McCourt08", "dim": 4, "int": None, "res": None},
    {"name": "McCourt09", "dim": 3, "int": None, "res": None},
    {"name": "McCourt10", "dim": 8, "int": None, "res": None},  # Already run
    {"name": "McCourt11", "dim": 8, "int": None, "res": None},  # Already run
    {"name": "McCourt12", "dim": 7, "int": None, "res": None},
    {"name": "McCourt13", "dim": 3, "int": None, "res": None},
    {"name": "McCourt14", "dim": 3, "int": None, "res": None},
    {"name": "McCourt16", "dim": 4, "int": None, "res": None},
    {"name": "McCourt16", "dim": 4, "int": None, "res": 10},
    {"name": "McCourt17", "dim": 7, "int": None, "res": None},
    {"name": "McCourt18", "dim": 8, "int": None, "res": None},
    {"name": "McCourt19", "dim": 2, "int": None, "res": None},
    {"name": "McCourt20", "dim": 2, "int": None, "res": None},
    {"name": "McCourt23", "dim": 6, "int": None, "res": None},
    {"name": "McCourt26", "dim": 3, "int": None, "res": None},
    {"name": "McCourt28", "dim": 4, "int": None, "res": None},
    {"name": "Michalewicz", "dim": 4, "int": None, "res": None},
    {"name": "Michalewicz", "dim": 4, "int": None, "res": 20},
    {"name": "Michalewicz", "dim": 8, "int": None, "res": None},
    {"name": "Mishra06", "dim": 2, "int": None, "res": None},
    {"name": "Ned01", "dim": 2, "int": None, "res": None},
    {"name": "OddSquare", "dim": 2, "int": None, "res": None},
    {"name": "Parsopoulos", "dim": 2, "int": [0], "res": None},
    {"name": "Pinter", "dim": 2, "int": [0, 1], "res": None},
    {"name": "Plateau", "dim": 2, "int": None, "res": None},
    {"name": "Problem03", "dim": 1, "int": None, "res": None},
    {"name": "RosenbrockLog", "dim": 11, "int": None, "res": None},  # Already run
    {"name": "Sargan", "dim": 5, "int": None, "res": None},
    {"name": "Sargan", "dim": 2, "int": [0], "res": None},
    {"name": "Schwefel20", "dim": 2, "int": None, "res": None},
    {"name": "Schwefel20", "dim": 2, "int": [0], "res": None},
    {"name": "Schwefel36", "dim": 2, "int": None, "res": None},
    {"name": "Shekel05", "dim": 4, "int": None, "res": None},
    {"name": "Sphere", "dim": 7, "int": [0, 1, 2, 3, 4], "res": None},
    {"name": "StyblinskiTang", "dim": 5, "int": None, "res": None},
    {"name": "Tripod", "dim": 2, "int": None, "res": None},
    {"name": "Xor", "dim": 9, "int": None, "res": None},  # Already run
]

evalset_pack = importlib.import_module("examples.polynomial.evalset.evalset")


def create_problem_instance(problem_config):
    """Create a problem instance from a config dict.

    Args:
        problem_config: dict with keys: name, dim, int, res
            - name: class name in evalset (e.g., "Michalewicz")
            - dim: dimension of the problem
            - int: (not used, kept for reference)
            - res: discretization resolution for output values, or None

    Returns:
        A TestFunction instance (possibly wrapped with Discretizer)
    """
    # 1. Get the base problem class and instantiate it
    problem_class = getattr(evalset_pack, problem_config["name"])
    problem = problem_class(dim=problem_config["dim"])

    # 2. Wrap with Discretizer if res is specified
    #    This discretizes the OUTPUT (function values)
    if problem_config["res"] is not None:
        problem = evalset_pack.Discretizer(problem, res=problem_config["res"])

    return problem


problems = [create_problem_instance(config) for config in problem_configs]


# if __name__ == "__main__":
#     import numpy as np

#     for config in problems:
#         print("=" * 80)
#         print(f"Config: {config}")
#         print("=" * 80)

#         problem = create_problem_instance(config)

#         print(f"  Problem type: {type(problem).__name__}")
#         print(f"  Bounds: {problem.bounds}")

#         # Create a test point at midpoint of bounds
#         x = np.array(
#             [
#                 (problem.bounds[i][0] + problem.bounds[i][1]) / 2
#                 for i in range(problem.dim)
#             ]
#         )

#         # Evaluate
#         y = problem.evaluate(x)
#         print(f"  Test point x: {x}")
#         print(f"  Evaluation y: {y}")

#         # For discretized problems, verify output is discretized
#         if config["res"] is not None:
#             step = 1 / config["res"]
#             is_discretized = abs(y - round(y * config["res"]) / config["res"]) < 1e-10
#             print(f"  Output discretized (step={step}): {is_discretized}")

#         print()
