"""``gepa.omni`` — backend-pluggable optimization for text artifacts.

A single :func:`optimize_anything` call dispatches to any compatible backend
(GEPA, Claude Code, Meta-Harness, …) over a shared eval-server contract.

Quickstart::

    from gepa.omni import OmniConfig, Task, optimize_anything

    def evaluate(candidate: str) -> tuple[float, dict]:
        return score(candidate), {}

    result = optimize_anything(
        task=Task(name="t", initial_candidate="...", objective="..."),
        evaluate=evaluate,
        config=OmniConfig(
            backend="gepa",
            max_evals=200,
            config={"reflection": {"reflection_lm": "openai/gpt-5"}},
        ),
    )
"""

from gepa.omni.api import optimize_anything
from gepa.omni.backend import Backend, Result
from gepa.omni.budget import BudgetExhausted, BudgetTracker
from gepa.omni.config import OmniConfig
from gepa.omni.ensemble import (
    optimize_best_of,
    optimize_parallel,
    optimize_sequential,
    optimize_vote,
)
from gepa.omni.eval_server import EvalServer
from gepa.omni.registry import (
    get_backend_cls,
    get_task,
    list_backends,
    list_tasks,
    register_backend,
    register_task,
    register_task_factory,
)
from gepa.omni.task import EvalFn, Task

__all__ = [
    "Backend",
    "BudgetExhausted",
    "BudgetTracker",
    "EvalFn",
    "EvalServer",
    "OmniConfig",
    "Result",
    "Task",
    "get_backend_cls",
    "get_task",
    "list_backends",
    "list_tasks",
    "optimize_anything",
    "optimize_best_of",
    "optimize_parallel",
    "optimize_sequential",
    "optimize_vote",
    "register_backend",
    "register_task",
    "register_task_factory",
]
