"""``gepa.omni`` — backend-pluggable optimization for text artifacts.

A single :func:`optimize_anything` call dispatches to any compatible backend
(GEPA, Claude Code, Meta-Harness, …) over a shared eval-server contract.

Quickstart::

    from gepa.omni import optimize_anything, Task

    def evaluate(candidate: str) -> tuple[float, dict]:
        return score(candidate), {}

    result = optimize_anything(
        task=Task(name="t", initial_candidate="...", objective="..."),
        evaluate=evaluate,
        backend="gepa",
        config={"reflection": {"reflection_lm": "openai/gpt-5"}},
        max_evals=200,
    )
"""

from gepa.omni.api import optimize_anything
from gepa.omni.backend import Backend, Result
from gepa.omni.budget import BudgetExhausted, BudgetTracker
from gepa.omni.eval_server import EvalServer
from gepa.omni.registry import (
    get_backend_cls,
    get_task,
    list_backends,
    list_tasks,
    make_backend,
    register_backend,
    register_task,
    register_task_factory,
)
from gepa.omni.task import EvalFn, Example, Task

__all__ = [
    "Backend",
    "BudgetExhausted",
    "BudgetTracker",
    "EvalFn",
    "EvalServer",
    "Example",
    "Result",
    "Task",
    "get_backend_cls",
    "get_task",
    "list_backends",
    "list_tasks",
    "make_backend",
    "optimize_anything",
    "register_backend",
    "register_task",
    "register_task_factory",
]
