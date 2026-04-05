"""Terrarium: Common evaluation infrastructure for evolving/auto-research systems.

Quickstart::

    from terrarium import get_task, run

    result = run("circle_packing", "my_adapter.py", max_evals=100)
    print(result.best_score, result.best_candidate[:80])
"""

from terrarium.adapter import Adapter, Result
from terrarium.registry import get_task, list_tasks, register_task
from terrarium.runner import run
from terrarium.task import Example, Task

__all__ = [
    "Adapter",
    "Example",
    "Result",
    "Task",
    "get_task",
    "list_tasks",
    "register_task",
    "run",
]
