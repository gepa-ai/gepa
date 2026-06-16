"""Engine infrastructure for :mod:`gepa.optimize_anything`.

Most users should import from :mod:`gepa.optimize_anything`. This package holds
the shared engine protocol, eval server, budget tracker, and registries used by
that public API.
"""

from gepa.oa.budget import BudgetExhausted, BudgetTracker
from gepa.oa.config import OptimizeAnythingConfig
from gepa.oa.engine import Engine, Result
from gepa.oa.eval_server import EvalServer
from gepa.oa.registry import (
    get_engine_cls,
    get_task,
    list_engines,
    list_tasks,
    register_engine,
    register_task,
    register_task_factory,
)
from gepa.oa.task import EvalFn, Task

__all__ = [
    "BudgetExhausted",
    "BudgetTracker",
    "Engine",
    "EvalFn",
    "EvalServer",
    "OptimizeAnythingConfig",
    "Result",
    "Task",
    "get_engine_cls",
    "get_task",
    "list_engines",
    "list_tasks",
    "register_engine",
    "register_task",
    "register_task_factory",
]
