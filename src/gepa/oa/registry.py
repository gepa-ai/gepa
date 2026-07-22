"""Registries for optimization engines and tasks.

Engines register a string name to a class. The public
``gepa.optimize_anything`` API resolves ``engine="gepa"`` to that class and
instantiates it with :class:`OptimizeAnythingConfig`.

Tasks may also be registered by name for convenience.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gepa.oa.task import Task

_ENGINES: dict[str, type] = {}
_TASKS: dict[str, Task] = {}
_TASK_FACTORIES: dict[str, Callable[[], Task]] = {}


def register_engine(name: str, cls: type) -> type:
    """Register an engine class under ``name``."""
    _ENGINES[name] = cls
    return cls


def get_engine_cls(name: str) -> type:
    """Resolve an engine class by name.

    On first lookup, eagerly imports built-in engines so they self-register.
    """
    if not _ENGINES:
        _load_builtin_engines()
    if name not in _ENGINES:
        raise KeyError(f"Unknown engine '{name}'. Available: {sorted(_ENGINES)}")
    return _ENGINES[name]


def list_engines() -> list[str]:
    if not _ENGINES:
        _load_builtin_engines()
    return sorted(_ENGINES)


def _load_builtin_engines() -> None:
    import gepa.oa.engines  # noqa: F401  triggers registration


def register_task(task: Task) -> Task:
    if task.name in _TASKS or task.name in _TASK_FACTORIES:
        raise ValueError(f"Task '{task.name}' is already registered")
    _TASKS[task.name] = task
    return task


def register_task_factory(name: str, factory: Callable[[], Task]) -> None:
    if name in _TASKS or name in _TASK_FACTORIES:
        raise ValueError(f"Task '{name}' is already registered")
    _TASK_FACTORIES[name] = factory


def get_task(name: str) -> Task:
    if name in _TASK_FACTORIES:
        task = _TASK_FACTORIES.pop(name)()
        _TASKS[name] = task
        return task
    if name not in _TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(_TASKS | _TASK_FACTORIES.keys())}")
    return _TASKS[name]


def list_tasks() -> list[str]:
    return sorted(set(_TASKS) | set(_TASK_FACTORIES))
