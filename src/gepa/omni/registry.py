"""Registries for backends and tasks.

Backends register a string name → class; the api resolves ``backend="gepa"``
to the class and calls ``cls.from_config(omni_config)``.

Tasks may also be registered by name for convenience.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gepa.omni.task import Task

_BACKENDS: dict[str, type] = {}
_TASKS: dict[str, Task] = {}
_TASK_FACTORIES: dict[str, Callable[[], Task]] = {}


def register_backend(name: str, cls: type) -> type:
    """Register a backend class under ``name``."""
    _BACKENDS[name] = cls
    return cls


def get_backend_cls(name: str) -> type:
    """Resolve a backend class by name.

    On first lookup, eagerly imports built-in backends so they self-register.
    """
    if not _BACKENDS:
        _load_builtin_backends()
    if name not in _BACKENDS:
        raise KeyError(f"Unknown backend '{name}'. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


def list_backends() -> list[str]:
    if not _BACKENDS:
        _load_builtin_backends()
    return sorted(_BACKENDS)


def _load_builtin_backends() -> None:
    import gepa.omni.backends  # noqa: F401  triggers registration


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
