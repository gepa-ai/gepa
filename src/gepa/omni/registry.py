"""Registries for backends and tasks.

Backends are registered by string name → class. The api resolves
``backend="gepa"`` → ``GepaBackend`` and instantiates with the user's config.

Tasks may also be registered for Hydra-driven runs. Task factories return a
fully-populated :class:`Task` (with ``eval_fn`` set when applicable) so a
single ``task=<name>`` resolves both the task spec and its evaluator.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gepa.omni.backend import Backend
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


def make_backend(spec: str | Backend, config: dict | None = None) -> Backend:
    """Resolve a backend spec into an instance.

    - ``spec`` is a string: look up the registered class and instantiate with
      ``**config``.
    - ``spec`` is already a Backend instance: return as-is. ``config`` is
      ignored (the caller already configured the backend).
    """
    from gepa.omni.backend import Backend as _Backend  # noqa: F401 — runtime check

    if isinstance(spec, str):
        cls = get_backend_cls(spec)
        return cls(**(config or {}))
    return spec
