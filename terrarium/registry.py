"""Task registry for discovering and loading Terrarium tasks by name."""

from __future__ import annotations

from terrarium.task import Task

_REGISTRY: dict[str, Task] = {}


def register_task(task: Task) -> Task:
    """Register a task. Raises ValueError if a task with the same name exists."""
    if task.name in _REGISTRY:
        raise ValueError(f"Task '{task.name}' is already registered")
    _REGISTRY[task.name] = task
    return task


def get_task(name: str) -> Task:
    """Retrieve a registered task by name.

    If the built-in tasks haven't been loaded yet, this triggers a lazy import
    of ``terrarium.tasks`` so that all bundled tasks are available.
    """
    if not _REGISTRY:
        _load_builtin_tasks()
    if name not in _REGISTRY:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_tasks() -> list[str]:
    """Return sorted names of all registered tasks."""
    if not _REGISTRY:
        _load_builtin_tasks()
    return sorted(_REGISTRY)


def _load_builtin_tasks() -> None:
    """Import the tasks subpackage to trigger registration of built-in tasks."""
    import terrarium.tasks  # noqa: F401
