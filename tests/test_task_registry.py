from __future__ import annotations

import json

import pytest

from gepa.oa import registry
from gepa.oa.registry import get_task, list_tasks, register_task
from gepa.oa.task import Task
from gepa.optimize_anything import register_task_factory


@pytest.fixture(autouse=True)
def isolated_task_registry(monkeypatch):
    monkeypatch.setattr(registry, "_TASKS", {})
    monkeypatch.setattr(registry, "_TASK_FACTORIES", {})


def test_file_backed_factory_can_retry_after_input_becomes_available(tmp_path):
    dataset = tmp_path / "examples.json"
    calls = []

    def load_task():
        calls.append(1)
        return Task(name="file-backed", train_set=json.loads(dataset.read_text()))

    register_task_factory("file-backed", load_task)
    with pytest.raises(FileNotFoundError):
        get_task("file-backed")

    assert list_tasks() == ["file-backed"]
    dataset.write_text('["ready"]')
    result = get_task("file-backed")
    assert result.train_set == ["ready"]
    dataset.unlink()
    assert get_task("file-backed") is result
    assert len(calls) == 2


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_repeated_factory_failures_preserve_error_and_name_registration(error_type):
    calls = []

    def fail():
        calls.append(1)
        raise error_type("input unavailable")

    register_task_factory("unavailable", fail)
    for _ in range(2):
        with pytest.raises(error_type, match="input unavailable"):
            get_task("unavailable")

    assert len(calls) == 2
    assert list_tasks() == ["unavailable"]
    with pytest.raises(ValueError, match="already registered"):
        register_task(Task(name="unavailable"))


def test_successful_factory_is_still_cached_and_protected_from_duplicates():
    calls = []
    task = Task(name="cached", train_set=["example"])

    def load():
        calls.append(1)
        return task

    register_task_factory("cached", load)
    assert get_task("cached") is task
    assert get_task("cached") is task
    assert len(calls) == 1
    assert list_tasks() == ["cached"]
    with pytest.raises(ValueError, match="already registered"):
        register_task_factory("cached", load)


def test_factory_can_still_register_the_task_it_returns():
    task = Task(name="self-registering")
    register_task_factory(task.name, lambda: register_task(task))

    assert get_task(task.name) is task
    assert get_task(task.name) is task
    assert list_tasks() == [task.name]
