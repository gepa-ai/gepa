"""Example ID collisions must not replace another example's payload."""

import pytest

from gepa.oa.budget import BudgetTracker
from gepa.oa.eval_server import EvalServer
from gepa.oa.task import Task


@pytest.mark.parametrize("split", ["train", "val"])
def test_fallback_id_collision_preserves_distinct_examples(split):
    examples = [
        {"id": f"{split}_1", "score": 0.0},
        {"id": f"{split}_1", "score": 1.0},
        {"id": f"{split}_1_1", "score": 2.0},
    ]
    task = Task(name="ids", **{f"{split}_set": examples})
    server = EvalServer(task, lambda candidate, example: (example["score"], {}), BudgetTracker())
    try:
        registered = list(server.iter_split(split))
        assert len({eid for eid, _ in registered}) == len(examples)
        assert all(actual is expected for (_, actual), expected in zip(registered, examples, strict=True))
        score, info = server.evaluate_examples("candidate", split=split)
        assert score == 1.0
        assert sorted(info["scores"].values()) == [0.0, 1.0, 2.0]
    finally:
        server.stop()


def test_cross_split_fallback_collision_preserves_training_payload():
    train = {"id": "val_0", "score": 0.0}
    val = {"id": "val_0", "score": 1.0}
    server = EvalServer(
        Task(name="ids", train_set=[train], val_set=[val]),
        lambda candidate, example: (example["score"], {}),
        BudgetTracker(),
    )
    try:
        assert list(server.iter_split("train"))[0][1] is train
        assert list(server.iter_split("val"))[0][1] is val
        assert server.evaluate_examples("candidate", split="train")[0] == 0.0
        assert server.validate("candidate")["val_score"] == 1.0
    finally:
        server.stop()


def test_same_object_across_splits_retains_its_stable_id():
    example = {"id": "shared", "score": 1.0}
    server = EvalServer(
        Task(name="ids", train_set=[example], val_set=[example]),
        lambda candidate, example: (example["score"], {}),
        BudgetTracker(),
    )
    try:
        assert list(server.iter_split("train")) == [("shared", example)]
        assert list(server.iter_split("val")) == [("shared", example)]
    finally:
        server.stop()
