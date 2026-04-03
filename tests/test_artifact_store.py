# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Tests for gepa.core.artifact_store — ArtifactStore protocol and implementations."""

from __future__ import annotations

from pathlib import Path

from gepa.core.artifact_store import (
    ArtifactStore,
    ArtifactStoreCallback,
    FileSystemArtifactStore,
    InMemoryArtifactStore,
)


class TestInMemoryArtifactStore:
    def test_protocol_compliance(self) -> None:
        store = InMemoryArtifactStore()
        assert isinstance(store, ArtifactStore)

    def test_save_and_load_candidate(self) -> None:
        store = InMemoryArtifactStore()
        candidate = {"instructions": "Be helpful"}
        store.save_candidate(0, candidate, parents=[], iteration=0)
        loaded = store.load_candidate(0)
        assert loaded == candidate

    def test_loaded_candidate_is_copy(self) -> None:
        store = InMemoryArtifactStore()
        candidate = {"instructions": "Be helpful"}
        store.save_candidate(0, candidate, parents=[], iteration=0)
        loaded = store.load_candidate(0)
        loaded["instructions"] = "modified"
        assert store.load_candidate(0)["instructions"] == "Be helpful"

    def test_lineage(self) -> None:
        store = InMemoryArtifactStore()
        store.save_candidate(0, {"a": "seed"}, parents=[], iteration=0)
        store.save_candidate(1, {"a": "child"}, parents=[0], iteration=1)
        store.save_candidate(2, {"a": "grandchild"}, parents=[1], iteration=2)

        assert store.get_lineage(0) == []
        assert store.get_lineage(1) == [0]
        assert store.get_lineage(2) == [1]

    def test_list_candidates(self) -> None:
        store = InMemoryArtifactStore()
        store.save_candidate(0, {"a": "x"}, parents=[], iteration=0)
        store.save_candidate(3, {"a": "y"}, parents=[0], iteration=1)
        assert store.list_candidates() == [0, 3]

    def test_save_evaluation(self) -> None:
        store = InMemoryArtifactStore()
        store.save_evaluation(0, data_id="q1", score=0.8, side_info={"time": 1.2})
        assert len(store._evaluations) == 1
        assert store._evaluations[0]["score"] == 0.8

    def test_save_candidate_with_metadata(self) -> None:
        store = InMemoryArtifactStore()
        store.save_candidate(0, {"a": "x"}, parents=[], iteration=0, metadata={"tag": "seed"})
        assert store._metadata[0]["tag"] == "seed"


class TestFileSystemArtifactStore:
    def test_protocol_compliance(self, tmp_path: Path) -> None:
        store = FileSystemArtifactStore(root=tmp_path)
        assert isinstance(store, ArtifactStore)

    def test_round_trip(self, tmp_path: Path) -> None:
        store = FileSystemArtifactStore(root=tmp_path)
        candidate = {"instructions": "Be concise", "system": "You are an assistant"}
        store.save_candidate(0, candidate, parents=[], iteration=0)
        loaded = store.load_candidate(0)
        assert loaded == candidate

    def test_lineage_round_trip(self, tmp_path: Path) -> None:
        store = FileSystemArtifactStore(root=tmp_path)
        store.save_candidate(0, {"a": "seed"}, parents=[], iteration=0)
        store.save_candidate(1, {"a": "child"}, parents=[0], iteration=1)
        assert store.get_lineage(0) == []
        assert store.get_lineage(1) == [0]

    def test_list_candidates(self, tmp_path: Path) -> None:
        store = FileSystemArtifactStore(root=tmp_path)
        store.save_candidate(0, {"a": "x"}, parents=[], iteration=0)
        store.save_candidate(2, {"a": "y"}, parents=[0], iteration=1)
        assert store.list_candidates() == [0, 2]

    def test_save_evaluation_creates_file(self, tmp_path: Path) -> None:
        store = FileSystemArtifactStore(root=tmp_path)
        store.save_candidate(0, {"a": "x"}, parents=[], iteration=0)
        store.save_evaluation(0, data_id="q1", score=0.9)
        eval_dir = tmp_path / "candidates" / "c00000" / "evals"
        assert eval_dir.exists()
        eval_files = list(eval_dir.glob("*.json"))
        assert len(eval_files) == 1


class TestArtifactStoreCallback:
    def test_on_valset_evaluated_saves_candidate_and_evals(self) -> None:
        store = InMemoryArtifactStore()
        callback = ArtifactStoreCallback(store)

        event = {
            "iteration": 1,
            "candidate_idx": 0,
            "candidate": {"instructions": "Be helpful"},
            "scores_by_val_id": {"q1": 0.8, "q2": 0.9},
            "average_score": 0.85,
            "num_examples_evaluated": 2,
            "total_valset_size": 2,
            "parent_ids": [],
            "is_best_program": True,
            "outputs_by_val_id": None,
        }

        callback.on_valset_evaluated(event)

        assert store.list_candidates() == [0]
        assert store.load_candidate(0) == {"instructions": "Be helpful"}
        assert len(store._evaluations) == 2
