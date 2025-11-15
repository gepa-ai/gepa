from __future__ import annotations

from types import SimpleNamespace

from turbo_gepa.adapters.default_adapter import DefaultAdapter
from turbo_gepa.interfaces import Candidate, EvalResult


class _DummyOrchestrator:
    def __init__(self) -> None:
        self.config = SimpleNamespace(promote_objective="quality")
        self.run_id = "dummy-run"
        self.stop_reason = "timeout"
        self.evaluations_run = 3
        self._runtime_shards = [0.4, 1.0]
        self._north_star_prompt = None
        self._north_star_quality = None
        self._north_star_shard = None
        self.latest_results: dict[str, EvalResult] = {}
        self._candidates_by_fp: dict[str, Candidate] = {}

    def metrics_snapshot(self) -> dict:
        return {}


def test_best_prompt_prefers_highest_non_seed_shard() -> None:
    orchestrator = _DummyOrchestrator()
    seed = Candidate(text="Seed prompt", meta={"source": "seed"})
    child = Candidate(text="Child prompt", meta={"source": "mutation"})
    seed_fp = seed.fingerprint
    child_fp = child.fingerprint
    orchestrator._candidates_by_fp = {seed_fp: seed, child_fp: child}
    orchestrator.latest_results = {
        seed_fp: EvalResult(objectives={"quality": 0.88}, traces=[], n_examples=10, shard_fraction=1.0),
        child_fp: EvalResult(objectives={"quality": 0.82}, traces=[], n_examples=4, shard_fraction=0.4),
    }
    adapter = object.__new__(DefaultAdapter)
    metadata = DefaultAdapter._build_run_metadata(adapter, orchestrator, [])
    assert metadata["best_prompt"] == "Child prompt"
    assert metadata["best_quality"] == 0.82
    assert metadata["best_quality_shard"] == 0.4
