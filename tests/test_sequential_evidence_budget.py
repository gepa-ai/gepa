import pytest

from gepa.core.adapter import EvaluationBatch
from gepa.evidence.adapter import GuardedAdapter
from gepa.evidence.budget import BudgetExhaustedError, MetricBudgetGuard
from gepa.evidence.config import SequentialEvidenceConfig
from gepa.evidence.persistence import EvidenceStore


def test_reservations_prevent_overshoot_and_replays_never_double_charge():
    guard = MetricBudgetGuard(5)
    token = guard.reserve("first", 4, "seed")
    with pytest.raises(BudgetExhaustedError):
        guard.reserve("second", 2, "proposal")
    guard.complete(token, 3)
    assert guard.used == 3 and guard.available == 2
    guard.complete(token, 3)
    assert guard.used == 3
    with pytest.raises(ValueError):
        guard.complete(token, 2)
    with pytest.raises(ValueError):
        guard.reserve("first", 4, "seed")


def test_failed_and_abandoned_calls_charge_declared_upper_bound():
    guard = MetricBudgetGuard(10)
    guard.reserve("in-flight", 4, "seed")
    snapshot = guard.to_dict()
    restored = MetricBudgetGuard.from_dict(snapshot, limit=12)
    assert restored.used == 4 and restored.available == 8
    assert restored.records["in-flight"].status == "abandoned"
    restored.fail(restored.reserve("failed", 3, "round"), "provider failure")
    assert restored.used == 7
    with pytest.raises(ValueError):
        MetricBudgetGuard.from_dict(snapshot, limit=9)


@pytest.mark.parametrize("count", [-1, True, 1.5, 5])
def test_invalid_adapter_cost_fails_closed(count):
    guard = MetricBudgetGuard(5)
    token = guard.reserve("call", 4, "proposal")
    with pytest.raises(ValueError):
        guard.complete(token, count)
    assert guard.used == 4
    assert guard.records[token].status == "failed"


def test_store_is_atomic_checked_and_context_bound(tmp_path):
    store = EvidenceStore(str(tmp_path), "context")
    store.save({"guard": {"used": 4}, "comparisons": []})
    assert store.load()["guard"]["used"] == 4
    with pytest.raises(ValueError, match="context"):
        EvidenceStore(str(tmp_path), "other").load()
    path = tmp_path / "sequential_evidence.json"
    path.write_text(path.read_text().replace('"used":4', '"used":5'))
    with pytest.raises(ValueError, match="checksum"):
        store.load()


class Adapter:
    propose_new_texts = None
    def __init__(self, cost=None, fail=False):
        self.calls = 0
        self.cost = cost
        self.fail = fail
    def evaluate(self, batch, candidate, capture_traces=False):
        self.calls += len(batch)
        if self.fail:
            raise RuntimeError("provider")
        return EvaluationBatch(outputs=list(batch), scores=[0.5] * len(batch), num_metric_calls=self.cost)


def test_proxy_guards_seed_before_state_exists_and_preserves_raw_results():
    raw = Adapter(cost=2)
    guard = MetricBudgetGuard(4)
    proxy = GuardedAdapter(raw, guard, SequentialEvidenceConfig("v1", 100, max_calls_per_example=2))
    result = proxy.evaluate(["x"], {"x": "y"})
    assert result.scores == [0.5] and result.num_metric_calls == 2
    assert guard.used == 2
    proxy.evaluate(["a"], {"x": "y"})
    with pytest.raises(BudgetExhaustedError):
        proxy.evaluate(["b"], {"x": "y"})
    assert raw.calls == 2 and guard.used == 4


def test_proxy_failure_charges_attempts_and_stops_batch_at_remaining_cap():
    raw = Adapter(fail=True)
    guard = MetricBudgetGuard(3)
    proxy = GuardedAdapter(raw, guard, SequentialEvidenceConfig("v1", 100, max_calls_per_example=2))
    with pytest.raises(RuntimeError, match="provider"):
        proxy.evaluate(["x"], {"x": "y"})
    assert guard.used == 2
    with pytest.raises(BudgetExhaustedError):
        proxy.batch_evaluate([({"x": "y"}, ["a"]), ({"x": "z"}, ["b"])])
    assert raw.calls == 1


def test_directory_fsync_unsupported_filesystem_preserves_atomic_artifact(tmp_path, monkeypatch):
    import errno
    import os
    real_fsync = os.fsync
    def fsync(descriptor):
        import stat
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.EINVAL, "directory fsync unsupported")
        return real_fsync(descriptor)
    monkeypatch.setattr(os, "fsync", fsync)
    store = EvidenceStore(str(tmp_path), "v1")
    store.save({"cost": 3})
    assert store.load() == {"cost": 3}


def test_failed_atomic_replace_preserves_previous_artifact_and_removes_temp(tmp_path, monkeypatch):
    import os
    store = EvidenceStore(str(tmp_path), "v1")
    store.save({"cost": 3})
    def fail_replace(source, target):
        raise OSError("disk write failure")
    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk"):
        store.save({"cost": 9})
    assert store.load() == {"cost": 3}
    assert list(tmp_path.glob(".sequential-evidence-*")) == []
