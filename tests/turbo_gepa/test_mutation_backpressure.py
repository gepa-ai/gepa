from collections import deque
import time

from turbo_gepa.config import Config
from turbo_gepa.interfaces import Candidate
from turbo_gepa.orchestrator import Orchestrator


def _make_orchestrator_stub(config: Config) -> Orchestrator:
    """Construct a lightweight orchestrator instance for private helper tests."""
    orch = object.__new__(Orchestrator)
    orch.config = config
    orch.queue = deque()
    orch._priority_queue = []
    orch._priority_counter = 0
    orch._pending_fingerprints = set()
    orch._inflight_fingerprints = set()
    orch._mutation_buffer_limit = 4
    orch._mutation_buffer = deque()
    orch._buffered_fingerprints = set()
    orch._queue_buffer_mult = 3.0
    orch._max_total_inflight = config.eval_concurrency
    orch._total_inflight = 0
    orch._examples_inflight = 0
    orch._effective_concurrency = config.eval_concurrency
    orch._mutation_min = 4
    orch._mutation_throttle = False
    orch._max_mutations_ceiling = config.max_mutations_per_round or 16
    orch._mutations_requested = 0
    orch._mutations_generated = 0
    orch._mutations_enqueued = 0
    orch._candidate_generations = {}
    orch._lineage_history = {}
    orch._sched_to_fingerprint = {}
    orch._promotion_pending = set()
    orch._candidate_island_meta = {}
    orch._migration_events = deque(maxlen=10)
    orch._high_rung_parent_quorum = 3
    orch._last_mutation_attempt = 0.0
    orch._candidate_eval_examples = {}
    orch._recent_final_straggler_events = deque(maxlen=64)
    orch._inflight_by_rung = {}
    orch._waiting_by_rung = {}
    orch._final_rung_index = 0
    orch._runtime_shards = [1.0]
    orch._cost_remaining = [1.0]
    orch._promotion_ema = [1.0]
    orch._final_success_ema = 1.0
    orch._priority_beta = 0.0
    orch._recent_delta_weight = 0.0
    orch._final_inflight_peak = 0
    orch._network_throttle_until = 0.0
    orch._last_network_warning = 0.0
    orch._final_rung_cap_floor = 1
    orch._final_rung_blocked_since = None
    orch._stop_reason = None
    orch._run_started_at = None
    orch._run_id = "test"
    orch.show_progress = False
    orch.logger = None  # Not used in these tests
    orch.metrics = type("MetricsStub", (), {"time_to_target_seconds": None})()
    orch.island_id = 0
    orch.scheduler = type("SchedulerStub", (), {"current_shard_index": lambda self, _c: 0})()
    orch._queue = orch.queue
    orch._total_inflight = 0
    orch._network_throttle_until = 0.0

    def enqueue_stub(candidates):
        for cand in candidates:
            orch._priority_counter += 1
            orch._priority_queue.append((0.0, 0, orch._priority_counter, cand))

    orch.enqueue = enqueue_stub  # type: ignore[assignment]
    return orch


def test_mutation_budget_pauses_when_buffer_and_queue_full():
    cfg = Config(eval_concurrency=12, max_mutations_per_round=32)
    orch = _make_orchestrator_stub(cfg)
    # Fill mutation buffer
    for idx in range(orch._mutation_buffer_limit):
        orch._mutation_buffer.append(Candidate(text=f"buf-{idx}"))
    # Simulate deep queue/backlog
    for idx in range(orch._queue_refill_threshold()):
        orch._priority_queue.append((0.0, 0, idx, Candidate(text=f"queued-{idx}")))

    assert orch._mutation_budget() == 0, "Mutation budget should stall when queue + buffer are saturated"


def test_buffer_does_not_refill_queue_during_network_throttle():
    cfg = Config(eval_concurrency=8, max_mutations_per_round=16)
    orch = _make_orchestrator_stub(cfg)
    orch._mutation_buffer.append(Candidate(text="staged"))
    hold_threshold = max(cfg.eval_concurrency, orch._queue_refill_threshold() // 2)
    for idx in range(hold_threshold):
        orch._priority_queue.append((0.0, 0, idx, Candidate(text=f"existing-{idx}")))
    orch._network_throttle_until = time.time() + 5.0

    orch._maybe_refill_queue()
    # Buffer should remain untouched because throttle is active and backlog high
    assert len(orch._mutation_buffer) == 1

    orch._maybe_refill_queue(force=True)
    assert len(orch._mutation_buffer) == 0
    assert len(orch._priority_queue) == hold_threshold + 1
