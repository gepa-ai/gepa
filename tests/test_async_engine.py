"""Tests for the stage-decoupled asynchronous engine (gepa.async_engine)."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any

import pytest

import gepa
from gepa.async_engine import AsyncEngineConfig, StageConfig, optimize
from gepa.async_engine.scheduler import StagePool, WorkerController
from gepa.async_engine.stages import PatchJob, parse_patch_output, render_patch_prompt
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

N_TOKENS = 12


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"k\d+", text))


class TokenAdapter(GEPAAdapter):
    """Example ``i`` scores 1.0 when the instruction contains token ``k{i % N_TOKENS}``.

    Deterministic and thread-safe. Optional sleeps make stage latencies visible, and a
    concurrency probe records how many evaluate calls overlapped.
    """

    propose_new_texts = None

    def __init__(self, eval_sleep: float = 0.0, fail_on: str | None = None):
        self.eval_sleep = eval_sleep
        self.fail_on = fail_on
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.calls = 0

    def evaluate(self, batch, candidate, capture_traces=False):
        with self._lock:
            self._active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if self.eval_sleep:
                time.sleep(self.eval_sleep)
            text = candidate["instructions"]
            if self.fail_on is not None and self.fail_on in text:
                raise RuntimeError(f"evaluator rejected candidate containing {self.fail_on}")
            have = _tokens(text)
            scores = [1.0 if f"k{ex % N_TOKENS}" in have else 0.0 for ex in batch]
            trajectories = [{"example": ex, "score": s} for ex, s in zip(batch, scores, strict=True)]
            return EvaluationBatch(
                outputs=[f"out-{ex}" for ex in batch],
                scores=scores,
                trajectories=trajectories if capture_traces else None,
            )
        finally:
            with self._lock:
                self._active -= 1

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = [
            {
                "Inputs": f"example {t['example']}",
                "Generated Outputs": f"out-{t['example']}",
                "Feedback": "correct" if t["score"] >= 1.0 else f"add token k{t['example'] % N_TOKENS}",
            }
            for t in eval_batch.trajectories
        ]
        return dict.fromkeys(components_to_update, records)


class FakeReflectionLM:
    """Deterministic stand-in for the reflection LM. Also answers patch prompts."""

    def __init__(self, sleep: float = 0.0):
        self.sleep = sleep
        self.reflect_calls = 0
        self.patch_calls = 0
        self._lock = threading.Lock()

    def __call__(self, prompt) -> str:
        if self.sleep:
            time.sleep(self.sleep)
        text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        blocks = re.findall(r"```\n(.*?)\n```", text, flags=re.S)
        if "A worker proposed a revision" in text:
            with self._lock:
                self.patch_calls += 1
            base, revision, current = (_tokens(b) for b in blocks[:3])
            gained = revision - base - current
            if not gained:
                return "DISCARD"
            return "```\n" + " ".join(sorted(current | gained, key=lambda t: int(t[1:]))) + "\n```"
        with self._lock:
            self.reflect_calls += 1
        current = blocks[0] if blocks else ""
        wanted = [t for t in re.findall(r"add token (k\d+)", text) if t not in _tokens(current)]
        new = f"{current} {wanted[0]}".strip() if wanted else current
        return f"```\n{new}\n```"


def _token_proposer(candidate, reflective_dataset, components_to_update):
    out = {}
    for comp in components_to_update:
        current = candidate[comp]
        wanted = [
            m.group(1)
            for rec in reflective_dataset[comp]
            if (m := re.search(r"add token (k\d+)", rec["Feedback"])) and m.group(1) not in _tokens(current)
        ]
        out[comp] = f"{current} {wanted[0]}".strip() if wanted else current
    return out


SEED = {"instructions": "solve the task"}
TRAIN = list(range(24))
VAL = list(range(100, 124))


def _silent_logger():
    class L:
        def log(self, *a, **k):
            pass

    return L()


# ----------------------------------------------------------------------
# equivalence with the synchronous engine
# ----------------------------------------------------------------------


def test_sequential_config_reproduces_synchronous_run():
    kwargs: dict[str, Any] = {
        "seed_candidate": dict(SEED),
        "trainset": TRAIN,
        "valset": VAL,
        "custom_candidate_proposer": _token_proposer,
        "max_metric_calls": 400,
        "seed": 0,
        "logger": _silent_logger(),
        "skip_perfect_score": True,
    }
    sync = gepa.optimize(adapter=TokenAdapter(), **kwargs)
    asyn = optimize(adapter=TokenAdapter(), async_config=AsyncEngineConfig.sequential(), **kwargs)

    assert asyn.candidates == sync.candidates
    assert asyn.parents == sync.parents
    assert asyn.val_aggregate_scores == sync.val_aggregate_scores
    assert asyn.discovery_eval_counts == sync.discovery_eval_counts
    assert asyn.total_metric_calls == sync.total_metric_calls
    assert len(asyn.candidates) > 3, "the toy task should make progress"
    # One item in the pipeline at a time means stages never overlap.
    assert asyn.metadata["async"]["max_busy_stages"] == 1
    assert set(asyn.metadata["async"]["staleness_at_commit"]) == {"0"}


# ----------------------------------------------------------------------
# overlap
# ----------------------------------------------------------------------


def _timed_run(config: AsyncEngineConfig, budget: int = 260):
    adapter = TokenAdapter(eval_sleep=0.02)
    lm = FakeReflectionLM(sleep=0.08)
    t0 = time.monotonic()
    result = optimize(
        seed_candidate=dict(SEED),
        trainset=TRAIN,
        valset=VAL,
        adapter=adapter,
        reflection_lm=lm,
        max_metric_calls=budget,
        logger=_silent_logger(),
        async_config=config,
    )
    return result, time.monotonic() - t0, adapter, lm


def test_stages_overlap_and_shorten_wall_clock():
    seq_result, seq_wall, _, _ = _timed_run(AsyncEngineConfig.sequential())
    config = AsyncEngineConfig(
        rollout=StageConfig(workers=2),
        propose=StageConfig(workers=4),
        screen=StageConfig(workers=2),
        validate=StageConfig(workers=1),
        staleness_policy="full",
    )
    result, wall, adapter, _ = _timed_run(config)
    stats = result.metadata["async"]

    assert stats["max_busy_stages"] >= 2
    assert stats["stage_overlap_fraction"] > 0.2
    assert stats["mean_tasks_in_flight"] > 1.0
    assert adapter.max_active >= 2, "evaluator calls from different stages should overlap"
    assert wall < 0.8 * seq_wall, f"decoupled run ({wall:.2f}s) should beat one-at-a-time ({seq_wall:.2f}s)"
    assert result.val_aggregate_scores[result.best_idx] >= seq_result.val_aggregate_scores[0]
    for stage in ("rollout", "propose", "screen", "validate"):
        assert stats["stages"][stage]["completed_tasks"] > 0


# ----------------------------------------------------------------------
# staleness policies
# ----------------------------------------------------------------------


def _wide(policy: str, max_staleness: int) -> AsyncEngineConfig:
    return AsyncEngineConfig(
        rollout=StageConfig(workers=4),
        propose=StageConfig(workers=8),
        screen=StageConfig(workers=4),
        validate=StageConfig(workers=2),
        staleness_policy=policy,  # type: ignore[arg-type]
        max_staleness=max_staleness,
    )


def test_full_policy_never_discards_and_records_gaps():
    result, _, _, _ = _timed_run(_wide("full", 0))
    stats = result.metadata["async"]
    assert stats["outcomes"].get("discarded_stale", 0) == 0
    assert any(int(g) > 0 for g in stats["staleness_at_screen"]), "a wide pipeline should produce stale items"


def test_guarded_policy_discards_items_beyond_the_bound():
    result, _, _, lm = _timed_run(_wide("guarded", 0))
    stats = result.metadata["async"]
    assert stats["outcomes"].get("discarded_stale", 0) > 0
    # The pool may move while a child is screened, but nothing stale is admitted to validation,
    # and guarded never calls the patch prompt.
    assert set(stats["staleness_at_validate"]) == {"0"}
    assert lm.patch_calls == 0
    assert stats["patched"] == 0


def test_reflective_policy_patches_stale_proposals_onto_the_current_best():
    result, _, _, lm = _timed_run(_wide("reflective", 0), budget=400)
    stats = result.metadata["async"]
    assert lm.patch_calls > 0
    assert stats["patched"] + stats["patch_discards"] > 0
    # A patched child descends from the candidate it was rewritten on top of, and keeps that parent's tokens.
    for idx, parents in enumerate(result.parents):
        for p in parents:
            if p is not None:
                assert _tokens(result.candidates[p]["instructions"]) <= _tokens(result.candidates[idx]["instructions"])


def test_reflective_policy_falls_back_without_a_reflection_lm():
    result = optimize(
        seed_candidate=dict(SEED),
        trainset=TRAIN,
        valset=VAL,
        adapter=TokenAdapter(),
        custom_candidate_proposer=_token_proposer,
        max_metric_calls=200,
        logger=_silent_logger(),
        async_config=_wide("reflective", 0),
    )
    assert result.metadata["async"]["patched"] == 0


def test_patch_prompt_round_trip():
    prompt = render_patch_prompt(PatchJob("instructions", "a k1", "a k1 k2", "a k1 k3", ["a k1 k3"]))
    assert "a k1 k2" in prompt and "a k1 k3" in prompt and "Accepted version 1" in prompt
    assert parse_patch_output("DISCARD") is None
    assert parse_patch_output("discard - already covered") is None
    assert parse_patch_output("```\nnew text\n```") == "new text"
    assert parse_patch_output("```text\nnew text\n```") == "new text"
    assert parse_patch_output("```\n\n```") is None


# ----------------------------------------------------------------------
# budget
# ----------------------------------------------------------------------


@pytest.mark.parametrize("budget", [90, 137, 200])
def test_reserved_budget_is_never_overshot(budget):
    result = optimize(
        seed_candidate=dict(SEED),
        trainset=TRAIN,
        valset=VAL,
        adapter=TokenAdapter(eval_sleep=0.002),
        custom_candidate_proposer=_token_proposer,
        max_metric_calls=budget,
        logger=_silent_logger(),
        async_config=_wide("full", 0),
    )
    assert result.total_metric_calls <= budget
    assert result.metadata["async"]["total_metric_calls"] == result.total_metric_calls


# ----------------------------------------------------------------------
# early-exit validation
# ----------------------------------------------------------------------


def test_prefix_validation_keeps_pool_scores_complete():
    config = _wide("full", 0)
    config.validate_prefix_fraction = 0.25
    result = optimize(
        seed_candidate=dict(SEED),
        trainset=TRAIN,
        valset=VAL,
        adapter=TokenAdapter(),
        custom_candidate_proposer=_token_proposer,
        max_metric_calls=500,
        logger=_silent_logger(),
        async_config=config,
    )
    assert len(result.candidates) > 1
    assert all(len(scores) == len(VAL) for scores in result.val_subscores)


# ----------------------------------------------------------------------
# errors
# ----------------------------------------------------------------------


def test_worker_errors_are_isolated_when_not_raising():
    result = optimize(
        seed_candidate=dict(SEED),
        trainset=TRAIN,
        valset=VAL,
        adapter=TokenAdapter(fail_on="k5"),
        custom_candidate_proposer=_token_proposer,
        max_metric_calls=300,
        raise_on_exception=False,
        logger=_silent_logger(),
        async_config=_wide("full", 0),
    )
    assert result.metadata["async"]["outcomes"].get("error", 0) > 0
    assert all("k5" not in c["instructions"] for c in result.candidates)
    assert len(result.candidates) > 1


def test_worker_errors_propagate_when_raising():
    with pytest.raises(RuntimeError, match="k5"):
        optimize(
            seed_candidate=dict(SEED),
            trainset=TRAIN,
            valset=VAL,
            adapter=TokenAdapter(fail_on="k5"),
            custom_candidate_proposer=_token_proposer,
            max_metric_calls=300,
            raise_on_exception=True,
            logger=_silent_logger(),
            async_config=_wide("full", 0),
        )


# ----------------------------------------------------------------------
# logging and resume
# ----------------------------------------------------------------------


def test_event_log_stats_and_resume(tmp_path):
    run_dir = str(tmp_path / "run")
    common: dict[str, Any] = {
        "seed_candidate": dict(SEED),
        "trainset": TRAIN,
        "valset": VAL,
        "custom_candidate_proposer": _token_proposer,
        "run_dir": run_dir,
        "async_config": _wide("full", 0),
    }
    first = optimize(adapter=TokenAdapter(), max_metric_calls=150, **common)
    events = [json.loads(line) for line in open(os.path.join(run_dir, "async_events.jsonl"))]
    kinds = {e["kind"] for e in events}
    assert {"run_start", "sample", "dispatch", "complete", "finish", "commit", "run_end"} <= kinds
    commits = [e for e in events if e["kind"] == "commit"]
    assert len(commits) == len(first.candidates) - 1
    assert os.path.exists(os.path.join(run_dir, "async_stats.json"))
    assert os.path.exists(os.path.join(run_dir, "gepa_state.bin"))

    second = optimize(adapter=TokenAdapter(), max_metric_calls=320, **common)
    assert second.candidates[: len(first.candidates)] == first.candidates
    assert len(second.candidates) > len(first.candidates)
    assert second.total_metric_calls <= 320


def test_callbacks_fire_per_work_order():
    seen: dict[str, int] = {}

    class Recorder:
        def __getattr__(self, name):
            if not name.startswith("on_"):
                raise AttributeError(name)

            def record(event):
                seen[name] = seen.get(name, 0) + 1

            return record

    result = optimize(
        seed_candidate=dict(SEED),
        trainset=TRAIN,
        valset=VAL,
        adapter=TokenAdapter(),
        custom_candidate_proposer=_token_proposer,
        max_metric_calls=200,
        callbacks=[Recorder()],  # type: ignore[list-item]
        logger=_silent_logger(),
        async_config=_wide("full", 0),
    )
    orders = result.metadata["async"]["orders"]
    assert seen["on_iteration_start"] == orders == seen["on_iteration_end"]
    assert seen["on_candidate_accepted"] == len(result.candidates) - 1
    assert seen["on_optimization_start"] == seen["on_optimization_end"] == 1


# ----------------------------------------------------------------------
# adaptive worker control
# ----------------------------------------------------------------------


def test_worker_controller_moves_caps_toward_the_bottleneck():
    import queue

    completions: queue.Queue = queue.Queue()
    names = ("rollout", "propose", "screen", "validate")
    pools = {
        n: StagePool(n, "r", StageConfig(workers=2, min_workers=1, max_workers=4, max_queue=2), completions)
        for n in names
    }
    controller = WorkerController(pools, interval=0.0)
    time.sleep(0.01)
    # propose is slow with work waiting and all workers busy; rollout is fast and its downstream buffer is full.
    pools["rollout"].window_completed_items = 100
    pools["propose"].window_completed_items = 5
    pools["screen"].window_completed_items = 20
    pools["validate"].window_completed_items = 20
    pools["propose"].buffer.extend([object(), object()])  # type: ignore[list-item]
    pools["propose"].inflight_tasks = 2
    changes = {s: (old, new) for s, old, new in controller.maybe_adjust()}
    assert changes["propose"] == (2, 3)
    assert changes["rollout"] == (2, 1)
    assert "screen" not in changes and "validate" not in changes
    for p in pools.values():
        p.shutdown()


def test_adaptive_run_completes_and_reports_final_workers():
    config = AsyncEngineConfig(
        rollout=StageConfig(workers=1, max_workers=3),
        propose=StageConfig(workers=1, max_workers=6),
        screen=StageConfig(workers=1, max_workers=3),
        validate=StageConfig(workers=1, max_workers=2),
        adaptive_workers=True,
        adjust_interval_seconds=0.2,
        staleness_policy="full",
    )
    result, _, _, _ = _timed_run(config, budget=400)
    stats = result.metadata["async"]
    assert set(stats["final_workers"]) >= {"rollout", "propose", "screen", "validate"}
    assert stats["final_workers"]["propose"] >= 1
    assert len(result.candidates) > 1


def test_config_validation():
    with pytest.raises(ValueError):
        StageConfig(workers=0)
    with pytest.raises(ValueError):
        StageConfig(workers=3, max_workers=2)
    with pytest.raises(ValueError):
        AsyncEngineConfig(staleness_policy="sometimes")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        AsyncEngineConfig(validate_prefix_fraction=1.5)
    assert AsyncEngineConfig.sequential().max_pipeline_items == 1


def test_budget_tail_does_not_strand_accepted_children(tmp_path):
    """Near the end of the budget the engine must stop sampling orders it cannot validate."""
    run_dir = str(tmp_path / "run")
    result = optimize(
        seed_candidate=dict(SEED),
        trainset=TRAIN,
        valset=VAL,
        adapter=TokenAdapter(eval_sleep=0.002),
        custom_candidate_proposer=_token_proposer,
        max_metric_calls=400,
        run_dir=run_dir,
        logger=_silent_logger(),
        async_config=_wide("full", 0),
    )
    events = [json.loads(line) for line in open(os.path.join(run_dir, "async_events.jsonl"))]
    finishes = [e for e in events if e["kind"] == "finish"]
    stranded = [e for e in finishes if e["outcome"] == "budget" and e.get("at") == "validate"]
    budget_drops = [e for e in finishes if e["outcome"] == "budget"]
    assert result.total_metric_calls <= 400
    # The admission rule reserves at the gate and estimates in-flight claims, so a couple of children can
    # still strand at the very end; the unfixed engine stranded five here and twelve on a real task.
    assert len(stranded) <= 2, f"{len(stranded)} accepted children were dropped for lack of validation budget"
    assert len(budget_drops) <= 0.15 * len(finishes), (
        f"{len(budget_drops)} of {len(finishes)} orders hit the budget wall"
    )
    # The budget should still be nearly used up, not abandoned early.
    assert result.total_metric_calls >= 400 - (len(VAL) + 6)


def test_orders_per_version_cap_restores_lineage_depth():
    """Without a cap a fast proposer samples most orders from the seed; the cap forces generations."""

    def depths(result):
        d = [0] * len(result.parents)
        for i, ps in enumerate(result.parents):
            ps = [p for p in ps if p is not None]
            d[i] = 0 if not ps else 1 + max(d[p] for p in ps)
        return d

    def run(cap):
        config = _wide("full", 0)
        config.max_orders_per_version = cap
        config.max_pipeline_items = 64
        # Validation is slow relative to proposing, as on a real task with a large validation set.
        adapter = TokenAdapter()
        original = adapter.evaluate

        def slow_validation(batch, candidate, capture_traces=False):
            if len(batch) > 3:
                time.sleep(0.15)
            return original(batch, candidate, capture_traces)

        adapter.evaluate = slow_validation  # type: ignore[method-assign]
        return optimize(
            seed_candidate=dict(SEED),
            trainset=TRAIN,
            valset=VAL,
            adapter=adapter,
            custom_candidate_proposer=_token_proposer,
            max_metric_calls=500,
            logger=_silent_logger(),
            async_config=config,
        )

    uncapped, capped = run(None), run(2)
    assert max(depths(capped)) > max(depths(uncapped))
    assert max(depths(capped)) >= 3


def test_validate_priority_orders_by_minibatch_score():
    """With a priority mode, a queued child with the higher minibatch score is validated first."""
    import queue

    from gepa.async_engine.items import WorkItem
    from gepa.core.adapter import EvaluationBatch

    def item(order, before, after):
        it = WorkItem(order, f"id{order}", 0, 0, {}, [], [], {}, created_at=0.0)
        it.parent_eval = EvaluationBatch(outputs=[], scores=before)
        it.child_eval = EvaluationBatch(outputs=[], scores=after)
        return it

    for mode, expected in (("fifo", [1, 2, 3]), ("score", [3, 2, 1]), ("improvement", [2, 3, 1])):
        cfg = AsyncEngineConfig(validate_priority=mode)  # type: ignore[arg-type]
        # Reuse the engine's ranking without constructing an engine.
        from gepa.async_engine.engine import AsyncStageEngine

        fn = AsyncStageEngine._validate_priority_fn(type("E", (), {"config": cfg})())
        pool = StagePool("validate", "evaluator", StageConfig(workers=1, batch_size=1), queue.Queue(), priority=fn)
        # order 1: score 1.0, gain 0.5; order 2: score 2.0, gain 2.0; order 3: score 3.0, gain 1.0
        for it in (item(1, [0.5], [1.0]), item(2, [0.0], [2.0]), item(3, [2.0], [3.0])):
            pool.put(it)
        got = [pool.take_batch()[0].order_id for _ in range(3)]
        assert got == expected, (mode, got)
        pool.shutdown()
    with pytest.raises(ValueError):
        AsyncEngineConfig(validate_priority="random")  # type: ignore[arg-type]


def test_stage_selection_layers_rank_and_cap():
    """Each stage's selector ranks its buffer; the rollout cap re-samples an over-used parent."""
    import queue
    import types

    from gepa.async_engine.engine import AsyncStageEngine
    from gepa.async_engine.items import WorkItem
    from gepa.core.adapter import EvaluationBatch

    state = types.SimpleNamespace(program_full_scores_val_set=[0.5, 0.9, 0.7])

    def item(order, parent, parent_scores):
        it = WorkItem(order, f"id{order}", 0, parent, {}, [], [], {}, created_at=0.0)
        it.parent_eval = EvaluationBatch(outputs=[], scores=parent_scores)
        return it

    items = [item(1, 0, [1.0, 1.0, 0.0]), item(2, 1, [0.0, 0.0, 0.0]), item(3, 2, [1.0, 0.0, 0.0])]

    def order_with(stage, **cfg):
        engine = types.SimpleNamespace(config=AsyncEngineConfig(**cfg))
        fn = AsyncStageEngine._stage_priority_fn(engine, stage, state)  # type: ignore[arg-type]
        pool = StagePool(stage, "r", StageConfig(workers=1, batch_size=1), queue.Queue(), priority=fn)
        for it in items:
            pool.put(it)
        got = [pool.take_batch()[0].order_id for _ in range(3)]
        pool.shutdown()
        return got

    assert order_with("propose") == [1, 2, 3]
    assert order_with("propose", propose_priority="headroom") == [2, 3, 1]
    assert order_with("propose", propose_priority="parent_score") == [2, 3, 1]
    assert order_with("screen", screen_priority="parent_score") == [2, 3, 1]
    with pytest.raises(ValueError):
        AsyncEngineConfig(propose_priority="luck")  # type: ignore[arg-type]

    # Rollout cap: the selector keeps returning parent 0; with a cap the engine re-draws until a
    # parent with room comes up, and gives up after bounded retries when none does.
    class Cycle:
        def __init__(self, seq):
            self.seq, self.i = seq, 0

        def select_candidate_idx(self, _state):
            v = self.seq[self.i % len(self.seq)]
            self.i += 1
            return v

    def pick(cap, inflight, seq):
        engine = types.SimpleNamespace(
            config=AsyncEngineConfig(max_inflight_per_parent=cap),
            candidate_selector=Cycle(seq),
            _inflight_by_parent=dict(inflight),
        )
        return AsyncStageEngine._select_parent(engine, state)  # type: ignore[arg-type]

    assert pick(None, {0: 5}, [0, 0, 1]) == 0  # no cap: first draw stands
    assert pick(2, {0: 1}, [0, 0, 1]) == 0  # parent 0 has room
    assert pick(2, {0: 2}, [0, 0, 1]) == 1  # parent 0 is full: re-drawn until parent 1
    assert pick(1, {0: 1, 1: 1}, [0, 1]) in (0, 1)  # every parent full: bounded retries, then accept
