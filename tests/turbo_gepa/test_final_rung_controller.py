import time

from turbo_gepa.config import Config
from turbo_gepa.orchestrator import FinalRungCapController


def _controller(
    eval_concurrency: int = 12,
    *,
    min_cap: int = 2,
    initial_cap: int | None = None,
) -> tuple[Config, FinalRungCapController]:
    cfg = Config(
        eval_concurrency=eval_concurrency,
        final_rung_min_inflight=min_cap,
        max_final_shard_inflight=initial_cap,
    )
    controller = FinalRungCapController.from_config(cfg)
    return cfg, controller


def test_final_rung_cap_increases_with_backlog():
    cfg, controller = _controller(initial_cap=2)
    now = time.time()
    current_cap = cfg.max_final_shard_inflight

    new_cap = controller.update(
        current_cap=current_cap,
        effective_concurrency=cfg.eval_concurrency,
        waiting=4,
        inflight=2,
        lat_mean=2.0,
        lat_p50=1.8,
        lat_p95=2.0,
        timeout_ratio=0.01,
        blocked_since=now - 5.0,
        straggler_events=[],
        now=now,
    )

    assert new_cap is not None and new_cap > current_cap


def test_final_rung_cap_shrinks_with_latency():
    cfg, controller = _controller(initial_cap=6)
    current_cap = cfg.max_final_shard_inflight
    now = time.time()

    new_cap = controller.update(
        current_cap=current_cap,
        effective_concurrency=cfg.eval_concurrency,
        waiting=0,
        inflight=1,
        lat_mean=20.0,
        lat_p50=18.0,
        lat_p95=45.0,
        timeout_ratio=0.3,
        blocked_since=None,
        straggler_events=[(now, 5)],
        now=now,
    )

    assert new_cap is not None and new_cap < current_cap
