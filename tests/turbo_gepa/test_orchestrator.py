import asyncio

from turbo_gepa.archive import Archive
from turbo_gepa.cache import DiskCache
from turbo_gepa.config import Config
from turbo_gepa.evaluator import AsyncEvaluator
from turbo_gepa.interfaces import Candidate
from turbo_gepa.mutator import MutationConfig, Mutator
from turbo_gepa.orchestrator import Orchestrator
from turbo_gepa.sampler import InstanceSampler


async def fake_task_runner(candidate: Candidate, example_id: str):
    quality = float(len(candidate.text)) / (len(example_id) + 4)
    tokens = len(candidate.text.split())
    return {"quality": quality, "neg_cost": -float(tokens), "tokens": float(tokens)}


def test_orchestrator_generates_archive(tmp_path):
    examples = ["ex-1", "ex-2", "ex-3", "ex-4", "ex-5"]
    sampler = InstanceSampler(examples, seed=123)
    cache = DiskCache(tmp_path.as_posix())
    config = Config(
        eval_concurrency=4,
        n_islands=1,
        shards=(0.2, 1.0),
        cache_path=tmp_path.as_posix(),
        log_path=(tmp_path / "logs").as_posix(),
        max_mutations_per_round=4,
        queue_limit=32,
    )
    evaluator = AsyncEvaluator(cache=cache, task_runner=fake_task_runner)
    archive = Archive(
        bins_length=config.qd_bins_length,
        bins_bullets=config.qd_bins_bullets,
        flags=config.qd_flags,
    )
    mutator = Mutator(
        MutationConfig(
            reflection_batch_size=config.reflection_batch_size,
            max_mutations=config.max_mutations_per_round,
            max_tokens=config.max_tokens,
        ),
        seed=1,
    )
    orchestrator = Orchestrator(
        config=config,
        evaluator=evaluator,
        archive=archive,
        sampler=sampler,
        mutator=mutator,
        cache=cache,
    )
    seeds = [Candidate(text="Seed candidate with guidance.")]

    async def run_loop() -> None:
        await orchestrator.run(seeds, max_rounds=1)

    asyncio.run(run_loop())
    assert archive.pareto_candidates()


def test_orchestrator_mutation_budget(tmp_path):
    examples = [f"ex-{i}" for i in range(12)]
    sampler = InstanceSampler(examples, seed=0)
    cache = DiskCache(tmp_path.as_posix())
    config = Config(
        eval_concurrency=4,
        max_total_inflight=8,
        batch_size=4,
        n_islands=1,
        shards=(0.2, 1.0),
        cache_path=tmp_path.as_posix(),
        log_path=(tmp_path / "logs").as_posix(),
        max_mutations_per_round=6,
        queue_limit=32,
    )
    evaluator = AsyncEvaluator(cache=cache, task_runner=fake_task_runner)
    archive = Archive(
        bins_length=config.qd_bins_length,
        bins_bullets=config.qd_bins_bullets,
        flags=config.qd_flags,
    )
    mutator = Mutator(
        MutationConfig(
            reflection_batch_size=config.reflection_batch_size,
            max_mutations=config.max_mutations_per_round,
            max_tokens=config.max_tokens,
        ),
        seed=1,
    )
    orchestrator = Orchestrator(
        config=config,
        evaluator=evaluator,
        archive=archive,
        sampler=sampler,
        mutator=mutator,
        cache=cache,
    )

    # Empty queue → request capped by max_mutations_per_round
    assert orchestrator._mutation_budget() == 6

    # Populate queue to reduce deficit
    candidates = [Candidate(text=f"seed-{i}") for i in range(5)]
    orchestrator.enqueue(candidates)
    assert orchestrator._mutation_budget() == 3  # target_ready=8, ready=5 → deficit 3

    # Fill queue and buffer above target → no new mutations
    more_candidates = [Candidate(text=f"extra-{i}") for i in range(5)]
    orchestrator.enqueue(more_candidates)
    assert len(orchestrator.queue) >= 8
    assert orchestrator._mutation_budget() == 0
