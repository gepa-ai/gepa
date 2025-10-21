import asyncio

from ufast_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst


def test_default_adapter_runs_small_loop(tmp_path):
    dataset = [
        DefaultDataInst(input="What is 2+2?", answer="4"),
        DefaultDataInst(input="Capital of France?", answer="Paris"),
    ]
    adapter = DefaultAdapter(dataset, cache_dir=tmp_path.as_posix(), log_dir=(tmp_path / "logs").as_posix())

    async def run():
        result = await adapter.optimize_async(
            seeds=["You are a helpful assistant."],
            max_rounds=1,
            max_evaluations=10,
        )
        assert result["pareto"], "Pareto set should not be empty"

    asyncio.run(run())
