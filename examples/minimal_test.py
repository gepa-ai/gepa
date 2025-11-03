"""
Minimal test to debug hanging issue with rung-aware scheduler.
"""
import asyncio
import os

os.environ["LITELLM_LOG"] = "DEBUG"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultDataInst
from turbo_gepa.config import Config


async def minimal_test():
    """Test with minimal dataset to isolate the issue."""
    print("=" * 80)
    print("MINIMAL TEST - Debugging Hang")
    print("=" * 80)

    # Use only 3 examples
    trainset, _, _ = gepa.examples.aime.init_dataset()
    dataset = [
        DefaultDataInst(
            input=trainset[i]["input"],
            answer=trainset[i]["answer"],
            id=f"aime_{i}",
            additional_context=trainset[i].get("additional_context"),
        )
        for i in range(3)
    ]

    print(f"\n✅ Loaded {len(dataset)} examples")

    # Minimal config
    config = Config(
        eval_concurrency=4,  # Low concurrency
        n_islands=1,
        shards=(1.0,),  # Single rung - no ASHA complexity
        batch_size=2,
        max_mutations_per_round=2,
        mutation_buffer_min=2,
        queue_limit=10,
        log_level="DEBUG",
        adaptive_shards_enabled=False,
        max_optimization_time_seconds=30,
    )

    print("\n📋 Config:")
    print(f"  eval_concurrency: {config.eval_concurrency}")
    print(f"  shards: {config.shards}")
    print(f"  max_mutations_per_round: {config.max_mutations_per_round}")

    # Simple seed
    seed = "You are a helpful assistant. Answer concisely."

    task_lm = "openrouter/openai/gpt-oss-20b:nitro"
    reflection_lm = "openrouter/x-ai/grok-4-fast"

    print(f"\n🤖 Models:")
    print(f"  Task: {task_lm}")
    print(f"  Reflection: {reflection_lm}")

    print("\n🚀 Starting optimization...\n")

    from turbo_gepa.adapters.default_adapter import DefaultAdapter

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        config=config,
        auto_config=False,
    )

    try:
        result = await adapter.optimize(
            seed_text=seed,
            max_evaluations=10,  # Stop after 10 evals
        )

        print("\n" + "=" * 80)
        print("✅ COMPLETED!")
        print("=" * 80)
        print(f"Total evaluations: {result.get('evaluations_run', 'unknown')}")
        print(f"Best quality: {result.get('best_quality', 'unknown')}")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(minimal_test())
