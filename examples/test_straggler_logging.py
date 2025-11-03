"""
Simple script to demonstrate straggler detection logging.
Creates artificial variance in evaluation times to trigger straggler cancellation.
"""
import asyncio
import random
from turbo_gepa.adapters.default_adapter import DefaultDataInst, DefaultAdapter
from turbo_gepa.config import Config
import gepa

def main():
    print("=" * 80)
    print("STRAGGLER DETECTION DEMO")
    print("=" * 80)

    # Small dataset
    trainset, _, _ = gepa.examples.aime.init_dataset()
    dataset = [
        DefaultDataInst(
            input=trainset[i]["input"],
            answer=trainset[i]["answer"],
            id=f"aime_{i}",
        )
        for i in range(10)  # Just 10 examples
    ]

    print(f"Dataset: {len(dataset)} examples\n")

    # Config designed to show straggler detection
    config = Config(
        eval_concurrency=10,  # Evaluate all 10 in parallel
        batch_size=10,
        shards=(1.0,),  # Single rung to simplify
        max_mutations_per_round=0,  # No mutations - just eval seed
        log_level="INFO",
    )

    print("Config:")
    print(f"  eval_concurrency: {config.eval_concurrency}")
    print(f"  batch_size: {config.batch_size}\n")

    # Use a fast model to get quick responses
    task_lm = "openrouter/openai/gpt-4o-mini"
    reflection_lm = "openrouter/x-ai/grok-4-fast"

    print(f"Models:")
    print(f"  task_lm: {task_lm}")
    print(f"  reflection_lm: {reflection_lm}\n")

    seed = "You are a math expert. Solve the problem and provide your answer in the format '### <answer>'"

    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        config=config,
        auto_config=False,
    )

    print("🚀 Starting optimization (seed evaluation only)...\n")
    print("Watch for straggler detection logs showing:")
    print("  🔍 Threshold calculations")
    print("  ✅/⚠️  Task status checks")
    print("  ⚡ Straggler cancellations (if variance is high)")
    print("  ⏱️  Batch completion metrics\n")
    print("=" * 80)
    print()

    result = adapter.optimize(
        seeds=[seed],
        max_evaluations=10,  # Just evaluate the seed
        display_progress=True,  # Enable logging
    )

    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    print(f"Evaluations: {result.get('evaluations_run', 'unknown')}")
    print(f"Best quality: {result.get('best_quality', 'unknown')}")


if __name__ == "__main__":
    main()
