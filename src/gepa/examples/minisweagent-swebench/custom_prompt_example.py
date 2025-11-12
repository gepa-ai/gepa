#!/usr/bin/env python3
"""
Example showing how to start from a custom prompt instead of the default config.

This is useful if you want to:
- Start from a completely different prompt
- Test a specific prompt variation
- Compare different starting points
"""

import argparse
from pathlib import Path

import litellm

from gepa import optimize
from gepa.adapters.minisweagent_adapter import MiniSWEAgentAdapter, load_swebench_instances

# Custom starting prompts
CUSTOM_SYSTEM_TEMPLATE = """You are an expert software engineer helping to fix bugs in codebases.

Your approach:
1. Carefully read the problem description
2. Explore the codebase to understand the structure
3. Identify the root cause of the issue
4. Implement a minimal, targeted fix
5. Test your fix

Provide exactly ONE bash command per response in a code block.
"""

CUSTOM_INSTANCE_TEMPLATE = """# Bug Report

{{task}}

# Your Task

Fix the bug described above. Work step-by-step:
1. Explore and understand the code
2. Reproduce the issue if needed
3. Implement a fix
4. Verify the fix works

When done, submit with:
```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached
```
"""


def main():
    parser = argparse.ArgumentParser(
        description="Train mini-swe-agent with custom starting prompts"
    )
    parser.add_argument("--model", type=str, default="anthropic/claude-sonnet-4")
    parser.add_argument("--n-instances", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("gepa_custom_prompt"))
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("GEPA Mini-SWE-Agent with Custom Prompts")
    print("=" * 80)
    print(f"Starting from custom prompts (not default config)")
    print(f"Model: {args.model}")
    print(f"Instances: {args.n_instances}")
    print("=" * 80 + "\n")
    
    # Load data
    print("Loading SWE-bench instances...")
    all_instances = load_swebench_instances(
        dataset="princeton-nlp/SWE-bench_Verified",
        split="test",
        slice_spec=f"0:{args.n_instances}",
    )
    
    n_train = int(args.n_instances * 0.6)
    trainset = all_instances[:n_train]
    valset = all_instances[n_train:]
    
    print(f"Train: {len(trainset)}, Val: {len(valset)}")
    
    # Define custom seed candidate
    seed_candidate = {
        "system_template": CUSTOM_SYSTEM_TEMPLATE,
        "instance_template": CUSTOM_INSTANCE_TEMPLATE,
    }
    
    print("\nCustom Seed Candidate:")
    print("-" * 80)
    print("System Template:", seed_candidate["system_template"][:200] + "...")
    print("\nInstance Template:", seed_candidate["instance_template"][:200] + "...")
    print("-" * 80 + "\n")
    
    # Create adapter with minimal config
    # We don't load from swebench.yaml since we're using fully custom prompts
    adapter = MiniSWEAgentAdapter(
        model_name=args.model,
        environment_class="docker",
        run_validation=False,
        timeout=300,
        temp_dir=args.output_dir / "temp",
    )
    
    # Setup reflection LM
    reflection_lm = (
        lambda prompt: litellm.completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        .choices[0]
        .message.content
    )
    
    # Run optimization
    print("Starting optimization...")
    optimized_results = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        use_wandb=False,
        max_metric_calls=30,
        reflection_minibatch_size=2,
        perfect_score=1.0,
        run_dir=str(args.output_dir),
    )
    
    # Save results
    best_candidate = optimized_results.best_candidate
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Best validation score: {optimized_results.best_score:.3f}")
    print(f"\nOptimized prompts saved to: {args.output_dir}")
    print("=" * 80)
    
    # Save optimized prompts
    with open(args.output_dir / "best_candidate.txt", "w") as f:
        f.write("OPTIMIZED SYSTEM TEMPLATE\n")
        f.write("=" * 80 + "\n")
        f.write(best_candidate["system_template"])
        f.write("\n\n" + "=" * 80 + "\n\n")
        f.write("OPTIMIZED INSTANCE TEMPLATE\n")
        f.write("=" * 80 + "\n")
        f.write(best_candidate["instance_template"])


if __name__ == "__main__":
    main()


