"""
Training script using GEPA optimize_anything API.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Suppress verbose LiteLLM logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

import re

import litellm
from datasets import load_dataset
from gskill.cost_tracker import reset_tracker
from gskill.experiment_logger import ExperimentLogger, set_logger
from gskill.swe_fitness_fn import create_swe_fitness_fn

from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, TrackingConfig, optimize_anything
from gepa.utils.stop_condition import TimeoutStopCondition


class TeeOutput:
    """Duplicate stdout/stderr to both terminal and a log file."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_file = None
        self.original_stdout = None
        self.original_stderr = None

    def start(self):
        """Start capturing output."""
        self.log_file = open(self.log_path, "w", buffering=1)  # Line buffered
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self._TeeStream(self.original_stdout, self.log_file)
        sys.stderr = self._TeeStream(self.original_stderr, self.log_file)
        print(f"[Logging terminal output to: {self.log_path}]")

    def stop(self):
        """Stop capturing and restore original streams."""
        if self.original_stdout:
            sys.stdout = self.original_stdout
        if self.original_stderr:
            sys.stderr = self.original_stderr
        if self.log_file:
            self.log_file.close()

    class _TeeStream:
        """Stream wrapper that writes to both original stream and file."""

        def __init__(self, original, log_file):
            self.original = original
            self.log_file = log_file

        def write(self, data):
            self.original.write(data)
            self.original.flush()
            self.log_file.write(data)
            self.log_file.flush()

        def flush(self):
            self.original.flush()
            self.log_file.flush()

        def fileno(self):
            return self.original.fileno()


class LoggingProposer:
    """Base proposer class that logs all inputs/outputs to separate files.

    Can be inherited to create custom proposers with different strategies.
    """

    # Custom prompt for learning terminal and repo-specific skills
    DEFAULT_PROMPT_TEMPLATE = """You are helping an AI coding agent learn skills for fixing bugs in a software repository.

The agent operates inside a Docker container with:
- The repository cloned at /testbed
- Access to bash commands (grep, find, cat, sed, python, pytest, git, etc.)
- The ability to read, edit, and create files

## Current Skills
The agent currently has these learned skills:
```
{curr_skills}
```

## Evaluation Results
Below are results from the agent attempting to fix bugs using the current skills. Each entry shows:
- The problem statement (bug description)
- The agent's actions and reasoning
- Whether the fix succeeded or failed
- Test output and error messages

```
{evaluation_data}
```

## Your Task
Analyze the evaluation results and propose IMPROVED SKILLS that will help the agent:

1. **Terminal/Bash Skills**: Common command patterns, file navigation, searching code, running tests
2. **Repository-Specific Knowledge**: Project structure, key modules, common patterns in this codebase
3. **Debugging Strategies**: How to locate bugs, understand test failures, verify fixes
4. **Code Editing Patterns**: Safe ways to modify files, handle imports, avoid syntax errors

Focus on:
- Patterns that led to SUCCESS - reinforce these
- Patterns that led to FAILURE - what should the agent do differently?
- Missing knowledge that would have helped
- Common mistakes to avoid

Write the improved skills as clear, actionable instructions. Be specific and concrete. Remember the agent will not be evaluated on the same task, but they are expected to use the skills to fix other tasks in the SAME repository.
If any older skills are still helpful, include them as well. However, always keep the overall skills concise and comprehensive.
Adding skills blindly will not help the agent, so only add skills that are actually helpful.

Provide the new skills within a SINGLE ``` block. Only include one ``` block, if you include multiple ``` blocks, the agent will not be able to parse the response."""

    def __init__(self, run_dir: Path, reflection_lm: str):
        """Initialize the proposer.

        Args:
            run_dir: Directory to save logs
            reflection_lm: Model name for reflection LLM
        """
        self.run_dir = run_dir
        self.reflection_lm = reflection_lm
        self.proposer_dir = run_dir / "proposer_calls"
        self.proposer_dir.mkdir(parents=True, exist_ok=True)
        self.call_counter = 0
        self.prompt_template = self.DEFAULT_PROMPT_TEMPLATE

    def format_prompt(self, curr_skills: str, evaluation_data: str) -> str:
        """Format the prompt with current skills and evaluation data.

        Override this method to customize prompt formatting.
        """
        return self.prompt_template.format(
            curr_skills=curr_skills if curr_skills else "(No skills learned yet)", evaluation_data=evaluation_data
        )

    def call_llm(self, prompt: str) -> str:
        """Call the LLM and return the response text.

        Override this method to customize LLM calling.
        """
        response = litellm.completion(
            model=self.reflection_lm,
            messages=[{"role": "user", "content": prompt}],
            drop_params=True,  # Let litellm drop unsupported params like temperature for gpt-5
        )
        return response.choices[0].message.content

    def extract_skills(self, response_text: str) -> str:
        """Extract skills from LLM response.

        Override this method to customize extraction logic.
        Matches from the first ``` to the last ``` to handle nested backtick references.
        """
        # Find the first code fence (with optional language tag)
        first = re.search(r"```(?:\w*\n)?", response_text)
        if not first:
            return response_text.strip()

        # Find the last occurrence of ``` to handle nested backtick references
        last_pos = response_text.rfind("```")
        if last_pos <= first.end():
            return response_text.strip()

        return response_text[first.end() : last_pos].strip()

    def log_call(self, call_num: int, log_entry: dict) -> Path:
        """Log the call to a file and return the file path.

        Override this method to customize logging.
        """
        call_file = self.proposer_dir / f"call_{call_num:03d}.json"
        with open(call_file, "w") as f:
            json.dump(log_entry, f, indent=2, default=str)
        return call_file

    def __call__(self, candidate: dict, reflective_dataset: dict, components_to_update: list) -> dict:
        """Main proposer logic - processes all side info at once.

        Args:
            candidate: Current candidate dict with component values
            reflective_dataset: Dict mapping components to their side info lists
            components_to_update: List of component names to update

        Returns:
            Dict mapping components to their new values
        """
        self.call_counter += 1
        call_num = self.call_counter

        results = {}

        for component in components_to_update:
            curr_value = candidate.get(component, "")
            side_info = reflective_dataset.get(component, [])

            # Format side info as string
            side_info_str = json.dumps(side_info, indent=2, default=str)

            # Format the prompt
            prompt = self.format_prompt(curr_value, side_info_str)

            # Prepare log entry
            log_entry = {
                "call_num": call_num,
                "component": component,
                "current_skills": curr_value,
                "evaluation_data": side_info,
                "full_prompt": prompt,
                "type": "batch",
            }

            # Call the reflection LLM
            try:
                response_text = self.call_llm(prompt)
                new_value = self.extract_skills(response_text)

                log_entry["llm_response"] = response_text
                log_entry["new_skills"] = new_value
                log_entry["success"] = True

                results[component] = new_value

            except Exception as e:
                log_entry["error"] = str(e)
                log_entry["success"] = False
                results[component] = curr_value  # Keep current on error

            # Write to separate file for this call
            call_file = self.log_call(call_num, log_entry)

            # Also print summary
            print(f"\n[PROPOSER] Call {call_num} for '{component}':")
            print(f"  Current skills: {len(curr_value)} chars")
            print(f"  Evaluation items: {len(side_info)}")
            print(f"  New skills: {len(results.get(component, ''))} chars")
            print(f"  Saved to: {call_file}")

        return results


class LoopProposer(LoggingProposer):
    """Proposer that processes one side info item at a time, then merges all skills."""

    MERGE_TEMPLATE = """You are helping an AI coding agent consolidate learned skills for fixing bugs.

## New Skills/Insights from Recent Evaluations
{intermediate_skills}

## Your Task
Merge all the above skills into a SINGLE, coherent skill set that:
1. Eliminates redundancy
2. Resolves conflicts (keep the most reliable/general one)
3. Prioritizes (most important first)
4. Stays concise

Provide the merged skills within a SINGLE ``` block."""

    def _process_single_item(
        self, component: str, curr_skills: str, item: dict, item_idx: int, total_items: int
    ) -> str | None:
        """Process a single evaluation item and return extracted skills, or None on failure."""
        self.call_counter += 1
        call_num = self.call_counter

        item_str = json.dumps(item, indent=2, default=str)
        prompt = self.format_prompt(curr_skills, item_str)

        log_entry = {
            "call_num": call_num,
            "component": component,
            "item_index": item_idx,
            "total_items": total_items,
            "current_skills": curr_skills,
            "evaluation_item": item,
            "full_prompt": prompt,
            "type": "single_item",
        }

        try:
            response_text = self.call_llm(prompt)
            new_skills = self.extract_skills(response_text)
            log_entry.update({"llm_response": response_text, "extracted_skills": new_skills, "success": True})
            result = new_skills
        except Exception as e:
            log_entry.update({"error": str(e), "success": False})
            result = None

        self.log_call(call_num, log_entry)
        print(
            f"\n[LOOP PROPOSER] Call {call_num} - Item {item_idx + 1}/{total_items} for '{component}': {'OK' if result else 'FAIL'}"
        )
        return result

    def _merge_skills(self, component: str, curr_skills: str, intermediate_skills: list) -> str:
        """Merge intermediate skills into final skill set."""
        self.call_counter += 1
        call_num = self.call_counter

        formatted = "\n\n".join(f"### Evaluation {i}\n```\n{s}\n```" for i, s in enumerate(intermediate_skills, 1))
        merge_prompt = self.MERGE_TEMPLATE.format(intermediate_skills=formatted)

        log_entry = {
            "call_num": call_num,
            "component": component,
            "type": "merge",
            "num_intermediate_skills": len(intermediate_skills),
            "intermediate_skills": intermediate_skills,
            "full_prompt": merge_prompt,
        }

        try:
            response_text = self.call_llm(merge_prompt)
            final_skills = self.extract_skills(response_text)
            log_entry.update({"llm_response": response_text, "final_skills": final_skills, "success": True})
        except Exception as e:
            log_entry.update({"error": str(e), "success": False})
            final_skills = "\n\n---\n\n".join(intermediate_skills)  # Fallback

        self.log_call(call_num, log_entry)
        print(
            f"\n[LOOP PROPOSER] Merge call {call_num} for '{component}': {len(intermediate_skills)} items -> {len(final_skills)} chars"
        )
        return final_skills

    def __call__(self, candidate: dict, reflective_dataset: dict, components_to_update: list) -> dict:
        """Process each side info item one at a time, then merge all skills."""
        results = {}

        for component in components_to_update:
            curr_skills = candidate.get(component, "")
            side_info_items = reflective_dataset.get(component, [])

            if not side_info_items:
                results[component] = curr_skills
                continue

            # Process each item
            intermediate_skills = [
                skills
                for i, item in enumerate(side_info_items)
                if (skills := self._process_single_item(component, curr_skills, item, i, len(side_info_items)))
            ]

            # Merge results
            results[component] = (
                self._merge_skills(component, curr_skills, intermediate_skills) if intermediate_skills else curr_skills
            )

        return results


def create_logging_proposer(run_dir: Path, reflection_lm: str):
    """Create a LoggingProposer instance (factory function for backward compatibility)."""
    return LoggingProposer(run_dir, reflection_lm)


def create_loop_proposer(run_dir: Path, reflection_lm: str):
    """Create a LoopProposer instance that processes items one at a time then merges."""
    return LoopProposer(run_dir, reflection_lm)


import random


def load_and_split_data(repo: str, train_size: int, val_size: int, test_size: int, seed: int = 42):
    """
    Load data directly from HuggingFace and split into train/val/test.

    Args:
        repo: Repository name to filter (e.g., "pygments__pygments" or "swesmith/pygments__pygments")
        train_size: Number of training examples
        val_size: Number of validation examples
        test_size: Number of test examples
        seed: Random seed for shuffling

    Returns:
        (train_data, val_data, test_data) tuple
    """
    total_needed = train_size + val_size + test_size
    print(f"Loading {repo} data from HuggingFace SWE-smith (need {total_needed})...")
    ds = load_dataset("SWE-bench/SWE-smith", split="train")

    # Normalize repo name for matching
    # Handle both "pygments__pygments" and "swesmith/pygments__pygments"
    repo_pattern = repo if "/" in repo else f"swesmith/{repo}"

    all_data = []
    for item in ds:
        repo_name = item.get("repo", "")
        if repo_pattern in repo_name:
            all_data.append(dict(item))
            if len(all_data) >= total_needed:
                break

    print(f"Loaded {len(all_data)} {repo} examples")

    # Shuffle with seed
    random.seed(seed)
    random.shuffle(all_data)

    # Calculate split sizes
    total_needed = train_size + val_size + test_size
    if len(all_data) < total_needed:
        print(f"WARNING: Only {len(all_data)} examples available, need {total_needed}")
        # Scale down proportionally
        ratio = len(all_data) / total_needed
        train_size = int(train_size * ratio)
        val_size = int(val_size * ratio)
        test_size = len(all_data) - train_size - val_size

    # Split
    train_data = all_data[:train_size]
    val_data = all_data[train_size : train_size + val_size]
    test_data = all_data[train_size + val_size : train_size + val_size + test_size]

    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    return train_data, val_data, test_data


def evaluate_on_test(fitness_fn, candidate: dict, test_data: list, name: str = "Test"):
    """Evaluate a candidate on test data and return results."""
    print(f"\n{'=' * 70}")
    print(f"Evaluating {name}...")
    print(f"{'=' * 70}")

    results = [fitness_fn(candidate, task) for task in test_data]

    scores = [r[0] for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    pass_count = sum(1 for s in scores if s == 1.0)

    print(f"\n{name} Results:")
    print(f"  Pass rate: {pass_count}/{len(test_data)} ({avg_score:.1%})")

    return {"avg_score": avg_score, "pass_count": pass_count, "total": len(test_data), "scores": scores}


def main():
    parser = argparse.ArgumentParser(description="GEPA optimization using optimize_anything API (recommended)")
    parser.add_argument(
        "--repo",
        type=str,
        default="pygments__pygments",
        help="Repository to train on (e.g., 'pygments__pygments', 'django__django')",
    )
    parser.add_argument("--train-size", type=int, default=200, help="Number of training examples")
    parser.add_argument("--val-size", type=int, default=50, help="Number of validation examples for Pareto selection")
    parser.add_argument("--test-size", type=int, default=100, help="Number of test examples for final evaluation")
    parser.add_argument(
        "--model", type=str, default="gpt-5-mini", help="Agent model for running tasks (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--reflection-model",
        type=str,
        default="gpt-5.2-pro",
        help="Model for GEPA reflection/prompt optimization (default: gpt-5.2-pro)",
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a quick smoke test")
    parser.add_argument(
        "--run-post-optimization-testset", action="store_true", help="Evaluate on test set after optimization"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a previous run directory (e.g., gepa_results/logs/run_XXXXXXXX)",
    )
    parser.add_argument(
        "--run-pre-optimization-testset", action="store_true", help="run the agent on the test set before optimization"
    )
    parser.add_argument(
        "--run-testset", action="store_true", help="run the agent on the test set before and after optimization"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--timeout", type=int, default=43200, help="Max seconds to run (default: 12 hours)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=6, help="Number of parallel workers (Docker containers)")
    parser.add_argument("--max-metric-calls", type=int, default=600, help="Max rollouts for GEPA (controls budget)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb tracking")
    parser.add_argument("--wandb-project", type=str, default="gepa-swesmith", help="Wandb project name")
    parser.add_argument(
        "--proposer",
        type=str,
        default="batch",
        choices=["batch", "loop"],
        help="Proposer type: 'batch' (all at once) or 'loop' (one at a time, then merge)",
    )
    args = parser.parse_args()

    if args.smoke_test:
        args.train_size = 3
        args.val_size = 2
        args.test_size = 2
        args.max_metric_calls = 20
        print("SMOKE TEST MODE")

    # Handle resume from previous run
    resumed_from = None
    if args.resume:
        resume_dir = Path(args.resume)
        if not resume_dir.exists():
            print(f"ERROR: Resume directory not found: {resume_dir}")
            sys.exit(1)

        state_file = resume_dir / "gepa_state.bin"
        if not state_file.exists():
            print(f"ERROR: No gepa_state.bin found in {resume_dir}")
            sys.exit(1)

        # Load previous config to get consistent parameters
        prev_config_file = resume_dir / "config.json"
        if prev_config_file.exists():
            with open(prev_config_file) as f:
                prev_config = json.load(f)
            # Use same repo and seed for data consistency
            args.repo = prev_config.get("repo", args.repo)
            args.seed = prev_config.get("seed", args.seed)
            print(f"RESUME MODE: Loading state from {resume_dir}")
            print(f"  Using repo={args.repo}, seed={args.seed} from previous run")

        resumed_from = str(resume_dir)

    # Initialize experiment logger FIRST to get run_id
    exp_logger = ExperimentLogger(log_dir="gepa_results/logs", repo=args.repo)
    set_logger(exp_logger)
    run_dir = exp_logger.log_dir

    # If resuming, copy the state file to the new run directory
    if resumed_from:
        import shutil

        src_state = Path(resumed_from) / "gepa_state.bin"
        dst_state = run_dir / "gepa_state.bin"
        shutil.copy2(src_state, dst_state)
        print(f"Copied state file from {src_state} to {dst_state}")

    # Setup output tee - capture all stdout/stderr to terminal.log
    terminal_log = run_dir / "terminal.log"
    tee = TeeOutput(terminal_log)
    tee.start()

    # Setup Python logging
    log_file = run_dir / "training.log"
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)

    # Reset cost tracking
    reset_tracker()

    print("\n" + "=" * 70)
    print("GEPA + SWE-smith Training (optimize_anything API)")
    print("=" * 70)
    print(f"Repository: {args.repo}")
    print(f"Model: {args.model}")
    print(f"Reflection Model: {args.reflection_model}")
    print(f"Proposer: {args.proposer}")
    print(f"Workers: {args.workers} (Docker containers)")
    print(f"Run directory: {run_dir}")
    if resumed_from:
        print(f"Resumed from: {resumed_from}")
    print("=" * 70 + "\n")

    # 1. Load Data - directly from HuggingFace, split with shuffle
    train_data, val_data, test_data = load_and_split_data(
        repo=args.repo, train_size=args.train_size, val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )

    print(f"Data splits: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    logger.info(f"Data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Determine reflection model
    reflection_model = args.reflection_model or args.model

    # Save experiment config
    exp_logger.save_config(
        {
            "api": "optimize_anything",
            "repo": args.repo,
            "model": args.model,
            "reflection_model": reflection_model,
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "max_metric_calls": args.max_metric_calls,
            "workers": args.workers,
            "seed": args.seed,
            "timeout": args.timeout,
            "resumed_from": resumed_from,
            "run_pre_optimization_testset": args.run_pre_optimization_testset,
            "run_post_optimization_testset": args.run_post_optimization_testset,
            "execution_mode": "docker",
            "wandb": args.wandb,
            "wandb_project": args.wandb_project if args.wandb else None,
            "proposer": args.proposer,
        }
    )

    # 2. Create Fitness Function
    print(f"\nCreating fitness function with {args.workers} workers...")
    fitness_fn = create_swe_fitness_fn(model_name=args.model, n_workers=args.workers)
    logger.info(f"Model: {args.model}")
    logger.info("Execution: Docker containers via SWE-smith")

    # 3. Initial Candidate (Baseline Prompt)
    # Start with empty skills - GEPA will learn and evolve them
    # The system_template in mini.yaml already has base instructions,
    # this just populates the {{ skills }} placeholder
    initial_skills = ""

    seed_candidate = {"skills": initial_skills}

    # 4. Setup stop conditions
    stop_callbacks = [
        TimeoutStopCondition(timeout_seconds=args.timeout),
    ]

    # 5. Configure GEPA with structured config
    wandb_kwargs = (
        {
            "project": args.wandb_project,
            "name": run_dir.name,
            "config": {
                "repo": args.repo,
                "model": args.model,
                "reflection_model": reflection_model,
                "train_size": len(train_data),
                "val_size": len(val_data),
            },
        }
        if args.wandb
        else None
    )

    # Configure LiteLLM for reflection model (OpenAI regional endpoint)
    if "openai" in reflection_model.lower() or reflection_model.startswith("gpt-"):
        os.environ["OPENAI_API_BASE"] = "https://us.api.openai.com/v1"

    # Create proposer based on command line choice
    if args.proposer == "loop":
        proposer = LoopProposer(run_dir, reflection_model)
    elif args.proposer == "batch":
        proposer = LoggingProposer(run_dir, reflection_model)

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=str(run_dir),
            seed=args.seed,
            display_progress_bar=True,
            max_metric_calls=args.max_metric_calls,
            candidate_selection_strategy="pareto",
            parallel=True,
            max_workers=args.workers,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_model,
            reflection_minibatch_size=3,
            skip_perfect_score=True,
            perfect_score=1.0,
            custom_candidate_proposer=proposer,
        ),
        tracking=TrackingConfig(
            use_wandb=args.wandb,
            wandb_init_kwargs=wandb_kwargs,
        ),
        stop_callbacks=stop_callbacks,
    )

    # 6. Baseline evaluations (before optimization)
    original_config_results = None
    baseline_test_results = None
    if args.run_pre_optimization_testset or args.run_testset:
        # 6a. Evaluate with original mini-swe-agent config (no skills placeholder)
        original_config_path = Path(__file__).parent / "mini_swe_agent_config" / "original_mini.yaml"
        print(f"\nEvaluating original mini-swe-agent config: {original_config_path}")
        original_fitness_fn = create_swe_fitness_fn(
            model_name=args.model, n_workers=args.workers, config_path=str(original_config_path)
        )
        original_config_results = evaluate_on_test(
            original_fitness_fn, {"skills": ""}, test_data, name="Original mini-swe-agent"
        )
        with open(run_dir / "original_config_test_results.json", "w") as f:
            json.dump(original_config_results, f, indent=2)

        # 6b. Evaluate with GEPA template + empty skills
        print("\nEvaluating GEPA template with empty skills")
        baseline_test_results = evaluate_on_test(
            fitness_fn, seed_candidate, test_data, name="GEPA template (empty skills)"
        )
        with open(run_dir / "baseline_test_results.json", "w") as f:
            json.dump(baseline_test_results, f, indent=2)

    # 7. Run GEPA
    print("\n" + "=" * 70)
    print("Starting GEPA Optimization (optimize_anything API)...")
    print("=" * 70 + "\n")

    result = optimize_anything(
        seed_candidate=seed_candidate,
        evaluator=fitness_fn,
        dataset=train_data,
        valset=val_data,
        config=config,
    )

    # 7. Results
    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)

    # best_candidate is a dict: {"skills": "..."}
    best_candidate_dict = result.best_candidate
    if isinstance(best_candidate_dict, dict):
        # It's already a dict with the prompt
        best_skills = best_candidate_dict.get("skills", str(best_candidate_dict))
    else:
        # It's an object with .candidate attribute
        best_skills = best_candidate_dict.candidate["skills"]

    # Get best score from result
    best_idx = result.best_idx
    best_score = result.val_aggregate_scores[best_idx] if result.val_aggregate_scores else 0.0
    num_candidates = result.num_candidates

    print(f"\nBest Prompt (Score: {best_score:.2%}):")
    print("-" * 70)
    print(best_skills)
    print("-" * 70)

    # Save best skills
    skills_file = run_dir / "prompts" / "best_skills.txt"
    skills_file.parent.mkdir(exist_ok=True)
    with open(skills_file, "w") as f:
        f.write(best_skills)

    print(f"\nBest skills saved to: {skills_file}")
    print(f"Full results in: {run_dir}")

    # 9. Post-optimization test evaluation
    optimized_test_results = None
    if args.run_post_optimization_testset or args.run_testset:
        optimized_candidate = {"skills": best_skills}
        optimized_test_results = evaluate_on_test(
            fitness_fn, optimized_candidate, test_data, name="Optimized (after optimization)"
        )
        # Save optimized results
        with open(run_dir / "optimized_test_results.json", "w") as f:
            json.dump(optimized_test_results, f, indent=2)

        # Print comparison
        print("\n" + "=" * 70)
        print("TEST SET COMPARISON")
        print("=" * 70)
        if original_config_results:
            print(
                f"  Original mini-swe-agent: {original_config_results['pass_count']}/{original_config_results['total']} ({original_config_results['avg_score']:.1%})"
            )
        if baseline_test_results:
            print(
                f"  GEPA template (empty):   {baseline_test_results['pass_count']}/{baseline_test_results['total']} ({baseline_test_results['avg_score']:.1%})"
            )
        print(
            f"  GEPA optimized:          {optimized_test_results['pass_count']}/{optimized_test_results['total']} ({optimized_test_results['avg_score']:.1%})"
        )
        if original_config_results:
            improvement_from_original = optimized_test_results["avg_score"] - original_config_results["avg_score"]
            print(f"  Improvement over original: {improvement_from_original:+.1%}")
        if baseline_test_results:
            improvement_from_baseline = optimized_test_results["avg_score"] - baseline_test_results["avg_score"]
            print(f"  Improvement over baseline: {improvement_from_baseline:+.1%}")
        print("=" * 70)

    # Summary
    summary_data = {
        "best_score": float(best_score),
        "num_candidates": num_candidates,
        "total_metric_calls": result.total_metric_calls,
        "final_skills_length": len(best_skills),
    }
    if original_config_results:
        summary_data["original_config_test_score"] = original_config_results["avg_score"]
    if baseline_test_results:
        summary_data["baseline_test_score"] = baseline_test_results["avg_score"]
    if optimized_test_results:
        summary_data["optimized_test_score"] = optimized_test_results["avg_score"]

    exp_logger.save_summary(best_prompt=best_skills, extra_info=summary_data)

    print("\nSummary:")
    print(f"  Best Val Score: {best_score:.2%}")
    print(f"  Candidates Generated: {num_candidates}")
    print(f"  Total Metric Calls: {result.total_metric_calls}")
    print(f"  Final Skills Length: {len(best_skills)} chars")
    if original_config_results:
        print(f"  Original Config Test Score: {original_config_results['avg_score']:.1%}")
    if baseline_test_results:
        print(f"  Baseline Test Score: {baseline_test_results['avg_score']:.1%}")
    if optimized_test_results:
        print(f"  Optimized Test Score: {optimized_test_results['avg_score']:.1%}")

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70 + "\n")

    # Stop tee logging
    tee.stop()


if __name__ == "__main__":
    main()
