"""
Structured experiment logging for GEPA + SWE-smith experiments.

Logs everything needed for analysis:
- Per-iteration metrics (tokens, steps, scores)
- Proposer inputs (prompts + traces sent to reflection LLM)
- Generated prompts at each iteration
- Aggregated statistics
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""

    instance_id: str
    success: bool
    score: float
    steps: int
    estimated_tokens: int
    has_patch: bool


@dataclass
class EvalBatchMetrics:
    """
    Metrics for one adapter.evaluate() call.

    Note: GEPA calls evaluate() multiple times:
    - Different candidates (prompts) on same tasks
    - Train set sampling for reflection
    - Val set for Pareto selection

    Use prompt_hash to identify which candidate/prompt was used.
    Same prompt_hash = same prompt being evaluated.
    """

    eval_id: int  # Sequential counter of evaluate() calls
    timestamp: str
    prompt_hash: str  # Unique identifier for this prompt/candidate
    prompt_preview: str  # First 100 chars for quick reference

    # Task-level data
    num_tasks: int
    num_passed: int
    pass_rate: float

    # Aggregated metrics
    avg_steps: float  # Avg agent steps per task
    avg_tokens: float  # Avg agent tokens per task (estimated)
    total_steps: int
    total_tokens: int

    # Per-task breakdown (instance_id tells you which task)
    task_metrics: list[TaskMetrics] = field(default_factory=list)


@dataclass
class ProposerInput:
    """What gets sent to the proposer/reflection LLM."""

    iteration: int
    timestamp: str
    current_prompt: str
    reflection_records: list[dict[str, Any]]  # From make_reflective_dataset


class ExperimentLogger:
    """
    Comprehensive logging for GEPA experiments.

    Creates structured JSON files for later analysis:
    - iterations.jsonl: One line per iteration with all metrics
    - proposer_inputs.jsonl: What the proposer sees each iteration
    - prompts/: Directory with each prompt version
    - summary.json: Final experiment summary

    Each run gets its own folder: logs/run_YYYYMMDD_HHMMSS_<short_uuid>/
    """

    def __init__(self, log_dir: str = "gepa_results/logs", run_name: str | None = None, repo: str | None = None):
        # Generate unique run ID
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]

        if run_name:
            self.run_id = run_name
        elif repo:
            # Extract short repo name (e.g., "pygments" from "pygments__pygments")
            short_repo = repo.split("__")[1]
            self.run_id = f"run_{short_repo}_{timestamp}_{short_id}"
        else:
            self.run_id = f"run_{timestamp}_{short_id}"

        # Create run-specific directory
        self.base_log_dir = Path(log_dir)
        self.log_dir = self.base_log_dir / self.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.prompts_dir = self.log_dir / "prompts"
        self.prompts_dir.mkdir(exist_ok=True)

        # Initialize log files
        self.iterations_file = self.log_dir / "iterations.jsonl"
        self.proposer_file = self.log_dir / "proposer_inputs.jsonl"
        self.summary_file = self.log_dir / "summary.json"
        self.config_file = self.log_dir / "config.json"

        # Track experiment state
        self.eval_count = 0  # Number of evaluate() calls
        self.all_evals: list[EvalBatchMetrics] = []
        self.baseline_metrics: EvalBatchMetrics | None = None
        self.prompts_seen: dict[str, str] = {}  # prompt_hash -> full prompt

        # Create symlink to latest run for easy access
        latest_link = self.base_log_dir / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        if not latest_link.exists():
            latest_link.symlink_to(self.run_id)

        print("📊 Experiment logger initialized")
        print(f"   Run ID: {self.run_id}")
        print(f"   Run dir: {self.log_dir}")
        print(f"   Iterations: {self.iterations_file.name}")
        print(f"   Proposer inputs: {self.proposer_file.name}")
        print(f"   Prompts: {self.prompts_dir.name}/")
        print(f"   Latest symlink: {latest_link}")

    def save_config(self, config: dict[str, Any]):
        """Save experiment configuration at the start of the run."""
        config_with_meta = {"run_id": self.run_id, "start_time": self.start_time.isoformat(), **config}
        with open(self.config_file, "w") as f:
            json.dump(config_with_meta, f, indent=2)
        print(f"   Config saved: {self.config_file.name}")

    def _prompt_hash(self, prompt: str) -> str:
        """Short hash for prompt identification."""
        import hashlib

        return hashlib.md5(prompt.encode()).hexdigest()[:8]

    def log_eval_batch(
        self,
        prompt: str,
        outputs: list[dict[str, Any]],
        scores: list[float],
        task_ids: list[str],
        is_baseline: bool = False,
    ) -> EvalBatchMetrics:
        """
        Log metrics for one adapter.evaluate() call.

        Args:
            prompt: The system prompt used for this evaluation
            outputs: List of output dicts from adapter (with steps, tokens, etc.)
            scores: List of scores (0.0 or 1.0)
            task_ids: List of instance_ids
            is_baseline: Whether this is the baseline evaluation
        """
        timestamp = datetime.now().isoformat()
        prompt_hash = self._prompt_hash(prompt)
        eval_id = self.eval_count

        # Track unique prompts
        if prompt_hash not in self.prompts_seen:
            self.prompts_seen[prompt_hash] = prompt

        # Build per-task metrics
        task_metrics = []
        for _i, (output, score, task_id) in enumerate(zip(outputs, scores, task_ids, strict=False)):
            tm = TaskMetrics(
                instance_id=task_id,
                success=score == 1.0,
                score=score,
                steps=output.get("steps", 0),
                estimated_tokens=output.get("estimated_tokens", 0),
                has_patch=bool(output.get("patch", "").strip()),
            )
            task_metrics.append(tm)

        # Aggregate metrics
        num_passed = sum(1 for s in scores if s == 1.0)
        total_steps = sum(tm.steps for tm in task_metrics)
        total_tokens = sum(tm.estimated_tokens for tm in task_metrics)

        # Create prompt preview (first 100 chars, single line)
        prompt_preview = prompt.replace("\n", " ")[:100] + "..." if len(prompt) > 100 else prompt.replace("\n", " ")

        metrics = EvalBatchMetrics(
            eval_id=eval_id,
            timestamp=timestamp,
            prompt_hash=prompt_hash,
            prompt_preview=prompt_preview,
            num_tasks=len(scores),
            num_passed=num_passed,
            pass_rate=num_passed / len(scores) if scores else 0.0,
            avg_steps=total_steps / len(scores) if scores else 0.0,
            avg_tokens=total_tokens / len(scores) if scores else 0.0,
            total_steps=total_steps,
            total_tokens=total_tokens,
            task_metrics=task_metrics,
        )

        # Track baseline for comparison
        if is_baseline or eval_id == 0:
            self.baseline_metrics = metrics

        self.all_evals.append(metrics)
        self.eval_count += 1

        # Write to JSONL (one line per eval batch)
        with open(self.iterations_file, "a") as f:
            # Convert dataclass to dict, handling nested dataclasses
            data = asdict(metrics)
            f.write(json.dumps(data) + "\n")

        # Save prompt to file (only if new prompt)
        prompt_file = self.prompts_dir / f"prompt_{prompt_hash}.txt"
        if not prompt_file.exists():
            with open(prompt_file, "w") as f:
                f.write(prompt)

        # Print summary
        is_new_prompt = len(self.prompts_seen) > 1 and prompt_hash == list(self.prompts_seen.keys())[-1]
        label = "BASELINE" if is_baseline else f"Eval #{eval_id}"
        if is_new_prompt:
            label += " (NEW PROMPT)"
        print(f"\n📈 [{label}] {num_passed}/{len(scores)} passed ({metrics.pass_rate * 100:.1f}%)")
        print(f"   Avg steps: {metrics.avg_steps:.1f}, Avg tokens: {metrics.avg_tokens:.0f}")
        print(f"   Prompt: {prompt_hash} ({len(self.prompts_seen)} unique prompts seen)")

        return metrics

    def log_proposer_input(self, iteration: int, current_prompt: str, reflection_records: list[dict[str, Any]]):
        """
        Log what gets sent to the proposer/reflection LLM.
        """
        timestamp = datetime.now().isoformat()

        proposer_input = ProposerInput(
            iteration=iteration,
            timestamp=timestamp,
            current_prompt=current_prompt,
            reflection_records=reflection_records,
        )

        # Write to JSONL
        with open(self.proposer_file, "a") as f:
            data = asdict(proposer_input)
            f.write(json.dumps(data) + "\n")

        # Also save a readable version
        readable_file = self.prompts_dir / f"proposer_input_{iteration:03d}.txt"
        with open(readable_file, "w") as f:
            f.write(f"=== PROPOSER INPUT FOR ITERATION {iteration} ===\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            f.write(f"=== CURRENT PROMPT ===\n{current_prompt}\n\n")
            f.write(f"=== REFLECTION RECORDS ({len(reflection_records)} tasks) ===\n")
            for i, record in enumerate(reflection_records):
                f.write(f"\n--- Task {i + 1} ---\n")
                f.write(json.dumps(record, indent=2))
                f.write("\n")

        print(f"   📝 Proposer input logged ({len(reflection_records)} reflection records)")

    def get_comparison(self) -> dict[str, Any]:
        """Get before/after comparison metrics."""
        if not self.all_evals:
            return {}

        baseline = self.baseline_metrics or self.all_evals[0]
        latest = self.all_evals[-1]

        return {
            "baseline": {
                "eval_id": baseline.eval_id,
                "prompt_hash": baseline.prompt_hash,
                "pass_rate": baseline.pass_rate,
                "avg_steps": baseline.avg_steps,
                "avg_tokens": baseline.avg_tokens,
            },
            "latest": {
                "eval_id": latest.eval_id,
                "prompt_hash": latest.prompt_hash,
                "pass_rate": latest.pass_rate,
                "avg_steps": latest.avg_steps,
                "avg_tokens": latest.avg_tokens,
            },
            "improvement": {
                "pass_rate_delta": latest.pass_rate - baseline.pass_rate,
                "pass_rate_pct_change": ((latest.pass_rate - baseline.pass_rate) / baseline.pass_rate * 100)
                if baseline.pass_rate > 0
                else 0,
                "steps_delta": latest.avg_steps - baseline.avg_steps,
                "tokens_delta": latest.avg_tokens - baseline.avg_tokens,
            },
            "unique_prompts_tested": len(self.prompts_seen),
            "total_eval_batches": len(self.all_evals),
        }

    def save_summary(self, best_prompt: str | None = None, extra_info: dict | None = None):
        """Save final experiment summary."""
        comparison = self.get_comparison()

        summary = {
            "experiment_start": self.start_time.isoformat(),
            "experiment_end": datetime.now().isoformat(),
            "total_eval_batches": self.eval_count,
            "unique_prompts_tested": len(self.prompts_seen),
            "comparison": comparison,
            "best_prompt_hash": self._prompt_hash(best_prompt) if best_prompt else None,
            "prompts_seen": list(self.prompts_seen.keys()),  # List of all prompt hashes
            "extra_info": extra_info or {},
        }

        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'=' * 60}")
        if comparison:
            print(f"Total eval batches: {comparison.get('total_eval_batches', self.eval_count)}")
            print(f"Unique prompts tested: {comparison.get('unique_prompts_tested', len(self.prompts_seen))}")
            print()
            print(f"Baseline prompt: {comparison['baseline']['prompt_hash']}")
            print(f"Final prompt:    {comparison['latest']['prompt_hash']}")
            print()
            print(f"Baseline pass rate: {comparison['baseline']['pass_rate'] * 100:.1f}%")
            print(f"Final pass rate:    {comparison['latest']['pass_rate'] * 100:.1f}%")
            print(f"Improvement:        {comparison['improvement']['pass_rate_delta'] * 100:+.1f}%")
            print()
            print(f"Baseline avg steps:  {comparison['baseline']['avg_steps']:.1f}")
            print(f"Final avg steps:     {comparison['latest']['avg_steps']:.1f}")
            print(f"Steps delta:         {comparison['improvement']['steps_delta']:+.1f}")
            print()
            print(f"Baseline avg tokens: {comparison['baseline']['avg_tokens']:.0f}")
            print(f"Final avg tokens:    {comparison['latest']['avg_tokens']:.0f}")
            print(f"Tokens delta:        {comparison['improvement']['tokens_delta']:+.0f}")
        print(f"\nSummary saved to: {self.summary_file}")

        return summary


# Global logger instance (set during experiment)
_experiment_logger: ExperimentLogger | None = None


def get_logger() -> ExperimentLogger | None:
    """Get the global experiment logger."""
    return _experiment_logger


def set_logger(logger: ExperimentLogger):
    """Set the global experiment logger."""
    global _experiment_logger
    _experiment_logger = logger
