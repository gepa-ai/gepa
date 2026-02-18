"""
Fitness function for GEPA optimize_anything API.

This wraps the SWE harness logic into a simple fitness function that:
- Takes a candidate (prompt) and batch of tasks
- Returns (score, output, side_info) tuples for each task
"""

import threading
from typing import Any

from gskill.swe_harness import SWEHarness


def create_swe_fitness_fn(model_name: str = "gpt-5-mini", n_workers: int = 6, config_path: str | None = None):
    """Create a fitness function for SWE tasks with Docker containers.

    Args:
        model_name: LiteLLM model name
        n_workers: Number of parallel workers
        config_path: Optional custom config path for the agent (defaults to mini.yaml)

    Returns:
        fitness_fn: Function that evaluates candidates on batches
    """

    # Create harness pool - one per worker
    harness_pool = [SWEHarness() for _ in range(n_workers)]
    harness_available = [True] * n_workers
    harness_lock = threading.Lock()

    def get_harness() -> tuple[int, SWEHarness]:
        """Get an available harness from the pool."""
        with harness_lock:
            for i, available in enumerate(harness_available):
                if available:
                    harness_available[i] = False
                    return i, harness_pool[i]
        raise RuntimeError("No harness available - pool exhausted")

    def release_harness(idx: int):
        """Release a harness back to the pool."""
        with harness_lock:
            harness_available[idx] = True

    def process_single_task(
        task: dict[str, Any], skills: str, task_idx: int, total_tasks: int
    ) -> tuple[float, dict[str, Any], dict[str, Any]]:
        """Process a single task and return (score, output, side_info)."""

        # Get a harness from the pool
        harness_idx, harness = get_harness()

        instance_id = task.get("instance_id", "unknown")[:50]

        try:
            print(f"[Task {task_idx + 1}/{total_tasks}] {instance_id}...", flush=True)

            # 1. Setup Docker Container
            harness.setup_task(task_instance=task)

            # 2. Run Agent
            problem = task["problem_statement"]
            patch, agent_trace, agent_metrics = harness.run_agent(
                problem, skills, model_name=model_name, config_path=config_path
            )

            # 3. Verify with tests
            has_patch = len(patch.strip()) > 0
            passed = False
            feedback_msg = ""
            test_output = ""

            if not has_patch:
                passed = False
                feedback_msg = "no_patch"
                test_output = "No patch to test."
            else:
                # Use SWE-smith's run_patch_in_container for proper verification
                # This creates a fresh container, checks out HEAD~1 (with tests),
                # applies the patch, and runs tests
                # First test FAIL_TO_PASS only
                f2p_passed, f2p_output = harness.verify_with_patch(patch, f2p_only=True)
                test_output = f"=== FAIL_TO_PASS TESTS ===\n{f2p_output}"

                if not f2p_passed:
                    passed = False
                    feedback_msg = "f2p_failed"
                else:
                    # PASS_TO_PASS Tests (Regression Check) - run full test suite
                    pass_to_pass = task.get("PASS_TO_PASS", [])

                    if pass_to_pass:
                        # Run full test (includes both f2p and p2p)
                        p2p_passed, p2p_output = harness.verify_with_patch(patch, f2p_only=False)
                        test_output += f"\n\n=== FULL TEST SUITE ===\n{p2p_output}"

                        if not p2p_passed:
                            passed = False
                            feedback_msg = "p2p_regression"
                        else:
                            passed = True
                            feedback_msg = "all_passed"
                    else:
                        passed = True
                        feedback_msg = "f2p_passed"

            score = 1.0 if passed else 0.0

            # Rollout output
            output = {
                "patch": patch,
                "success": passed,
                "steps": agent_metrics.get("steps", 0),
                "estimated_tokens": agent_metrics.get("estimated_tokens", 0),
            }

            # Side info for reflection
            side_info = {
                "Input": {
                    "Task ID": instance_id,
                    "Problem": problem[:200] + "..." if len(problem) > 200 else problem,
                },
                "Generated Outputs": {
                    "Patch": patch[:500] + "..." if len(patch) > 500 else patch,
                    "Agent Trace": agent_trace,
                },
                "Feedback": {
                    "Status": feedback_msg,
                    "Test Output": test_output[:500] + "..." + test_output[-500:]
                    if len(test_output) > 1000
                    else test_output,
                },
                "scores": {
                    "correctness": score,
                },
            }

            status = "✓ PASS" if passed else f"✗ FAIL ({feedback_msg})"
            print(f"  [{instance_id}] {status} [steps: {agent_metrics.get('steps', 0)}]", flush=True)

            # Show brief debug info on failure
            if not passed:
                # Extract last few lines of test output for quick debugging
                output_lines = test_output.strip().split("\n")
                # Get last 5 non-empty lines
                relevant_lines = [line for line in output_lines if line.strip()][-5:]
                if relevant_lines:
                    print(f"    Last output: {relevant_lines[-1][:100]}", flush=True)

            # Cleanup
            harness.cleanup()

            return score, output, side_info

        except Exception as e:
            # Handle setup failures (e.g., git checkout failed, Docker issues)
            error_msg = str(e)
            print(f"  [{instance_id[:30]}] ✗ FAIL (setup_error)", flush=True)
            print(f"    Error: {error_msg[:100]}", flush=True)

            # Return failure with error info
            output = {"patch": "", "success": False, "steps": 0, "estimated_tokens": 0, "error": error_msg}
            side_info = {
                "Input": {
                    "Task ID": instance_id,
                    "Problem": task.get("problem_statement", "")[:200],
                },
                "Generated Outputs": {
                    "Patch": "",
                    "Agent Trace": "",
                },
                "Feedback": {
                    "Status": "setup_error",
                    "Test Output": f"Setup failed: {error_msg}",
                },
                "scores": {
                    "correctness": 0.0,
                },
            }

            try:
                harness.cleanup()
            except Exception:
                pass

            return 0.0, output, side_info

        finally:
            release_harness(harness_idx)

    def fitness_fn(
        candidate: dict[str, str],
        example: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Evaluate candidate on a single task (called per-example by GEPA).

        Args:
            candidate: Dict with 'skills' key
            example: A single task instance dict

        Returns:
            (score, side_info) tuple
        """
        skills = candidate["skills"]
        score, _output, side_info = process_single_task(example, skills, 0, 1)
        return score, side_info

    return fitness_fn
