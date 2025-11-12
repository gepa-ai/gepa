# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
MiniSWEAgent adapter for GEPA.

This adapter integrates mini-swe-agent with GEPA for optimizing agent configurations
on SWE-bench tasks. It supports both agent execution and optional validation using
the SWE-bench harness.

Requirements:
    - mini-swe-agent (included in the repo)
    - swebench (optional, for validation): pip install swebench

Usage:
    from gepa.adapters.minisweagent_adapter import MiniSWEAgentAdapter
    
    adapter = MiniSWEAgentAdapter(
        model_name="anthropic/claude-sonnet-4",
        agent_config_path="path/to/config.yaml",
        run_validation=True  # Set to False to skip validation
    )
"""

import json
import subprocess
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypedDict

import yaml
from datasets import load_dataset

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


class MiniSWEAgentDataInst(TypedDict):
    """
    Data instance for SWE-bench tasks.
    
    Attributes:
        instance_id: Unique identifier for the SWE-bench instance (e.g., "django__django-12345")
        problem_statement: The issue description/problem to solve
        base_commit: The base commit hash
        patch: The ground truth patch (for reference)
        test_patch: The test patch to validate the solution
        repo: Repository name
        version: Version/commit of the codebase
        environment_setup_commit: Commit for environment setup
        hints_text: Optional hints
        created_at: Creation timestamp
        FAIL_TO_PASS: Tests that should pass after fix
        PASS_TO_PASS: Tests that should continue passing
        image_name: Docker image name (optional)
    """

    instance_id: str
    problem_statement: str
    base_commit: str
    patch: str
    test_patch: str
    repo: str
    version: str
    environment_setup_commit: str
    hints_text: str
    created_at: str
    FAIL_TO_PASS: str
    PASS_TO_PASS: str
    image_name: str | None


class MiniSWEAgentRolloutOutput(TypedDict):
    """
    Output from a single agent rollout.
    
    Attributes:
        instance_id: The instance ID
        model_patch: The generated patch/diff
        exit_status: How the agent terminated (e.g., "Submitted", "LimitsExceeded")
        submission: The final submission message
        n_calls: Number of model calls made
        cost: Total cost of the rollout
        validation_result: Optional validation results from SWE-bench harness
    """

    instance_id: str
    model_patch: str
    exit_status: str
    submission: str
    n_calls: int
    cost: float
    validation_result: dict[str, Any] | None


class MiniSWEAgentTrajectory(TypedDict):
    """
    Complete trajectory of an agent rollout for reflective learning.
    
    Attributes:
        data: The input data instance
        messages: The full message history (system, user, assistant)
        exit_status: How the agent terminated
        submission: The final submission
        model_patch: The generated patch
        n_calls: Number of model calls
        cost: Cost of the rollout
        error_message: Error message if any
        validation_result: Optional validation results
    """

    data: MiniSWEAgentDataInst
    messages: list[dict[str, str]]
    exit_status: str
    submission: str
    model_patch: str
    n_calls: int
    cost: float
    error_message: str | None
    validation_result: dict[str, Any] | None


class MiniSWEAgentAdapter(GEPAAdapter[MiniSWEAgentDataInst, MiniSWEAgentTrajectory, MiniSWEAgentRolloutOutput]):
    """
    GEPA adapter for mini-swe-agent on SWE-bench tasks.
    
    This adapter runs the mini-swe-agent on SWE-bench instances and optionally validates
    the generated patches using the SWE-bench harness. It supports optimizing various
    agent configuration components like system prompts, instance templates, and more.
    
    The candidate dictionary passed to evaluate() contains the text components being optimized
    (e.g., {"system_template": "You are...", "instance_template": "Task: {{task}}..."}),
    which are merged into the base agent configuration to build the complete agent.
    
    Args:
        model_name: The model to use (e.g., "anthropic/claude-sonnet-4")
        model_class: The model class to use (e.g., "anthropic" or full path)
        agent_config_base: Base agent configuration dictionary with default values (optional)
        agent_config_path: Path to base agent config YAML file (optional)
        environment_class: Environment type (e.g., "docker", "singularity", "local")
        environment_config: Additional environment configuration
        run_validation: Whether to run SWE-bench validation (requires swebench package)
        validation_max_workers: Number of workers for validation
        max_workers: Number of parallel workers for agent execution (default: 4)
        timeout: Timeout for agent execution in seconds
        failure_score: Score to assign to failed rollouts
        temp_dir: Temporary directory for storing outputs
    """

    def __init__(
        self,
        model_name: str,
        model_class: str | None = None,
        agent_config_base: dict[str, Any] | None = None,
        agent_config_path: str | Path | None = None,
        environment_class: str = "docker",
        environment_config: dict[str, Any] | None = None,
        run_validation: bool = False,
        validation_max_workers: int = 8,
        max_workers: int = 8,
        timeout: int = 300,
        failure_score: float = 0.0,
        temp_dir: str | Path | None = None,
    ):
        self.model_name = model_name
        self.model_class = model_class
        self.environment_class = environment_class
        self.environment_config = environment_config or {}
        self.run_validation = run_validation
        self.validation_max_workers = validation_max_workers
        self.max_workers = max_workers
        self.timeout = timeout
        self.failure_score = failure_score
        
        # Setup base configuration
        if agent_config_path is not None:
            with open(agent_config_path) as f:
                self.agent_config_base = yaml.safe_load(f)
        elif agent_config_base is not None:
            self.agent_config_base = agent_config_base
        else:
            # Use default mini-swe-agent config structure
            self.agent_config_base = {
                "agent": {},
                "model": {},
                "environment": {},
            }
        
        # Setup temp directory
        if temp_dir is None:
            self.temp_dir = Path(tempfile.gettempdir()) / "gepa_minisweagent"
        else:
            self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _build_config(self, candidate: dict[str, str]) -> dict[str, Any]:
        """
        Build a complete agent config from the base config and candidate components.
        
        The candidate contains optimized text components (e.g., {"system_template": "...", "instance_template": "..."})
        which are merged into the base agent configuration.
        """
        import copy

        config = copy.deepcopy(self.agent_config_base)
        
        # Update model settings
        config.setdefault("model", {})
        config["model"]["model_name"] = self.model_name
        if self.model_class is not None:
            config["model"]["model_class"] = self.model_class
        
        # Update environment settings
        config.setdefault("environment", {})
        config["environment"]["environment_class"] = self.environment_class
        if self.timeout:
            config["environment"]["timeout"] = self.timeout
        config["environment"].update(self.environment_config)
        
        # Update agent settings with candidate components
        # Candidate contains the optimized text components
        config.setdefault("agent", {})
        for component_name, component_text in candidate.items():
            # Map component names to agent config keys
            if component_name in ["system_template", "instance_template", "format_error_template",
                                   "action_observation_template", "timeout_template"]:
                # These are template strings used by the agent
                config["agent"][component_name] = component_text
            elif component_name == "step_limit":
                config["agent"]["step_limit"] = int(component_text)
            elif component_name == "cost_limit":
                config["agent"]["cost_limit"] = float(component_text)
            else:
                # Handle generic components - store them in agent config
                config["agent"][component_name] = component_text
        
        return config

    def _get_swebench_docker_image_name(self, instance: dict) -> str:
        """Get the Docker image name for a SWE-bench instance."""
        image_name = instance.get("image_name", None)
        if image_name is None:
            iid = instance["instance_id"]
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        return image_name

    def _run_agent_on_instance(
        self, instance: dict, config: dict[str, Any], capture_traces: bool
    ) -> tuple[MiniSWEAgentRolloutOutput, MiniSWEAgentTrajectory | None]:
        """
        Run the agent on a single SWE-bench instance.
        
        Note: This method catches all exceptions to comply with GEPA's adapter contract,
        which requires returning valid outputs even for failed instances rather than raising.
        """
        from minisweagent.agents.default import DefaultAgent
        from minisweagent.environments import get_environment
        from minisweagent.models import get_model

        instance_id = instance["instance_id"]
        task = instance["problem_statement"]
        
        # Update environment config with instance-specific image
        env_config = config.get("environment", {}).copy()
        image_name = self._get_swebench_docker_image_name(instance)
        if env_config["environment_class"] == "docker":
            env_config["image"] = image_name
        elif env_config["environment_class"] == "singularity":
            env_config["image"] = "docker://" + image_name
        
        agent = None
        exit_status = "Unknown"
        submission = ""
        error_message = None
        
        try:
            # Create model and environment
            model = get_model(config=config.get("model", {}))
            env = get_environment(env_config)
            
            # Create and run agent
            agent = DefaultAgent(model, env, **config.get("agent", {}))
            exit_status, submission = agent.run(task)
            
        except Exception as e:
            # Must catch to comply with GEPA contract: never raise for individual failures
            exit_status = type(e).__name__
            submission = str(e)
            error_message = traceback.format_exc()
        
        # Extract results
        n_calls = agent.model.n_calls if agent else 0
        cost = agent.model.cost if agent else 0.0
        messages = agent.messages if agent else []
        
        # Build output
        output: MiniSWEAgentRolloutOutput = {
            "instance_id": instance_id,
            "model_patch": submission,
            "exit_status": exit_status,
            "submission": submission,
            "n_calls": n_calls,
            "cost": cost,
            "validation_result": None,
        }
        
        # Build trajectory if requested
        trajectory: MiniSWEAgentTrajectory | None = None
        if capture_traces:
            trajectory = {
                "data": instance,  # type: ignore
                "messages": messages,
                "exit_status": exit_status,
                "submission": submission,
                "model_patch": submission,
                "n_calls": n_calls,
                "cost": cost,
                "error_message": error_message,
                "validation_result": None,
            }
        
        return output, trajectory

    def _run_validation(
        self, outputs: list[MiniSWEAgentRolloutOutput], batch: list[MiniSWEAgentDataInst]
    ) -> list[dict[str, Any]]:
        """
        Run SWE-bench validation on the generated patches.
        
        Returns a list of validation results, one per instance.
        
        Raises:
            ImportError: If swebench package is not installed
            subprocess.TimeoutExpired: If validation times out
            subprocess.CalledProcessError: If validation command fails
        """
        # Check if swebench is available (optional dependency)
        try:
            import swebench  # noqa: F401  # type: ignore
        except ImportError as e:
            raise ImportError(
                "swebench package is required for validation. "
                "Install it with: pip install swebench"
            ) from e
        
        # Create predictions file in SWE-bench format
        predictions = []
        for output in outputs:
            predictions.append({
                "instance_id": output["instance_id"],
                "model_name_or_path": self.model_name,
                "model_patch": output["model_patch"],
            })
        
        # Write predictions to temporary file
        preds_file = self.temp_dir / "predictions.jsonl"
        with open(preds_file, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        
        # Determine dataset name from instance IDs
        # This is a heuristic - you may need to pass the dataset name explicitly
        dataset_name = "princeton-nlp/SWE-bench_Verified"  # Default
        
        # Run evaluation using swebench harness
        results_dir = self.temp_dir / "validation_results"
        results_dir.mkdir(exist_ok=True)
        
        # Run the SWE-bench harness
        cmd = [
            "python", "-m", "swebench.harness.run_evaluation",
            "--dataset_name", dataset_name,
            "--predictions_path", str(preds_file),
            "--max_workers", str(self.validation_max_workers),
            "--run_id", "gepa_validation",
        ]
        
        subprocess.run(
            cmd,
            cwd=results_dir,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout for validation
            check=True,  # Raise if command fails
        )
        
        # Parse validation results
        validation_results = []
        results_file = results_dir / "gepa_validation" / "results.json"
        if results_file.exists():
            with open(results_file) as f:
                results_data = json.load(f)
                for output in outputs:
                    instance_id = output["instance_id"]
                    instance_result = results_data.get(instance_id, {})
                    validation_results.append(instance_result)
        else:
            # If results file doesn't exist, return empty results
            validation_results = [{} for _ in outputs]
        
        return validation_results

    def evaluate(
        self,
        batch: list[MiniSWEAgentDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[MiniSWEAgentTrajectory, MiniSWEAgentRolloutOutput]:
        """
        Evaluate the candidate agent configuration on a batch of SWE-bench instances.
        
        Args:
            batch: List of SWE-bench instances to evaluate
            candidate: Dictionary mapping component names to their optimized text values.
                      For example: {"system_template": "You are...", "instance_template": "Task: {{task}}..."}
                      These are merged with the base agent config to build the complete agent.
            capture_traces: Whether to capture full trajectories for reflection
        
        Returns:
            EvaluationBatch containing outputs, scores, and optionally trajectories
        """
        config = self._build_config(candidate)
        
        outputs: list[MiniSWEAgentRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[MiniSWEAgentTrajectory] | None = [] if capture_traces else None
        
        # Run agent on each instance in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all instances to the executor
            future_to_instance = {
                executor.submit(self._run_agent_on_instance, instance, config, capture_traces): instance
                for instance in batch
            }
            
            # Collect results as they complete
            results = []
            for future in as_completed(future_to_instance):
                instance = future_to_instance[future]
                try:
                    output, trajectory = future.result()
                    results.append((instance, output, trajectory))
                except Exception as e:
                    # Handle unexpected errors (should be rare as _run_agent_on_instance catches exceptions)
                    instance_id = instance.get("instance_id", "unknown")
                    print(f"Unexpected error processing instance {instance_id}: {e}")
                    # Create a failed output
                    output: MiniSWEAgentRolloutOutput = {
                        "instance_id": instance_id,
                        "model_patch": "",
                        "exit_status": "UnexpectedError",
                        "submission": str(e),
                        "n_calls": 0,
                        "cost": 0.0,
                        "validation_result": None,
                    }
                    trajectory = None
                    results.append((instance, output, trajectory))
            
            # Sort results to maintain original batch order
            instance_to_result = {r[0]["instance_id"]: (r[1], r[2]) for r in results}
            for instance in batch:
                instance_id = instance["instance_id"]
                output, trajectory = instance_to_result[instance_id]
                
                # Determine score based on exit status
                # Successful submission gets a base score, validation can refine it
                if output["exit_status"] == "Submitted":
                    score = 1.0
                else:
                    score = self.failure_score
                
                outputs.append(output)
                scores.append(score)
                if trajectories is not None and trajectory is not None:
                    trajectories.append(trajectory)
        
        # Run validation if requested
        if self.run_validation:
            validation_results = self._run_validation(outputs, batch)
            
            # Update outputs and scores with validation results
            for i, (output, validation_result) in enumerate(zip(outputs, validation_results)):
                output["validation_result"] = validation_result
                
                # Update score based on validation
                if validation_result:
                    # Check if the instance was resolved
                    resolved = validation_result.get("resolved", False)
                    if resolved:
                        scores[i] = 1.0
                    else:
                        # Partial credit based on tests passed
                        tests_passed = validation_result.get("tests_passed", 0)
                        total_tests = validation_result.get("total_tests", 1)
                        scores[i] = tests_passed / max(total_tests, 1)
                
                # Update trajectory if captured
                if trajectories is not None:
                    trajectories[i]["validation_result"] = validation_result
        
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[MiniSWEAgentTrajectory, MiniSWEAgentRolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """
        Build a reflective dataset for instruction refinement.
        
        Args:
            candidate: The current candidate configuration
            eval_batch: Results from evaluate() with capture_traces=True
            components_to_update: List of component names to update
        
        Returns:
            Dictionary mapping component names to lists of reflective examples
        """
        if eval_batch.trajectories is None:
            raise ValueError("Trajectories are required to build a reflective dataset. "
                           "Call evaluate() with capture_traces=True.")
        
        reflective_dataset: dict[str, list[dict[str, Any]]] = {
            component: [] for component in components_to_update
        }
        
        for trajectory, score, output in zip(
            eval_batch.trajectories, eval_batch.scores, eval_batch.outputs, strict=False
        ):
            instance = trajectory["data"]
            instance_id = instance["instance_id"]
            problem_statement = instance["problem_statement"]
            
            # Build feedback based on results
            if score >= 1.0:
                feedback = (
                    f"✓ Successfully solved the task! "
                    f"The agent generated a correct patch in {trajectory['n_calls']} steps."
                )
            elif score > 0.0:
                feedback = (
                    f"⚠ Partially solved the task (score: {score:.2f}). "
                    f"Exit status: {trajectory['exit_status']}. "
                )
                if trajectory["validation_result"]:
                    validation = trajectory["validation_result"]
                    if "tests_passed" in validation:
                        feedback += (
                            f"Passed {validation['tests_passed']}/{validation['total_tests']} tests. "
                        )
            else:
                feedback = (
                    f"✗ Failed to solve the task. "
                    f"Exit status: {trajectory['exit_status']}. "
                    f"Reason: {trajectory['submission']}"
                )
                if trajectory["error_message"]:
                    feedback += f"\n\nError details:\n{trajectory['error_message']}"
            
            # Add hints and context
            if instance.get("hints_text"):
                feedback += f"\n\nHint: {instance['hints_text']}"
            
            # Create reflective record for each component
            for component in components_to_update:
                record: dict[str, Any] = {
                    "Instance ID": instance_id,
                    "Inputs": {
                        "problem_statement": problem_statement,
                        "repo": instance.get("repo", ""),
                        "version": instance.get("version", ""),
                    },
                    "Current Component Text": candidate.get(component, ""),
                    "Generated Outputs": {
                        "patch": trajectory["model_patch"][:500],  # Truncate for brevity
                        "exit_status": trajectory["exit_status"],
                        "n_calls": trajectory["n_calls"],
                        "cost": trajectory["cost"],
                    },
                    "Feedback": feedback,
                    "Score": score,
                }
                
                # Add message history for context (truncated)
                if trajectory["messages"]:
                    # Include first and last few messages
                    messages = trajectory["messages"]
                    if len(messages) > 6:
                        sample_messages = messages[:3] + ["..."] + messages[-3:]  # type: ignore
                    else:
                        sample_messages = messages
                    record["Sample Message History"] = sample_messages
                
                reflective_dataset[component].append(record)
        
        return reflective_dataset


def load_swebench_instances(
    dataset: str = "princeton-nlp/SWE-bench_Verified",
    split: str = "test",
    slice_spec: str = "",
) -> list[MiniSWEAgentDataInst]:
    """
    Helper function to load SWE-bench instances as MiniSWEAgentDataInst.
    
    Args:
        dataset: The dataset name or path
        split: The dataset split (train/dev/test)
        slice_spec: Slice specification (e.g., "0:10" for first 10 instances)
    
    Returns:
        List of MiniSWEAgentDataInst instances
    """
    instances = list(load_dataset(dataset, split=split))
    
    if slice_spec:
        values = [int(x) if x else None for x in slice_spec.split(":")]
        instances = instances[slice(*values)]
    
    # Convert to MiniSWEAgentDataInst format
    data_instances: list[MiniSWEAgentDataInst] = []
    for instance in instances:
        # Ensure all required fields are present
        data_inst: MiniSWEAgentDataInst = {
            "instance_id": instance.get("instance_id", ""),
            "problem_statement": instance.get("problem_statement", ""),
            "base_commit": instance.get("base_commit", ""),
            "patch": instance.get("patch", ""),
            "test_patch": instance.get("test_patch", ""),
            "repo": instance.get("repo", ""),
            "version": instance.get("version", ""),
            "environment_setup_commit": instance.get("environment_setup_commit", ""),
            "hints_text": instance.get("hints_text", ""),
            "created_at": instance.get("created_at", ""),
            "FAIL_TO_PASS": instance.get("FAIL_TO_PASS", ""),
            "PASS_TO_PASS": instance.get("PASS_TO_PASS", ""),
            "image_name": instance.get("image_name"),
        }
        data_instances.append(data_inst)
    
    return data_instances

