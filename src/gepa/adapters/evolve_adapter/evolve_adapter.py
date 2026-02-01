from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from openevolve.evaluation_result import EvaluationResult  # type: ignore

from gepa import EvaluationBatch, GEPAAdapter
from gepa.adapters.evolve_adapter.evolve_program_proposal_signature import (
    EvolveProgramProposalSignature,
)


def _process_evaluation_result(result: Any) -> EvaluationResult:
    """
    Process evaluation result to handle both dict and EvaluationResult returns

    Args:
        result: Raw result from evaluation function

    Returns:
        EvaluationResult instance
    """
    if isinstance(result, dict):
        # Wrap dict in EvaluationResult
        return EvaluationResult.from_dict(result)
    elif isinstance(result, EvaluationResult):
        return result
    else:
        # Error case - return error metrics
        logging.warning(f"Unexpected evaluation result type: {type(result)}")
        return EvaluationResult(metrics={"error": 0.0})


def _passes_threshold(metrics: dict[str, float], threshold: float) -> bool:
    """
    Check if metrics pass a threshold

    Uses 'combined_score' if available (for consistency with evolution),
    otherwise falls back to averaging all numeric metrics except 'error'

    Args:
        metrics: Dictionary of metric name to score
        threshold: Threshold to pass

    Returns:
        True if metrics pass threshold
    """
    if not metrics:
        return False

    # Use combined_score if available - this is what evolution uses
    if "combined_score" in metrics:
        score = metrics.get("combined_score")
        if isinstance(score, (int, float)):
            return float(score) >= threshold

    # Fallback: average all numeric metrics except 'error'
    # This maintains backward compatibility
    valid_metrics = []
    for name, value in metrics.items():
        # Skip 'error' keys and ensure values are numeric
        if name != "error" and isinstance(value, (int, float)):
            try:
                valid_metrics.append(float(value))
            except (TypeError, ValueError):
                logging.warning(f"Skipping non-numeric metric: {name}={value}")
                continue

    if not valid_metrics:
        return False

    avg_score = sum(valid_metrics) / len(valid_metrics)
    return avg_score >= threshold


class EvaluationStrategy:
    def evaluate(self, program_path: str, batch: list[Any]) -> list[EvaluationResult]:
        """
        Evaluate the program on a batch of data instances.

        Args:
            program_path: Path to the program file to evaluate
            batch: List of data instances

        Returns:
            List of EvaluationResult objects, one per batch item
        """
        raise NotImplementedError


class DefaultEvaluationStrategy(EvaluationStrategy):
    def __init__(self, path: Path):
        self.path = path
        spec = importlib.util.spec_from_file_location("evaluator", self.path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load evaluator module from {self.path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.path.stem] = module
        spec.loader.exec_module(module)
        if not hasattr(module, "evaluate"):
            raise AttributeError(f"evaluate function not found in {self.path}")
        self.module = module.evaluate

    def evaluate(self, program_path: str, batch: list[Any]) -> list[EvaluationResult]:
        """
        Evaluate the program on a batch of data instances.

        The user's evaluate function should take (program_path: str, data_instance: Any)
        and return a single EvaluationResult. This method calls
        the user's evaluate function for each instance in the batch.
        """
        eval_results = []

        for i, data_instance in enumerate(batch):
            try:
                # Call user's evaluate function with single data instance
                result = self.module(program_path, data_instance)

                # Process the result (handles both dict and EvaluationResult)
                eval_result = _process_evaluation_result(result)
                eval_results.append(eval_result)
            except Exception as e:
                logging.error(f"Error evaluating batch item {i}: {e}")
                # Return error result for this item
                eval_results.append(
                    EvaluationResult(
                        metrics={"passed": 0.0, "error": 0.0},
                        artifacts={
                            "stderr": str(e),
                            "traceback": traceback.format_exc(),
                            "batch_item_index": i,
                        },
                    )
                )

        return eval_results


class CascadeEvaluationStrategy(EvaluationStrategy):
    def __init__(self, path: Path, cascade_thresholds: list[float]):
        self.path = path
        self.stages = self.load_stages()
        self.cascade_thresholds = cascade_thresholds

    def load_stages(self) -> dict[str, Callable]:
        required_stages = ["evaluate_stage1"]
        possible_stages = ["evaluate_stage2", "evaluate_stage3"]
        stages = {}
        # import
        spec = importlib.util.spec_from_file_location("evaluator", self.path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load evaluator module from {self.path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[self.path.stem] = module
        spec.loader.exec_module(module)
        for stage in required_stages:
            if hasattr(module, stage):
                stages[stage] = getattr(module, stage)
            else:
                raise AttributeError(f"Required stage {stage} not found in {self.path}")
        for stage in possible_stages:
            if hasattr(module, stage):
                stages[stage] = getattr(module, stage)
        return stages

    def evaluate(self, program_path: str, batch: list[Any]) -> list[EvaluationResult]:
        """
        Evaluate in cascading stages using the thresholds.

        The user's evaluate function should take (program_path: str, data_instance: Any)
        and return a single EvaluationResult. This method calls
        the user's evaluate stage functions for each instance.
        """
        # Run stage 1 for all instances
        stage1 = self.stages["evaluate_stage1"]
        stage1_eval_results = []

        for i, data_instance in enumerate(batch):
            try:
                # Call user's stage1 function with single data instance
                stage1_result = stage1(program_path, data_instance)
                eval_result = _process_evaluation_result(stage1_result)
                stage1_eval_results.append(eval_result)
            except Exception as e:
                logging.error(f"Error in stage 1 evaluation for batch item {i}: {e!s}")
                # Return error result for this item
                stage1_eval_results.append(
                    EvaluationResult(
                        metrics={"stage1_passed": 0.0, "error": 0.0},
                        artifacts={
                            "stderr": str(e),
                            "traceback": traceback.format_exc(),
                            "batch_item_index": i,
                        },
                    )
                )

        # Initialize final results with stage1
        final_results = []
        stage2_indices = []
        stage3_indices = []

        for i, stage1_result in enumerate(stage1_eval_results):
            final_results.append(stage1_result)

            if "evaluate_stage2" in self.stages:
                if _passes_threshold(stage1_result.metrics, self.cascade_thresholds[0]):
                    stage2_indices.append(i)
                    # Check stage 3
                    if len(self.cascade_thresholds) > 1:
                        if _passes_threshold(stage1_result.metrics, self.cascade_thresholds[1]):
                            stage3_indices.append(i)
                else:
                    logging.warning(f"Batch item {i} failed to meet stage 1 threshold")

        # Run stage2 for qualifying instances
        if stage2_indices and "evaluate_stage2" in self.stages:
            stage2 = self.stages["evaluate_stage2"]

            for idx in stage2_indices:
                try:
                    # Call user's stage2 function with single data instance
                    stage2_result = stage2(program_path, batch[idx])
                    stage2_eval_result = _process_evaluation_result(stage2_result)

                    # Merge stage2 results back into final_results
                    merged_metrics = {}
                    # Convert all values to float to avoid type errors
                    for name, value in final_results[idx].metrics.items():
                        if isinstance(value, (int, float)) and name != "error":
                            merged_metrics[name] = float(value)
                    for name, value in stage2_eval_result.metrics.items():
                        if isinstance(value, (int, float)) and name != "error":
                            merged_metrics[name] = float(value)

                    # Merge artifacts
                    merged_artifacts = {}
                    merged_artifacts.update(final_results[idx].artifacts)
                    merged_artifacts.update(stage2_eval_result.artifacts)

                    final_results[idx] = EvaluationResult(metrics=merged_metrics, artifacts=merged_artifacts)
                except Exception as e:
                    logging.error(f"Error in stage 2 evaluation for batch item {idx}: {e!s}")
                    # Mark stage2 as failed for this instance
                    final_results[idx].metrics["stage2_passed"] = 0.0
                    final_results[idx].artifacts.update(
                        {
                            "stage2_stderr": str(e),
                            "stage2_traceback": traceback.format_exc(),
                        }
                    )

        # Run stage3 for qualifying instances
        if stage3_indices and "evaluate_stage3" in self.stages:
            stage3 = self.stages["evaluate_stage3"]

            for idx in stage3_indices:
                try:
                    # Call user's stage3 function with single data instance
                    stage3_result = stage3(program_path, batch[idx])
                    stage3_eval_result = _process_evaluation_result(stage3_result)

                    # Merge stage3 results back into final_results
                    for name, value in stage3_eval_result.metrics.items():
                        if isinstance(value, (int, float)) and name != "error":
                            final_results[idx].metrics[name] = float(value)

                    # Merge artifacts
                    final_results[idx].artifacts.update(stage3_eval_result.artifacts)
                except Exception as e:
                    logging.error(f"Error in stage 3 for batch item {idx}: {e}")
                    # Capture stage 3 failure, but keep previous results
                    final_results[idx].artifacts.update(
                        {
                            "stage3_stderr": str(e),
                            "stage3_traceback": traceback.format_exc(),
                            "failure_stage": "stage3",
                        }
                    )
                    final_results[idx].metrics["stage3_passed"] = 0.0

        return final_results


class EvolveAdapter(GEPAAdapter):
    def __init__(
        self,
        path: Path,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.path = path
        self.config = yaml.safe_load(open(path / "config.yaml"))
        self.cascade = self.config["evaluator"].get("cascade_evaluation", False)
        self.evaluator_path = path / "evaluator.py"
        self.temp_env_path = Path(tempfile.mkdtemp())
        self.evaluation_strategy = (
            CascadeEvaluationStrategy(self.evaluator_path, self.config["evaluator"]["cascade_thresholds"])
            if self.cascade
            else DefaultEvaluationStrategy(self.evaluator_path)
        )

        # Store the original program template for reconstructing complete programs
        # This preserves fixed code outside EVOLVE-BLOCK markers
        self.original_program_template = None
        initial_program_path = path / "initial_program.py"
        if initial_program_path.exists():
            with open(initial_program_path) as f:
                self.original_program_template = f.read()

        # Pick the model with the highest weight (first if already sorted)
        if "models" in self.config["llm"]:
            primary = sorted(self.config["llm"]["models"], key=lambda m: m["weight"], reverse=True)[0]
            model_name = primary["name"]
            api_key = primary.get("api_key") or self.config["llm"].get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key not found. Set it in config.yaml (llm.api_key) or environment variable (OPENAI_API_KEY)"
                )
            api_base = primary.get("api_base") or self.config["llm"].get("api_base")
            temperature = primary.get("temperature", 0.7)
            top_p = primary.get("top_p", 1.0)
            max_tokens = primary.get("max_tokens", 4000)
        else:
            # Sensible fall-back
            model_name = self.config["llm"].get("primary_model") or "gpt-3.5-turbo"
            api_key = self.config["llm"].get("api_key") or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key not found. Set it in config.yaml (llm.api_key) or environment variable (OPENAI_API_KEY)"
                )
            api_base = self.config["llm"].get("api_base")
            temperature = self.config["llm"].get("temperature", 0.7)
            top_p = self.config["llm"].get("top_p", 1.0)
            max_tokens = self.config["llm"].get("max_tokens", 4000)

        # Use litellm
        # NOTE: EvolveAdapter handles Gemini models differently from other adapters:
        # - Other adapters' (e.g. AnyMathsAdapter) examples/documentation use "vertex_ai/gemini-*" (requires Google Cloud setup)
        # - EvolveAdapter matches what users would be more familiar with (OpenAI-compatible endpoint + API key only)
        #   to make porting existing projects to GEPA using the adapter a bit easier.
        import litellm  # type: ignore

        self.litellm = litellm

        def _call_lm(prompt: str) -> str:
            # For Gemini models, litellm automatically routes
            # to Vertex AI authentication. This adapter instead uses OpenAI client directly
            # for OpenAI-compatible Gemini models.
            # This allows users to use their existing config (API key only) without
            # needing to set up Google Cloud credentials.
            if api_key and "gemini" in model_name.lower() and api_base and "openai" in api_base.lower():
                # Use OpenAI client directly (matches OpenEvolve's approach for Gemini + OpenAI-compatible endpoints)
                try:
                    from openai import OpenAI

                    client = OpenAI(
                        api_key=api_key,
                        base_url=api_base,
                    )
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )
                    return response.choices[0].message.content or ""
                except ImportError:
                    # Fall back to litellm if openai package not available
                    logging.warning("openai package not available, falling back to litellm")
                except Exception as e:
                    # If OpenAI client fails, fall back to litellm
                    logging.warning(f"OpenAI client failed: {e}, falling back to litellm")

            # use litellm (for non-Gemini models or when OpenAI client unavailable)
            completion_kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }

            if api_key:
                completion_kwargs["api_key"] = api_key

            if api_base:
                completion_kwargs["base_url"] = api_base

            completion = self.litellm.completion(**completion_kwargs)
            try:
                return completion.choices[0].message.content or ""
            except Exception:
                return str(completion)

        self.reflection_lm = _call_lm

        # Load prompt config for proposal signature
        self.prompt_config = self.config.get("prompt", {})

    def _construct_complete_program(self, candidate_program: str) -> str:
        """
        Construct a complete program by replacing the EVOLVE-BLOCK section.

        This ensures that fixed code (evaluate_function, run_search, etc.) is always
        preserved, even if the LLM only proposes the evolved block.

        Args:
            candidate_program: The program text from the candidate (may be full program
                             or just the evolved block)

        Returns:
            Complete program with evolved block inserted into template
        """
        if self.original_program_template is None:
            return candidate_program

        start_marker = "# EVOLVE-BLOCK-START"
        end_marker = "# EVOLVE-BLOCK-END"

        # Extract the evolved block content from the candidate
        candidate_start = candidate_program.find(start_marker)
        candidate_end = candidate_program.find(end_marker)

        if candidate_start != -1 and candidate_end != -1 and candidate_start < candidate_end:
            # Candidate has markers, extract just the content between them
            block_content_start = candidate_start + len(start_marker)
            block_content = candidate_program[block_content_start:candidate_end].strip()
        else:
            # No markers found, assume the entire candidate is the evolved block content
            block_content = candidate_program.strip()

        # Reconstruct the full program by replacing the EVOLVE-BLOCK in the template
        template_start = self.original_program_template.find(start_marker)
        template_end = self.original_program_template.find(end_marker)

        if template_start != -1 and template_end != -1:
            # Replace the EVOLVE-BLOCK section in the template
            before_block = self.original_program_template[:template_start]
            after_block = self.original_program_template[template_end + len(end_marker) :]
            complete_program = f"{before_block}{start_marker}\n{block_content}\n{end_marker}{after_block}"
        else:
            # Template doesn't have markers
            logging.warning("Original program template missing EVOLVE-BLOCK markers, using candidate as-is")
            complete_program = candidate_program

        return complete_program

    def evaluate(
        self,
        batch: list,
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        # candidate = {'program': '...'} where program may be full program or just evolved block
        # Construct complete program by replacing EVOLVE-BLOCK section
        candidate_program = candidate.get("program", "")
        complete_program = self._construct_complete_program(candidate_program)

        # write the code to a temporary file
        tmp_code_path = self.temp_env_path / "temp_code.py"
        # Delete file if it exists
        if tmp_code_path.exists() and tmp_code_path.is_file():
            tmp_code_path.unlink()
        elif tmp_code_path.exists():
            tmp_code_path.rmdir()

        try:
            with open(tmp_code_path, "w") as f:
                f.write(complete_program)
        except Exception as e:
            # Return error results for all batch items
            error_output = {"metrics": {"error": 0.0}, "artifacts": {"error": f"Failed to write code: {e!s}"}}
            return EvaluationBatch(
                outputs=[error_output] * len(batch),
                scores=[0.0] * len(batch),
                trajectories=[None] * len(batch) if capture_traces else None,
            )

        # Run the evaluate method with the temporary file and batch
        eval_results = self.evaluation_strategy.evaluate(str(tmp_code_path), batch)

        # Ensure we have one result per batch item
        if len(eval_results) != len(batch):
            raise ValueError(f"Evaluation returned {len(eval_results)} results, but batch has {len(batch)} items")

        # Convert list[EvaluationResult] to EvaluationBatch
        all_outputs = []
        all_scores = []
        all_trajectories = [] if capture_traces else None

        for i, eval_result in enumerate(eval_results):
            score = eval_result.metrics.get("combined_score", 0.0)
            output = {
                "metrics": eval_result.metrics,
                "artifacts": eval_result.artifacts,
            }

            all_outputs.append(output)
            all_scores.append(score)

            if capture_traces:
                # Create trajectory with input data and evaluation result
                trajectory = {
                    "data": batch[i] if i < len(batch) else {},  # Input data from batch
                    "evaluation_result": {
                        "metrics": eval_result.metrics,
                        "artifacts": eval_result.artifacts,
                    },
                    "program_path": str(tmp_code_path),
                }
                assert all_trajectories is not None
                all_trajectories.append(trajectory)

        return EvaluationBatch(
            outputs=all_outputs, scores=all_scores, trajectories=all_trajectories if capture_traces else None
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        if "program" not in components_to_update:
            return {}

        reflective_examples = []

        program_code = candidate.get("program", "")

        # Process each output/score pair to create reflective examples
        # Use trajectories if available for more detailed input information
        trajectories = eval_batch.trajectories or [None] * len(eval_batch.outputs)

        for output, score, trajectory in zip(eval_batch.outputs, eval_batch.scores, trajectories, strict=False):
            metrics = output.get("metrics", {})
            artifacts = output.get("artifacts", {})

            combined_score = metrics.get("combined_score", score)
            error = metrics.get("error", 0.0)

            feedback = self._create_feedback(metrics, artifacts, combined_score, error)

            # Extract input data from trajectory if available, otherwise use generic
            if trajectory and "data" in trajectory:
                inputs = {"data": trajectory["data"]}
            else:
                inputs = {"problem": "Program evaluation task"}

            example = {
                "Inputs": inputs,
                "Generated Outputs": {"program": program_code, "metrics": metrics, "artifacts": artifacts},
                "Feedback": feedback,
            }

            reflective_examples.append(example)

        if len(reflective_examples) == 0:
            raise Exception("No valid predictions found for any module.")

        return {"program": reflective_examples}

    def _create_feedback(
        self,
        metrics: dict[str, float],
        artifacts: dict[str, Any],
        combined_score: float,
        error: float,
    ) -> str:
        """
        Create feedback string from metrics and artifacts.

        Args:
            metrics: Dictionary of metric names to values
            artifacts: Dictionary of artifact information
            combined_score: The combined score for this evaluation
            error: Error metric (0.0 if no error, >0.0 if error occurred)

        Returns:
            Feedback string for the LLM
        """
        feedback_parts = []

        if error > 0.0 or combined_score == 0.0:
            error_msg = artifacts.get("error", "Unknown error occurred")
            feedback_parts.append(f"The program evaluation failed (score: {combined_score:.3f}).")
            feedback_parts.append(f"Error: {error_msg}.")

            if "traceback" in artifacts:
                feedback_parts.append(f"Traceback: {artifacts['traceback']}")

            feedback_parts.append("The program needs improvement to handle this evaluation correctly.")
        else:
            feedback_parts.append(f"The program achieved a combined score of {combined_score:.3f}.")

            metric_info = []
            for key, value in metrics.items():
                if key not in ("combined_score", "error") and isinstance(value, (int, float)):
                    metric_info.append(f"{key}: {value:.3f}")

            if metric_info:
                feedback_parts.append(f"Metrics: {', '.join(metric_info)}.")

            # Add artifact information if available
            if artifacts:
                artifact_info = []
                for key, value in artifacts.items():
                    if key not in ("error", "traceback") and value is not None:
                        # Convert key from snake_case to readable format
                        readable_key = key.replace("_", " ").title()

                        # Format the value appropriately
                        if isinstance(value, (str, int, float)):
                            artifact_info.append(f"{readable_key}: {value}")
                        elif isinstance(value, bytes):
                            try:
                                artifact_info.append(f"{readable_key}: {value.decode('utf-8')}")
                            except UnicodeDecodeError:
                                artifact_info.append(f"{readable_key}: <binary data>")
                        else:
                            artifact_info.append(f"{readable_key}: {value!s}")

                if artifact_info:
                    feedback_parts.append(" ".join(artifact_info) + ".")

        return " ".join(feedback_parts)

    def propose_new_texts(  # type: ignore[override]
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        new_texts: dict[str, str] = {}

        # Get the appropriate signature class (custom if config provided, else default)
        SignatureClass = EvolveProgramProposalSignature.from_config(self.prompt_config)

        for name in components_to_update:
            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset.get(name, [])

            result = SignatureClass.run(
                lm=self.reflection_lm,
                input_dict={
                    "curr_program": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                },
            )
            new_texts[name] = result.get("new_program", base_instruction)

        return new_texts
