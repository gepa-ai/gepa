"""GEPAAdapter implementation for the ``optimize_anything`` API.

This adapter bridges the user-facing ``optimize_anything`` API and GEPA's
internal engine.  It handles:

- **Evaluation**: calls the user's wrapped evaluator (single or parallel)
- **Caching**: optional memory or disk cache for ``(candidate, example)`` pairs
- **Refinement**: when :class:`~gepa.optimize_anything.RefinerConfig` is set,
  iteratively improves candidates via an LLM after each evaluation
- **Best-evals tracking**: maintains top-K evaluations per example for
  warm-starting via :class:`~gepa.optimize_anything.OptimizationState`
- **Reflective dataset**: formats evaluation results for the reflection LLM
"""

import hashlib
import json
import logging
import pickle
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

from gepa.core.adapter import DataInst, EvaluationBatch, GEPAAdapter
from gepa.proposer.reflective_mutation.base import LanguageModel

if TYPE_CHECKING:
    from gepa.optimize_anything import Candidate, OptimizationState, RefinerConfig, SideInfo


REFINER_PROMPT_TEMPLATE = """You are refining a candidate to improve its performance.

## Instructions
{refiner_prompt}

## Current Candidate (JSON)
```json
{candidate_to_improve}
```

## Evaluation History
The following shows all evaluation attempts so far, including scores and feedback:
```json
{evaluation_feedback}
```

## Task
Analyze the evaluation history and propose an improved version of the candidate.
Return ONLY a valid JSON object with the improved parameters (no explanation, no markdown fences).
"""


class OptimizeAnythingAdapter(GEPAAdapter):
    """Adapter connecting the ``optimize_anything`` API to GEPA's engine.

    Created automatically by :func:`~gepa.optimize_anything.optimize_anything` —
    users do not instantiate this directly.
    """

    def __init__(
        self,
        evaluator: Callable[..., tuple[float, Any, dict[str, Any]]],
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | None = None,
        parallel: bool = True,
        max_workers: int | None = None,
        refiner_config: "RefinerConfig | None" = None,
        best_example_evals_k: int = 1,
        objective: str | None = None,
        background: str | None = None,
        cache_mode: str = "memory",  # "off", "memory", "disk"
        cache_dir: str | Path | None = None,
    ):
        self.evaluator = evaluator
        self.parallel = parallel
        self.max_workers = max_workers
        self.refiner_config = refiner_config
        self.best_example_evals_k = best_example_evals_k
        self.objective = objective
        self.background = background
        self.cache_mode = cache_mode
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Track top-K best evaluations per example for warm-start support
        self._best_evals_by_example: dict[str, list[dict]] = {}  # keyed by _example_hash
        self._best_evals_lock = threading.Lock()  # Thread safety for parallel execution

        # Initialize evaluation cache
        self._eval_cache: dict[tuple[str, str], tuple[float, Any, dict]] = {}
        self._eval_cache_lock = threading.Lock()

        # Setup disk cache directory and load existing entries
        self._cache_dir_path: Path | None = None
        if self.cache_mode == "disk" and self.cache_dir:
            self._cache_dir_path = self.cache_dir / "fitness_cache"
            self._cache_dir_path.mkdir(parents=True, exist_ok=True)
            self._load_cache()

        # Refiner uses LiteLLM directly via refiner_config.refiner_lm callable
        # No separate predictor needed - the LM callable is set up in optimize_anything.py

    def _get_best_example_evals(self, example: object) -> list[dict]:
        """Get sorted top-K best evaluations for this example (thread-safe)."""
        key = self._example_hash(example)
        with self._best_evals_lock:
            # Return a copy to avoid mutation issues
            return list(self._best_evals_by_example.get(key, []))

    def _update_best_example_evals(self, example: object, score: float, side_info: "SideInfo") -> None:
        """Add evaluation to example's best evals, maintain sorted top-K (thread-safe)."""
        key = self._example_hash(example)
        with self._best_evals_lock:
            if key not in self._best_evals_by_example:
                self._best_evals_by_example[key] = []

            self._best_evals_by_example[key].append({"score": score, "side_info": side_info})

            # Sort descending by score, keep top K
            self._best_evals_by_example[key] = sorted(
                self._best_evals_by_example[key],
                key=lambda x: x["score"],
                reverse=True,
            )[: self.best_example_evals_k]

    # --- Evaluation caching methods ---

    def _candidate_hash(self, candidate: dict[str, str]) -> str:
        """SHA256 hash of candidate for cache key (first 16 chars)."""
        return hashlib.sha256(json.dumps(sorted(candidate.items())).encode()).hexdigest()[:16]

    def _example_hash(self, example: Any) -> str:
        """Hash example for cache key (first 16 chars)."""
        if example is None:
            return "none"
        try:
            # Try JSON serialization for dicts, lists, primitives
            return hashlib.sha256(json.dumps(example, sort_keys=True, default=str).encode()).hexdigest()[:16]
        except (TypeError, ValueError):
            # Fallback to id for unhashable objects
            return f"id_{id(example)}"

    def _cache_key(self, candidate: dict[str, str], example: Any) -> tuple[str, str]:
        """Build cache key tuple."""
        return (self._candidate_hash(candidate), self._example_hash(example))

    def _cache_filename(self, cache_key: tuple[str, str]) -> str:
        """Build filename from cache key."""
        return f"{cache_key[0]}_{cache_key[1]}.pkl"

    def _load_cache(self) -> None:
        """Load all cache entries from disk into memory."""
        if self._cache_dir_path is None:
            return
        for cache_file in self._cache_dir_path.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    # File stores: {"key": (cand_hash, ex_hash), "result": (score, output, side_info)}
                    self._eval_cache[data["key"]] = data["result"]
            except Exception as exc:
                logger.warning("Failed to load cache file %s: %s", cache_file, exc)

    def _save_cache_entry(self, cache_key: tuple[str, str], result: tuple) -> None:
        """Save single cache entry to disk."""
        if self._cache_dir_path is None:
            return
        filename = self._cache_filename(cache_key)
        filepath = self._cache_dir_path / filename
        with open(filepath, "wb") as f:
            pickle.dump({"key": cache_key, "result": result}, f)

    def _build_opt_state(self, example: Any) -> "OptimizationState":
        """Build an OptimizationState for the given example."""
        from gepa.optimize_anything import OptimizationState

        return OptimizationState(best_example_evals=self._get_best_example_evals(example))

    def _call_evaluator(
        self,
        candidate: "Candidate",
        example: Any,
    ) -> tuple[float, Any, dict]:
        """Call evaluator with optional caching."""
        # No caching
        if self.cache_mode == "off":
            return self.evaluator(
                candidate,
                example=example,
                opt_state=self._build_opt_state(example),
            )

        # Build cache key
        cache_key = self._cache_key(candidate, example)

        # Check cache (thread-safe)
        with self._eval_cache_lock:
            if cache_key in self._eval_cache:
                return self._eval_cache[cache_key]

        # Cache miss - call evaluator
        result = self.evaluator(
            candidate,
            example=example,
            opt_state=self._build_opt_state(example),
        )

        # Store in cache (thread-safe)
        with self._eval_cache_lock:
            self._eval_cache[cache_key] = result
            if self.cache_mode == "disk":
                self._save_cache_entry(cache_key, result)

        return result

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: "Candidate",
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate a candidate on a batch of examples, with optional refinement.

        When ``refiner_config`` is set, each example goes through an
        evaluate→refine→re-evaluate loop before returning the best result.
        Multi-objective scores from ``side_info["scores"]`` are extracted and
        forwarded as ``objective_scores`` in the returned batch.
        """
        # Backward compatibility: if refiner_config is None, use old behavior
        if self.refiner_config is None:
            # Old path: direct evaluation without refinement
            if self.parallel and len(batch) > 1:
                raw_results = self._evaluate_parallel(batch, candidate)
            else:
                raw_results = [self._call_evaluator(candidate, example) for example in batch]
            # Package outputs as (score, candidate, side_info) tuples
            eval_output = []
            for score, _, side_info in raw_results:
                output = (score, candidate, side_info)  # Package as tuple
                eval_output.append((score, output, side_info))
        else:
            # New path: evaluate with refinement
            # eval_output is list of (score, output, side_info)
            # where output = (score, best_candidate, side_info)
            eval_output = self._evaluate_with_refinement(batch, candidate)

        # Update best evals history for each example
        # (skip when refiner is on — refiner path already records evals internally)
        if self.refiner_config is None:
            for example, (score, _, side_info) in zip(batch, eval_output, strict=True):
                self._update_best_example_evals(example, score, side_info)

        scores = [score for score, _, _ in eval_output]
        side_infos: list[SideInfo] = [info for _, _, info in eval_output]
        # outputs = list of (score, best_candidate, side_info) tuples
        outputs = [out for _, out, _ in eval_output]

        objective_scores = []
        for side_info in side_infos:
            objective_score = {}
            if "scores" in side_info:
                objective_score.update(side_info["scores"])

            for param_name in candidate.keys():
                if param_name + "_specific_info" in side_info and "scores" in side_info[param_name + "_specific_info"]:
                    objective_score.update(
                        {f"{param_name}::{k}": v for k, v in side_info[param_name + "_specific_info"]["scores"].items()}
                    )

            objective_scores.append(objective_score)

        return EvaluationBatch(
            outputs=outputs, scores=scores, trajectories=side_infos, objective_scores=objective_scores
        )

    def _evaluate_with_refinement(
        self,
        batch: list[DataInst],
        candidate: "Candidate",
    ) -> list[tuple[float, Any, "SideInfo"]]:
        """Evaluate batch with automatic refinement."""
        if self.parallel and len(batch) > 1:
            return self._evaluate_with_refinement_parallel(batch, candidate)

        results = []
        for example in batch:
            result = self._evaluate_single_with_refinement(candidate, example)
            results.append(result)
        return results

    def _evaluate_single_with_refinement(
        self,
        candidate: "Candidate",
        example: object,
    ) -> tuple[float, Any, "SideInfo"]:
        """Evaluate a single example with refinement."""
        assert self.refiner_config is not None

        # refiner_prompt is always in candidate (auto-injected by optimize_anything)
        refiner_prompt = candidate.get("refiner_prompt", "")

        # 1. Evaluate original candidate
        original_score, original_output, original_side_info = self._call_evaluator(candidate, example)

        # Update best evals with original evaluation
        self._update_best_example_evals(example, original_score, original_side_info)

        # 2. Refine and evaluate
        best_refined_score, best_refined_candidate, best_refined_side_info, all_attempts = self._refine_and_evaluate(
            candidate, example, refiner_prompt, original_score, original_side_info
        )

        # 3. Score = best of original and refined (refiner is a score booster)
        # Track best_candidate (the actual candidate dict that won)
        if best_refined_score > original_score:
            final_score = best_refined_score
            best_candidate = best_refined_candidate  # Refined candidate won
        else:
            final_score = original_score
            best_candidate = candidate  # Original candidate won

        # 4. Side_info = original evaluation (so reflection sees raw candidate quality)
        aggregated_side_info = dict(original_side_info)

        # 5. Add refiner_prompt_specific_info with attempt history
        #    Only include "scores" if the user's side_info has "scores" (for objective frontier)
        refiner_side_info: dict[str, Any] = {"Attempts": all_attempts}

        if "scores" in original_side_info:
            # Only consider attempts that produced real evaluation results (have "side_info")
            # so that failed attempts (JSON parse errors, exceptions) with placeholder score=0.0
            # don't mask real evaluations when original scores are negative
            evaluated_attempts = [a for a in all_attempts if "side_info" in a]
            best_refined_scores = {}
            if evaluated_attempts:
                best_attempt = max(
                    evaluated_attempts,
                    key=lambda x: x.get("score", float("-inf")),
                )
                best_attempt_side_info = best_attempt.get("side_info")
                if isinstance(best_attempt_side_info, dict) and "scores" in best_attempt_side_info:
                    best_refined_scores = best_attempt_side_info["scores"]
            refiner_side_info["scores"] = best_refined_scores

        aggregated_side_info["refiner_prompt_specific_info"] = refiner_side_info

        # Package output as (score, best_candidate, side_info) tuple
        output = (final_score, best_candidate, aggregated_side_info)
        return final_score, output, aggregated_side_info

    def _run_parallel(
        self,
        batch: list[DataInst],
        candidate: "Candidate",
        eval_fn,  # callable(candidate, example) -> result
    ) -> list[tuple[float, Any, "SideInfo"]]:
        """Run evaluation function in parallel across batch."""
        results: list[tuple[int, tuple[float, Any, SideInfo]]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers or len(batch)) as executor:
            future_to_idx = {executor.submit(eval_fn, candidate, example): idx for idx, example in enumerate(batch)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results.append((idx, future.result()))

        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    def _evaluate_parallel(self, batch, candidate):
        """Evaluate batch in parallel (no refinement)."""
        return self._run_parallel(batch, candidate, self._call_evaluator)

    def _evaluate_with_refinement_parallel(self, batch, candidate):
        """Evaluate batch in parallel with refinement."""
        return self._run_parallel(batch, candidate, self._evaluate_single_with_refinement)

    def _refine_and_evaluate(
        self,
        candidate: "Candidate",
        example: object,
        refiner_prompt: str,
        original_score: float,
        original_side_info: "SideInfo",
    ) -> tuple[float, dict[str, str] | None, "SideInfo | None", list[dict]]:
        """
        Refine a candidate using the refiner LLM and evaluate the refined version.
        Refines ALL non-refiner params at once via JSON dict.

        Returns:
            (best_refined_score, best_refined_candidate, best_refined_side_info, all_attempts)
            best_refined_candidate is the candidate dict that achieved best refined score,
            or None if no refinement improved over original.
            best_refined_side_info is None if no refinement improved over original.
        """
        # This method should only be called when refiner_config is not None
        assert self.refiner_config is not None, "refiner_config must be set to use refinement"

        # Build params dict: all non-refiner params
        params_dict = {k: v for k, v in candidate.items() if k != "refiner_prompt"}

        best_score = original_score
        best_candidate: dict[str, str] | None = None
        best_side_info: SideInfo | None = None

        # Initialize attempts with original evaluation (iteration 0)
        all_attempts: list[dict] = [
            {
                "iteration": 0,
                "candidate": params_dict,
                "score": original_score,
                "side_info": original_side_info,
            }
        ]

        # Iterative refinement
        refiner_lm = self.refiner_config.refiner_lm
        assert callable(refiner_lm), (
            "refiner_lm must be a callable LanguageModel, not a string or None. "
            "Ensure optimize_anything() has converted it via make_litellm_lm()."
        )

        current_params = params_dict
        for refinement_iter in range(self.refiner_config.max_refinements):
            # Format ALL attempts so far for the refiner (provides full history)
            current_feedback = self._format_all_attempts_feedback(all_attempts)
            try:
                # Call refiner LLM with formatted prompt
                prompt = REFINER_PROMPT_TEMPLATE.format(
                    refiner_prompt=refiner_prompt,
                    candidate_to_improve=json.dumps(current_params, indent=2),
                    evaluation_feedback=current_feedback,
                )
                raw_output = refiner_lm(prompt).strip()
                # Strip markdown code fences if present
                if raw_output.startswith("```"):
                    # Remove first line (```json or ```) and last line (```)
                    lines = raw_output.split("\n")
                    raw_output = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()

                try:
                    parsed_refined = json.loads(raw_output)
                    if not isinstance(parsed_refined, dict):
                        raise ValueError(f"Expected JSON dict, got {type(parsed_refined).__name__}")
                except (json.JSONDecodeError, ValueError) as parse_err:
                    # JSON parse failed: record error so refiner can learn, continue to next iteration
                    all_attempts.append(
                        {
                            "iteration": refinement_iter + 1,
                            "error": f"JSON parse error: {parse_err}",
                            "raw_output": raw_output[:2000],
                            "score": -1e9,
                        }
                    )
                    continue

                # Reconstruct full candidate: refined params + original refiner_prompt
                refined_candidate_dict = {**parsed_refined, "refiner_prompt": candidate.get("refiner_prompt", "")}
                refined_score, refined_output, refined_eval_side_info = self._call_evaluator(
                    refined_candidate_dict, example
                )

                # Update best evals with this refinement evaluation
                self._update_best_example_evals(example, refined_score, refined_eval_side_info)

                # Track attempt
                all_attempts.append(
                    {
                        "iteration": refinement_iter + 1,
                        "candidate": parsed_refined,
                        "score": refined_score,
                        "side_info": refined_eval_side_info,
                    }
                )

                # Update best if improved
                if refined_score > best_score:
                    best_score = refined_score
                    best_candidate = refined_candidate_dict  # Track the actual candidate dict
                    best_side_info = refined_eval_side_info
                    current_params = parsed_refined
                else:
                    # Stop when no improvement
                    break

            except Exception as e:
                # Refinement failed, record error and stop
                all_attempts.append(
                    {
                        "iteration": refinement_iter + 1,
                        "error": str(e),
                        "score": -1e9,
                    }
                )
                break

        return best_score, best_candidate, best_side_info, all_attempts

    def _format_all_attempts_feedback(self, all_attempts: list[dict]) -> str:
        """Format all attempts (including original) into readable feedback for the refiner LLM.

        This provides the refiner with full history of optimization attempts,
        allowing it to understand the progression and avoid repeating failed approaches.
        """
        return json.dumps(all_attempts, indent=2, default=str)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, Any]]]:
        """Format evaluation results into per-component reflection datasets.

        For each component being updated, produces a list of dicts (one per
        example) combining shared SideInfo fields with any
        ``<component>_specific_info`` data.  The ``"scores"`` key is renamed
        to ``"Scores (Higher is Better)"`` for clarity in the LLM prompt.
        """
        scores, side_infos = eval_batch.scores, eval_batch.trajectories
        assert side_infos is not None
        ret: dict[str, list[dict[str, Any]]] = {}
        for component_name in components_to_update:
            ret[component_name] = []
            for score, side_info in zip(scores, side_infos, strict=False):
                ret[component_name].append({})
                for k, v in side_info.items():
                    if k == "scores":
                        ret[component_name][-1]["Scores (Higher is Better)"] = v
                    elif not k.endswith("_specific_info"):
                        ret[component_name][-1][k] = v
                    elif k == f"{component_name}_specific_info":
                        ret[component_name][-1].update(v)
                    else:
                        continue
        return ret
