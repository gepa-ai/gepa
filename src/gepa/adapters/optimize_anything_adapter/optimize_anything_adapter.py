from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import pickle
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from gepa.core.adapter import DataInst, EvaluationBatch, GEPAAdapter
from gepa.proposer.reflective_mutation.base import LanguageModel

if TYPE_CHECKING:
    from gepa.optimize_anything import FitnessFn, RefinerConfig, SideInfo


class OptimizeAnythingAdapter(GEPAAdapter):
    def __init__(
        self,
        fitness_fn: "FitnessFn",
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
        self.fitness_fn = fitness_fn
        self.parallel = parallel
        self.max_workers = max_workers
        self.refiner_config = refiner_config
        self.best_example_evals_k = best_example_evals_k
        self.objective = objective
        self.background = background
        self.cache_mode = cache_mode
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Track top-K best evaluations per example for warm-start support
        self._best_evals_by_example: dict[DataInst, list[dict]] = {}
        self._best_evals_lock = threading.Lock()  # Thread safety for parallel execution

        # Initialize fitness evaluation cache
        self._fitness_cache: dict[tuple[str, str], tuple[float, Any, dict]] = {}
        self._fitness_cache_lock = threading.Lock()

        # Setup disk cache directory and load existing entries
        self._cache_dir_path: Path | None = None
        if self.cache_mode == "disk" and self.cache_dir:
            self._cache_dir_path = self.cache_dir / "fitness_cache"
            self._cache_dir_path.mkdir(parents=True, exist_ok=True)
            self._load_cache()

        # Initialize refiner predictor if refiner is configured
        self.refiner_predictor = None
        if self.refiner_config is not None:
            try:
                import dspy

                # Define refiner signature
                class RefinerSignature(dspy.Signature):
                    """Refine a candidate based on evaluation feedback."""

                    refiner_prompt = dspy.InputField(
                        desc="Instructions for how to refine the candidate"
                    )
                    candidate_to_improve = dspy.InputField(
                        desc="The candidate (code/prompt/parameter) to improve"
                    )
                    evaluation_feedback = dspy.InputField(
                        desc="Evaluation results showing performance and errors"
                    )
                    refined_candidate = dspy.OutputField(
                        desc="The improved candidate that addresses the issues"
                    )

                self.refiner_predictor = dspy.Predict(RefinerSignature)
            except ImportError:
                raise ImportError(
                    "DSPy is required for refiner functionality. "
                    "Install dspy-ai: pip install dspy-ai"
                )

    def _get_best_example_evals(self, example: DataInst) -> list[dict]:
        """Get sorted top-K best evaluations for this example (thread-safe)."""
        with self._best_evals_lock:
            # Return a copy to avoid mutation issues
            return list(self._best_evals_by_example.get(example, []))

    def _update_best_example_evals(
        self, example: DataInst, score: float, side_info: "SideInfo"
    ) -> None:
        """Add evaluation to example's best evals, maintain sorted top-K (thread-safe)."""
        with self._best_evals_lock:
            if example not in self._best_evals_by_example:
                self._best_evals_by_example[example] = []

            self._best_evals_by_example[example].append({"score": score, "side_info": side_info})

            # Sort descending by score, keep top K
            self._best_evals_by_example[example] = sorted(
                self._best_evals_by_example[example],
                key=lambda x: x["score"],
                reverse=True,
            )[:self.best_example_evals_k]

    # --- Fitness evaluation caching methods ---

    def _candidate_hash(self, candidate: dict[str, str]) -> str:
        """SHA256 hash of candidate for cache key (first 16 chars)."""
        return hashlib.sha256(
            json.dumps(sorted(candidate.items())).encode()
        ).hexdigest()[:16]

    def _example_hash(self, example: Any) -> str:
        """Hash example for cache key (first 16 chars)."""
        if example is None:
            return "none"
        try:
            # Try JSON serialization for dicts, lists, primitives
            return hashlib.sha256(
                json.dumps(example, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]
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
                    self._fitness_cache[data["key"]] = data["result"]
            except Exception:
                pass  # Skip corrupted files

    def _save_cache_entry(self, cache_key: tuple[str, str], result: tuple) -> None:
        """Save single cache entry to disk."""
        if self._cache_dir_path is None:
            return
        filename = self._cache_filename(cache_key)
        filepath = self._cache_dir_path / filename
        with open(filepath, "wb") as f:
            pickle.dump({"key": cache_key, "result": result}, f)

    def _call_fitness_fn(
        self,
        candidate: dict[str, str],
        example: Any,
    ) -> tuple[float, Any, dict]:
        """Call fitness_fn with optional caching."""
        # No caching
        if self.cache_mode == "off":
            return self.fitness_fn(
                candidate,
                example=example,
                best_example_evals=self._get_best_example_evals(example),
            )

        # Build cache key
        cache_key = self._cache_key(candidate, example)

        # Check cache (thread-safe)
        with self._fitness_cache_lock:
            if cache_key in self._fitness_cache:
                return self._fitness_cache[cache_key]

        # Cache miss - call fitness_fn
        result = self.fitness_fn(
            candidate,
            example=example,
            best_example_evals=self._get_best_example_evals(example),
        )

        # Store in cache (thread-safe)
        with self._fitness_cache_lock:
            self._fitness_cache[cache_key] = result
            if self.cache_mode == "disk":
                self._save_cache_entry(cache_key, result)

        return result

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        # Backward compatibility: if refiner_config is None, use old behavior
        if self.refiner_config is None:
            # Old path: direct evaluation without refinement
            if self.parallel and len(batch) > 1:
                eval_output = self._evaluate_parallel(batch, candidate)
            else:
                eval_output = [
                    self._call_fitness_fn(candidate, example)
                    for example in batch
                ]
        else:
            # New path: evaluate with refinement
            eval_output = self._evaluate_with_refinement(batch, candidate)

        # Update best evals history for each example
        for example, (score, _, side_info) in zip(batch, eval_output):
            self._update_best_example_evals(example, score, side_info)

        scores = [score for score, _, _ in eval_output]
        side_infos: list[SideInfo] = [info for _, _, info in eval_output]
        outputs = [output for _, output, _ in eval_output]

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
        candidate: dict[str, str],
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
        candidate: dict[str, str],
        example: DataInst,
    ) -> tuple[float, Any, "SideInfo"]:
        """Evaluate a single example with refinement."""
        assert self.refiner_config is not None

        # Determine refiner_prompt source (optimize or default mode)
        if "refiner_prompt" in candidate:
            # Optimize mode: use prompt from candidate
            refiner_prompt = candidate["refiner_prompt"]
        else:
            # Default mode: use DEFAULT_REFINER_PROMPT with objective/background
            from gepa.optimize_anything import DEFAULT_REFINER_PROMPT
            refiner_prompt = DEFAULT_REFINER_PROMPT.format(
                objective=self.objective or "Maximize the score",
                background=self.background or "No additional background provided.",
            )

        # Determine target parameter
        target_param = self.refiner_config.refinement_target_param
        if target_param is None:
            if len(candidate) == 1:
                # Single-key candidate: use that key
                target_param = list(candidate.keys())[0]
            else:
                # Multi-key: first param that's not "refiner_prompt"
                target_param = next(
                    (k for k in candidate.keys() if k != "refiner_prompt"),
                    list(candidate.keys())[0]
                )

        # 1. Evaluate original candidate
        original_score, original_output, original_side_info = self._call_fitness_fn(
            candidate, example
        )

        # Update best evals with original evaluation
        self._update_best_example_evals(example, original_score, original_side_info)

        # 2. Refine and evaluate
        refined_score, refined_output, refinement_side_info = self._refine_and_evaluate(
            candidate, example, refiner_prompt, original_score, original_side_info
        )

        # 3. Aggregate: use max score
        if refined_score > original_score:
            final_score = refined_score
            best_output = refined_output
        else:
            final_score = original_score
            best_output = original_output

        # 4. Build aggregated side_info (matching current manual implementation pattern)
        aggregated_side_info = {
            "scores": {
                "best_score_from_original_and_refiner": final_score,
                "original": original_score,
                "refiner_prompt": refined_score,
            },
            f"{target_param}_specific_info": original_side_info,
            "refiner_prompt_specific_info": refinement_side_info,
        }

        return final_score, best_output, aggregated_side_info

    def _evaluate_with_refinement_parallel(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
    ) -> list[tuple[float, Any, "SideInfo"]]:
        """Evaluate batch in parallel with refinement."""
        results: list[tuple[int, tuple[float, Any, SideInfo]]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers or len(batch)) as executor:
            future_to_idx = {
                executor.submit(self._evaluate_single_with_refinement, candidate, example): idx
                for idx, example in enumerate(batch)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results.append((idx, future.result()))

        # Sort by index to maintain order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    def _evaluate_parallel(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
    ) -> list[tuple[float, Any, "SideInfo"]]:
        """Evaluate batch in parallel using ThreadPoolExecutor."""
        results: list[tuple[int, tuple[float, Any, SideInfo]]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers or len(batch)) as executor:
            # Submit all tasks with their index to maintain order
            future_to_idx = {
                executor.submit(
                    self._call_fitness_fn,
                    candidate,
                    example,
                ): idx
                for idx, example in enumerate(batch)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results.append((idx, future.result()))

        # Sort by index to maintain original order
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    def _refine_and_evaluate(
        self,
        candidate: dict[str, str],
        example: DataInst,
        refiner_prompt: str,
        original_score: float,
        original_side_info: "SideInfo",
    ) -> tuple[float, Any, "SideInfo"]:
        """
        Refine a candidate using the refiner LLM and evaluate the refined version.

        Returns:
            (best_refined_score, best_refined_output, refinement_side_info)
        """
        import dspy

        # Determine target parameter to refine
        target_param = self.refiner_config.refinement_target_param
        if target_param is None:
            if len(candidate) == 1:
                # Single-key candidate: use that key
                target_param = list(candidate.keys())[0]
            else:
                # Multi-key: first param that's not "refiner_prompt"
                target_param = next(
                    (k for k in candidate.keys() if k != "refiner_prompt"),
                    list(candidate.keys())[0]
                )

        candidate_text = candidate[target_param]
        best_score = original_score
        best_output = None

        # Initialize attempts with original evaluation (iteration 0)
        all_attempts = [{
            "iteration": 0,
            "candidate": candidate_text,
            "score": original_score,
            "side_info": original_side_info,
        }]

        # Iterative refinement
        current_text = candidate_text
        print(f"\n[Refiner] Starting refinement loop (max_refinements={self.refiner_config.max_refinements}, original_score={original_score:.4f})", flush=True)

        for refinement_iter in range(self.refiner_config.max_refinements):
            # Format ALL attempts so far for the refiner (provides full history)
            current_feedback = self._format_all_attempts_feedback(all_attempts)
            try:
                # Call refiner LLM
                with dspy.context(lm=self.refiner_config.refiner_lm):
                    refiner_result = self.refiner_predictor(
                        refiner_prompt=refiner_prompt,
                        candidate_to_improve=current_text,
                        evaluation_feedback=current_feedback,
                    )

                # Extract refined candidate (handle code blocks)
                refined_text = refiner_result.refined_candidate.strip()
                refined_text = refined_text.replace("```python", "").replace("```", "").strip()

                # Print refined proposal
                print(f"\n{'='*60}", flush=True)
                print(f"Refiner iteration {refinement_iter + 1}/{self.refiner_config.max_refinements}", flush=True)
                print(f"{'='*60}", flush=True)
                print(refined_text[:2000] + ("..." if len(refined_text) > 2000 else ""), flush=True)
                print(f"{'='*60}\n", flush=True)

                # Evaluate refined candidate
                refined_candidate_dict = {**candidate, target_param: refined_text}
                refined_score, refined_output, refined_eval_side_info = self._call_fitness_fn(
                    refined_candidate_dict, example
                )

                # Update best evals with this refinement evaluation
                self._update_best_example_evals(example, refined_score, refined_eval_side_info)

                # Track attempt (iteration is +1 since 0 is original)
                all_attempts.append({
                    "iteration": refinement_iter + 1,
                    "candidate": refined_text,
                    "score": refined_score,
                    "side_info": refined_eval_side_info,
                })

                # Update best if improved
                improved = refined_score > best_score
                print(f"Refinement {refinement_iter + 1}: score={refined_score:.4f} (prev best={best_score:.4f}) {'✓ improved' if improved else ''}", flush=True)
                if improved:
                    best_score = refined_score
                    best_output = refined_output
                    current_text = refined_text
                else:
                    # Stop when no improvement
                    break

            except Exception as e:
                # Refinement failed, record error and stop
                all_attempts.append({
                    "iteration": refinement_iter + 1,
                    "error": str(e),
                    "score": original_score,
                })
                break

        # Build refinement side_info
        # Get scores from the best refinement attempt (or original if no improvement)
        best_attempt = max(all_attempts, key=lambda x: x.get("score", float("-inf")))
        best_attempt_scores = best_attempt.get("side_info", {}).get("scores", {})

        refinement_side_info = {
            "scores": {
                "best_refined_score": best_score,
                "refinement_improvement": best_score - original_score,
                **best_attempt_scores,  # Include all metrics from the best attempt
            },
            "Attempts": all_attempts,  # Includes original (iteration 0) + all refinements
            "Refiner prompt used": refiner_prompt,
        }

        return best_score, best_output, refinement_side_info

    def _format_evaluation_feedback(self, side_info: "SideInfo") -> str:
        """Format side_info into readable feedback for the refiner LLM."""
        return json.dumps(side_info, indent=2, default=str)

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
