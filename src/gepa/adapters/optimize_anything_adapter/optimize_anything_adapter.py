from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import pickle
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from gepa.core.adapter import DataInst, EvaluationBatch, GEPAAdapter
from gepa.proposer.reflective_mutation.base import LanguageModel

if TYPE_CHECKING:
    import dspy  # type: ignore[import-not-found]

    from gepa.optimize_anything import RefinerConfig, SideInfo


class OptimizeAnythingAdapter(GEPAAdapter):
    def __init__(
        self,
        fitness_fn: Callable[..., tuple[float, Any, dict[str, Any]]],
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
        self._best_evals_by_example: dict[str, list[dict]] = {}  # keyed by _example_hash
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

        # Refiner predictor will be lazy-initialized when first needed
        self.refiner_predictor: "dspy.Predict | None" = None
        self._refiner_initialized = False

    def _initialize_refiner(self) -> None:
        """Lazy-initialize the refiner predictor when first needed."""
        if self._refiner_initialized:
            return

        try:
            import dspy  # type: ignore[import-not-found]

            # Define refiner signature — input/output are JSON dicts of all non-refiner params
            class RefinerSignature(dspy.Signature):
                """Refine a candidate based on evaluation feedback."""

                refiner_prompt = dspy.InputField(
                    desc="Instructions for how to refine the candidate"
                )
                candidate_to_improve = dspy.InputField(
                    desc="JSON dict of all parameters to improve"
                )
                evaluation_feedback = dspy.InputField(
                    desc="Evaluation results showing performance and errors"
                )
                refined_candidate = dspy.OutputField(
                    desc="JSON dict of all improved parameters"
                )

            self.refiner_predictor = dspy.Predict(RefinerSignature)
            self._refiner_initialized = True
        except ImportError as e:
            raise ImportError(
                "DSPy is required for refiner functionality. "
                "Install dspy-ai: pip install dspy-ai"
            ) from e

    def _get_best_example_evals(self, example: DataInst) -> list[dict]:
        """Get sorted top-K best evaluations for this example (thread-safe)."""
        key = self._example_hash(example)
        with self._best_evals_lock:
            # Return a copy to avoid mutation issues
            return list(self._best_evals_by_example.get(key, []))

    def _update_best_example_evals(
        self, example: DataInst, score: float, side_info: "SideInfo"
    ) -> None:
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
        # (skip when refiner is on — refiner path already records evals internally)
        if self.refiner_config is None:
            for example, (score, _, side_info) in zip(batch, eval_output):
                self._update_best_example_evals(example, score, side_info)

        scores = [score for score, _, _ in eval_output]
        side_infos: list[SideInfo] = [info for _, _, info in eval_output]
        outputs = side_infos

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

        # refiner_prompt is always in candidate (auto-injected by optimize_anything)
        refiner_prompt = candidate.get("refiner_prompt", "")

        # 1. Evaluate original candidate
        original_score, original_output, original_side_info = self._call_fitness_fn(
            candidate, example
        )

        # Update best evals with original evaluation
        self._update_best_example_evals(example, original_score, original_side_info)

        # 2. Refine and evaluate
        best_refined_score, best_refined_output, best_refined_side_info, all_attempts = self._refine_and_evaluate(
            candidate, example, refiner_prompt, original_score, original_side_info
        )

        # 3. Score = best of original and refined (refiner is a score booster)
        if best_refined_score > original_score:
            final_score = best_refined_score
            best_output = best_refined_output
        else:
            final_score = original_score
            best_output = original_output

        # 4. Side_info = original evaluation (so reflection sees raw candidate quality)
        aggregated_side_info = dict(original_side_info)

        # 5. Add refiner_prompt_specific_info with best refined scores + attempt history
        #    "scores" = actual best metric values across all attempts (for objective frontier)
        best_attempt = max(all_attempts, key=lambda x: x.get("score", float("-inf")))
        best_attempt_side_info = best_attempt.get("side_info")
        best_refined_scores = {}
        if isinstance(best_attempt_side_info, dict) and "scores" in best_attempt_side_info:
            best_refined_scores = best_attempt_side_info["scores"]

        aggregated_side_info["refiner_prompt_specific_info"] = {
            "scores": best_refined_scores,
            "Attempts": all_attempts,
        }

        return final_score, best_output, aggregated_side_info

    def _run_parallel(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        eval_fn,  # callable(candidate, example) -> result
    ) -> list[tuple[float, Any, "SideInfo"]]:
        """Run evaluation function in parallel across batch."""
        results: list[tuple[int, tuple[float, Any, SideInfo]]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers or len(batch)) as executor:
            future_to_idx = {
                executor.submit(eval_fn, candidate, example): idx
                for idx, example in enumerate(batch)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results.append((idx, future.result()))

        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    def _evaluate_parallel(self, batch, candidate):
        """Evaluate batch in parallel (no refinement)."""
        return self._run_parallel(batch, candidate, self._call_fitness_fn)

    def _evaluate_with_refinement_parallel(self, batch, candidate):
        """Evaluate batch in parallel with refinement."""
        return self._run_parallel(batch, candidate, self._evaluate_single_with_refinement)

    def _refine_and_evaluate(
        self,
        candidate: dict[str, str],
        example: DataInst,
        refiner_prompt: str,
        original_score: float,
        original_side_info: "SideInfo",
    ) -> tuple[float, Any, "SideInfo | None", list[dict[str, Any]]]:
        """
        Refine a candidate using the refiner LLM and evaluate the refined version.
        Refines ALL non-refiner params at once via JSON dict.

        Returns:
            (best_refined_score, best_refined_output, best_refined_side_info, all_attempts)
            best_refined_side_info is None if no refinement improved over original.
        """
        # This method should only be called when refiner_config is not None
        assert self.refiner_config is not None, "refiner_config must be set to use refinement"

        # Lazy-initialize refiner on first use (imports dspy here)
        self._initialize_refiner()
        assert self.refiner_predictor is not None, "refiner_predictor must be initialized"

        # Import dspy for context manager
        import dspy  # type: ignore[import-not-found]

        # Build params dict: all non-refiner params
        params_dict = {k: v for k, v in candidate.items() if k != "refiner_prompt"}

        best_score = original_score
        best_output = None
        best_side_info: SideInfo | None = None

        # Initialize attempts with original evaluation (iteration 0)
        all_attempts: list[dict] = [{
            "iteration": 0,
            "candidate": params_dict,
            "score": original_score,
            "side_info": original_side_info,
        }]

        # Iterative refinement
        current_params = params_dict
        for refinement_iter in range(self.refiner_config.max_refinements):
            # Format ALL attempts so far for the refiner (provides full history)
            current_feedback = self._format_all_attempts_feedback(all_attempts)
            try:
                # Call refiner LLM with JSON dict of all params
                with dspy.context(lm=self.refiner_config.refiner_lm):
                    refiner_result = self.refiner_predictor(
                        refiner_prompt=refiner_prompt,
                        candidate_to_improve=json.dumps(current_params, indent=2),
                        evaluation_feedback=current_feedback,
                    )

                # Parse refined candidate as JSON dict
                raw_output = refiner_result.refined_candidate.strip()
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
                    all_attempts.append({
                        "iteration": refinement_iter + 1,
                        "error": f"JSON parse error: {parse_err}",
                        "raw_output": raw_output[:2000],
                        "score": 0.0,
                    })
                    print(f"Refinement {refinement_iter + 1}: JSON parse error: {parse_err}", flush=True)
                    continue

                # Print refined proposal
                print(f"\n{'='*60}", flush=True)
                print(f"Refiner iteration {refinement_iter + 1}/{self.refiner_config.max_refinements}", flush=True)
                print(f"{'='*60}", flush=True)
                refined_summary = json.dumps(parsed_refined, indent=2)
                print(refined_summary[:2000] + ("..." if len(refined_summary) > 2000 else ""), flush=True)
                print(f"{'='*60}\n", flush=True)

                # Reconstruct full candidate: refined params + original refiner_prompt
                refined_candidate_dict = {**parsed_refined, "refiner_prompt": candidate.get("refiner_prompt", "")}
                refined_score, refined_output, refined_eval_side_info = self._call_fitness_fn(
                    refined_candidate_dict, example
                )

                # Update best evals with this refinement evaluation
                self._update_best_example_evals(example, refined_score, refined_eval_side_info)

                # Track attempt
                all_attempts.append({
                    "iteration": refinement_iter + 1,
                    "candidate": parsed_refined,
                    "score": refined_score,
                    "side_info": refined_eval_side_info,
                })

                # Update best if improved
                improved = refined_score > best_score
                print(f"Refinement {refinement_iter + 1}: score={refined_score:.4f} (prev best={best_score:.4f}) {'✓ improved' if improved else ''}", flush=True)
                if improved:
                    best_score = refined_score
                    best_output = refined_output
                    best_side_info = refined_eval_side_info
                    current_params = parsed_refined
                else:
                    # Stop when no improvement
                    break

            except Exception as e:
                # Refinement failed, record error and stop
                all_attempts.append({
                    "iteration": refinement_iter + 1,
                    "error": str(e),
                    "score": 0.0,
                })
                break

        return best_score, best_output, best_side_info, all_attempts

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
