"""GEPAAdapter implementation for the ``optimize_anything`` API.

This adapter bridges the user-facing ``optimize_anything`` API and GEPA's
internal engine.  It handles:

- **Evaluation**: calls the user's wrapped evaluator (single or parallel)
- **Caching**: optional memory or disk cache for ``(candidate, example)`` pairs
- **Refinement**: when :class:`~gepa.gepa_launcher.RefinerConfig` is set,
  iteratively improves candidates via an LLM after each evaluation
- **Best-evals tracking**: maintains top-K evaluations per example for
  warm-starting via :class:`~gepa.gepa_launcher.OptimizationState`
- **Reflective dataset**: formats evaluation results for the reflection LLM
"""

import hashlib
import inspect
import json
import logging
import pickle
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gepa.core.adapter import DataInst, EvaluationBatch, GEPAAdapter
from gepa.proposer.reflective_mutation.base import LanguageModel

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gepa.gepa_launcher import Candidate, OptimizationState, RefinerConfig, SideInfo


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


class BatchEvaluatorWrapper:
    """Wraps a user ``batch_evaluator`` while preserving its single-external-call contract.

    The whole point of ``batch_evaluator`` is that the user receives ALL
    (candidate, example) pairs in ONE call (e.g. to submit a provider batch job
    or fan out over their own cluster), so per-call features of the standard
    evaluator path — ``oa.log()`` capture and stdout/stderr capture — cannot
    apply here: there is no per-evaluation call to scope them to. Diagnostics
    belong in each returned ``side_info``.

    What IS restored for parity with the standard evaluator path:

    - ``str_candidate_mode``: candidates are unwrapped to ``str`` before the
      user function sees them (mirror of ``EvaluatorWrapper``).
    - ``opt_states`` injection: if the function's signature accepts an
      ``opt_states`` keyword (or ``**kwargs``), it receives a list of
      :class:`OptimizationState` aligned with ``pairs`` — the batch analogue of
      the per-call ``opt_state`` kwarg (warm-starting, #160).
    - Exception policy: when ``raise_on_exception=False``, a raised batch call
      is converted to per-pair ``(0.0, {"error": ...})`` results instead of
      sinking the iteration.
    - Result normalization: each per-pair result may be ``float``,
      ``(score, side_info)`` (the mirror of the evaluator protocol), or a
      legacy ``(score, output, side_info)`` 3-tuple; all normalize to
      ``(score, side_info)``.

    Concurrency: grouped evaluations arrive as one call at a time, but when
    ``refiner_config`` is combined with ``parallel=True`` (and no single-pair
    evaluator exists), per-example refinement loops run on a thread pool and
    may call this wrapper concurrently with singleton batches — the wrapped
    function should be thread-safe in that configuration.
    """

    def __init__(
        self,
        batch_fn: Callable[..., Any],
        *,
        str_candidate_key: str | None = None,
        raise_on_exception: bool = True,
        single_instance_mode: bool = False,
    ):
        self._fn = batch_fn
        self._str_candidate_key = str_candidate_key
        self._raise_on_exception = raise_on_exception
        self._single_instance_mode = single_instance_mode
        sig = inspect.signature(batch_fn)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        self._accepts_opt_states = has_var_kw or "opt_states" in sig.parameters

    def __call__(
        self,
        pairs: list[tuple["Candidate", Any]],
        opt_states: "list[OptimizationState] | None" = None,
    ) -> list[tuple[float, dict[str, Any]]]:
        call_pairs: list[tuple[Any, Any]] = list(pairs)
        if self._str_candidate_key is not None:
            call_pairs = [(candidate[self._str_candidate_key], example) for candidate, example in call_pairs]
        if self._single_instance_mode:
            # Mirror of EvaluatorWrapper's single-instance handling: the
            # internal placeholder example must never reach user code — the
            # user-visible example is None in this mode.
            call_pairs = [(candidate, None) for candidate, _example in call_pairs]

        kwargs: dict[str, Any] = {}
        if self._accepts_opt_states and opt_states is not None:
            kwargs["opt_states"] = opt_states

        try:
            raw_results = list(self._fn(call_pairs, **kwargs))
        except Exception as e:
            if self._raise_on_exception:
                raise
            # _gepa_transient_failure marks these as synthetic whole-batch
            # failure results: the adapter will NOT cache them, so one
            # transient infrastructure error cannot poison the (disk) cache
            # for every pair across runs.
            return [(0.0, {"error": str(e), "_gepa_transient_failure": True}) for _ in pairs]

        if len(raw_results) != len(pairs):
            raise ValueError(f"batch_evaluator returned {len(raw_results)} results but expected {len(pairs)}")

        normalized: list[tuple[float, dict[str, Any]]] = []
        for idx, r in enumerate(raw_results):
            if isinstance(r, list | tuple):
                r_seq = list(r)
                if len(r_seq) >= 3:
                    # Legacy (score, output, side_info): the output slot is
                    # ignored — the adapter packages outputs identically to the
                    # sequential path as (score, candidate, side_info).
                    si_raw = r_seq[2]
                elif len(r_seq) == 2:
                    si_raw = r_seq[1]
                else:
                    si_raw = {}
                if si_raw is None:
                    si_raw = {}
                if not isinstance(si_raw, dict):
                    # Mirror the evaluator path's loudness instead of silently
                    # discarding user diagnostics.
                    raise TypeError(
                        f"batch_evaluator result {idx}: side_info must be a dict or None, "
                        f"got {type(si_raw).__name__}"
                    )
                # Defensive copy (mirror of EvaluatorWrapper): the user may
                # retain and mutate their dict; the cache, best-evals history,
                # and objective scores must not be corruptible after return.
                normalized.append((float(r_seq[0]), dict(si_raw)))
            else:
                normalized.append((float(r), {}))
        return normalized


class OptimizeAnythingAdapter(GEPAAdapter):
    """Adapter connecting the ``optimize_anything`` API to GEPA's engine.

    Created automatically by :func:`~gepa.gepa_launcher.optimize_anything` —
    users do not instantiate this directly.
    """

    def __init__(
        self,
        evaluator: Callable[..., tuple[float, Any, dict[str, Any]]] | None = None,
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
        batch_evaluator: Callable[..., Any] | None = None,
    ):
        if evaluator is None and batch_evaluator is None:
            raise ValueError("Provide evaluator=, batch_evaluator=, or both.")
        self.evaluator = evaluator
        if batch_evaluator is not None and not isinstance(batch_evaluator, BatchEvaluatorWrapper):
            batch_evaluator = BatchEvaluatorWrapper(batch_evaluator)
        self._batch_evaluator = batch_evaluator
        if batch_evaluator is not None and parallel:
            logger.info(
                "batch_evaluator provided: grouped evaluations route through it, so the "
                "parallel=True thread pool is not used as their transport. If refiner_config "
                "is set, refinement loops still run on the thread pool and may invoke "
                "batch_evaluator concurrently with singleton batches — ensure it is thread-safe."
            )
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
        from gepa.gepa_launcher import OptimizationState

        return OptimizationState(best_example_evals=self._get_best_example_evals(example))

    def _resolve_pair(self, candidate: "Candidate", example: Any) -> tuple[float, Any, dict]:
        """Resolve ONE (candidate, example) pair, whichever evaluation transport exists.

        Uses the single-pair evaluator when provided; otherwise routes a
        singleton batch through ``batch_evaluator``. This is the seam that
        makes the refiner (and any other per-pair resolution) work identically
        with or without a single-pair evaluator — including shared caching,
        since :meth:`_call_evaluator`'s cache wraps this method on both paths.
        """
        if self.evaluator is not None:
            return self.evaluator(
                candidate,
                example=example,
                opt_state=self._build_opt_state(example),
            )
        assert self._batch_evaluator is not None
        ((score, side_info),) = self._batch_evaluator(
            [(candidate, example)], opt_states=[self._build_opt_state(example)]
        )
        # Slot 1 is the "output" placeholder, mirroring EvaluatorWrapper's
        # (score, None, side_info) contract; downstream packaging rebuilds it
        # from the candidate (_assemble_no_refiner_batch / refiner path).
        return score, None, side_info

    def _call_evaluator(
        self,
        candidate: "Candidate",
        example: Any,
    ) -> tuple[float, Any, dict]:
        """Call evaluator with optional caching."""
        # No caching
        if self.cache_mode == "off":
            return self._resolve_pair(candidate, example)

        # Build cache key
        cache_key = self._cache_key(candidate, example)

        # Check cache (thread-safe)
        with self._eval_cache_lock:
            if cache_key in self._eval_cache:
                return self._eval_cache[cache_key]

        # Cache miss - call evaluator
        result = self._resolve_pair(candidate, example)

        # Store in cache (thread-safe); synthetic transient-failure results
        # from a batch_evaluator exception are never cached.
        side_info = result[2] if len(result) > 2 and isinstance(result[2], dict) else {}
        if not side_info.get("_gepa_transient_failure"):
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
        # No-refiner path: evaluate each example (optionally in parallel) and
        # package. Factored into ``_assemble_no_refiner_batch`` so that
        # ``batch_evaluate``'s parallel fallback can reuse the exact same
        # packaging (best-evals update, objective-score extraction, counts).
        if self.refiner_config is None:
            if self._batch_evaluator is not None:
                # batch_evaluator is the preferred transport for any multi-pair
                # evaluation (valset evals, merge evals, seed evals) — one
                # external call for the whole batch, cache-aware.
                return self._batch_evaluate_via_user_fn([(candidate, batch)])[0]
            if self.parallel and len(batch) > 1:
                raw_results = self._evaluate_parallel(batch, candidate)
            else:
                raw_results = [self._call_evaluator(candidate, example) for example in batch]
            return self._assemble_no_refiner_batch(candidate, batch, raw_results)

        # Refiner path: evaluate with refinement. eval_output is a list of
        # (score, output, side_info) where output = (score, best_candidate, side_info).
        eval_output = self._evaluate_with_refinement(batch, candidate)

        scores = [score for score, _, _ in eval_output]
        side_infos: list[SideInfo] = [info for _, _, info in eval_output]
        # outputs = list of (score, best_candidate, side_info) tuples
        outputs = [out for _, out, _ in eval_output]
        objective_scores = [self._extract_objective_scores(candidate, si) for si in side_infos]

        # Count actual evaluator invocations from the attempt history recorded
        # in each side_info (the refiner path may call the evaluator multiple
        # times per example).
        num_metric_calls = 0
        for si in side_infos:
            rp_info = si.get("refiner_prompt_specific_info", {})
            attempts = rp_info.get("Attempts", [])
            num_metric_calls += sum(1 for a in attempts if "side_info" in a)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=side_infos,
            objective_scores=objective_scores,
            num_metric_calls=num_metric_calls,
        )

    def batch_evaluate(
        self,
        items: list[tuple["Candidate", list]],
    ) -> list[EvaluationBatch]:
        """Evaluate multiple (candidate, batch) pairs.

        When a ``batch_evaluator`` was provided at construction time (and no
        refiner is configured), all (candidate, example) pairs are flattened
        into a single call to the user's batch function.

        Falls back to sequential ``evaluate()`` calls for the refiner path or a
        single item. Always captures traces. With ``refiner_config`` set,
        evaluation is per-example by construction (each example runs its own
        evaluate->refine->re-evaluate loop), so grouped calls degrade to
        singleton batches through ``_resolve_pair`` — refinement is never
        silently skipped.
        """
        if self._batch_evaluator is not None and self.refiner_config is None:
            return self._batch_evaluate_via_user_fn(items)
        if self.parallel and self.refiner_config is None and len(items) > 1:
            return self._batch_evaluate_parallel_fallback(items)
        return [self.evaluate(batch, candidate, capture_traces=True) for candidate, batch in items]

    def _batch_evaluate_via_user_fn(
        self,
        items: list[tuple["Candidate", list]],
    ) -> list[EvaluationBatch]:
        """Flatten all (candidate, example) pairs, call batch_evaluator once, repackage.

        The (wrapped) batch_evaluator normalizes every per-pair result to
        ``(score, side_info)``; packaging then goes through
        :meth:`_assemble_no_refiner_batch` so the batch path produces byte-identical
        EvaluationBatches to the sequential path: ``outputs`` are
        ``(score, candidate, side_info)`` tuples, best-evals history is updated
        (feeding ``OptimizationState.best_example_evals``), objective scores are
        extracted, and metric calls are counted the same way.
        """
        assert self._batch_evaluator is not None

        pairs: list[tuple[dict[str, str], Any]] = []
        pair_to_item_idx: list[int] = []
        for item_idx, (candidate, batch) in enumerate(items):
            for example in batch:
                pairs.append((candidate, example))
                pair_to_item_idx.append(item_idx)

        # Cache layer: identical key/value format to _call_evaluator, so
        # entries are shared across the batch path, the sequential path, and
        # the refiner's singleton resolutions. Only misses reach the user's
        # function — still as ONE call.
        results: list[tuple[float, dict[str, Any]] | None] = [None] * len(pairs)
        miss_indices = list(range(len(pairs)))
        if self.cache_mode != "off":
            miss_indices = []
            with self._eval_cache_lock:
                for idx, (cand, example) in enumerate(pairs):
                    cached = self._eval_cache.get(self._cache_key(cand, example))
                    if cached is not None:
                        score, _output, side_info = cached
                        results[idx] = (score, side_info)
                    else:
                        miss_indices.append(idx)

        if miss_indices:
            # With caching enabled, duplicate (candidate, example) pairs within
            # one grouped call resolve to a single user-fn evaluation (the
            # sequential path gets the same effect through per-pair cache hits).
            if self.cache_mode != "off":
                first_by_key: dict[tuple[str, str], int] = {}
                unique_miss: list[int] = []
                alias_of: dict[int, int] = {}
                for idx in miss_indices:
                    key = self._cache_key(*pairs[idx])
                    if key in first_by_key:
                        alias_of[idx] = first_by_key[key]
                    else:
                        first_by_key[key] = idx
                        unique_miss.append(idx)
            else:
                unique_miss = list(miss_indices)
                alias_of = {}

            miss_pairs = [pairs[idx] for idx in unique_miss]
            miss_opt_states = [self._build_opt_state(example) for _, example in miss_pairs]
            miss_results = self._batch_evaluator(miss_pairs, opt_states=miss_opt_states)
            for idx, res in zip(unique_miss, miss_results, strict=True):
                results[idx] = res
                score, side_info = res
                # Transient whole-batch failures (see BatchEvaluatorWrapper)
                # are never cached — they must be re-evaluated next time.
                if self.cache_mode != "off" and not side_info.get("_gepa_transient_failure"):
                    cache_key = self._cache_key(*pairs[idx])
                    with self._eval_cache_lock:
                        self._eval_cache[cache_key] = (score, None, side_info)
                        if self.cache_mode == "disk":
                            self._save_cache_entry(cache_key, (score, None, side_info))
            for idx, src in alias_of.items():
                results[idx] = results[src]

        raw_by_item: dict[int, list[tuple[float, Any, SideInfo]]] = {}
        for res, item_idx in zip(results, pair_to_item_idx, strict=True):
            assert res is not None, "unresolved evaluation pair (cache/miss bookkeeping bug)"
            score, side_info = res
            raw_by_item.setdefault(item_idx, []).append((score, None, side_info))

        return [
            self._assemble_no_refiner_batch(items[item_idx][0], items[item_idx][1], raw_by_item.get(item_idx, []))
            for item_idx in range(len(items))
        ]

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
        original_score, _original_output, original_side_info = self._call_evaluator(candidate, example)

        # Update best evals with original evaluation
        self._update_best_example_evals(example, original_score, original_side_info)

        # 2. Refine and evaluate
        best_refined_score, best_refined_candidate, _best_refined_side_info, all_attempts = self._refine_and_evaluate(
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

    def _run_parallel_pairs(
        self,
        pairs: list[tuple["Candidate", Any]],
    ) -> list[tuple[float, Any, "SideInfo"]]:
        """Evaluate ``(candidate, example)`` pairs concurrently, preserving input order.

        Unlike :meth:`_run_parallel` (one candidate, many examples), this fans
        out over pairs whose candidate may differ, so a single thread pool is
        shared across all candidates in a ``batch_evaluate`` call.
        """
        results: list[tuple[int, tuple[float, Any, SideInfo]]] = []
        with ThreadPoolExecutor(max_workers=self.max_workers or len(pairs)) as executor:
            future_to_idx = {
                executor.submit(self._call_evaluator, candidate, example): idx
                for idx, (candidate, example) in enumerate(pairs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results.append((idx, future.result()))
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]

    def _batch_evaluate_parallel_fallback(
        self,
        items: list[tuple["Candidate", list]],
    ) -> list[EvaluationBatch]:
        """Fan all ``(candidate, example)`` pairs out across one thread pool, regroup per item.

        No-refiner path only. Saturates ``max_workers`` across candidates and
        examples, then repackages each item via :meth:`_assemble_no_refiner_batch`
        so results are identical to per-item ``evaluate()`` calls.
        """
        pairs: list[tuple[Candidate, Any]] = []
        pair_to_item_idx: list[int] = []
        for item_idx, (candidate, batch) in enumerate(items):
            for example in batch:
                pairs.append((candidate, example))
                pair_to_item_idx.append(item_idx)

        raw_by_pair = self._run_parallel_pairs(pairs) if pairs else []

        raw_per_item: list[list[tuple[float, Any, SideInfo]]] = [[] for _ in items]
        for pair_idx, raw in enumerate(raw_by_pair):
            raw_per_item[pair_to_item_idx[pair_idx]].append(raw)

        return [
            self._assemble_no_refiner_batch(candidate, batch, raw_per_item[item_idx])
            for item_idx, (candidate, batch) in enumerate(items)
        ]

    def _extract_objective_scores(self, candidate: "Candidate", side_info: "SideInfo") -> dict[str, float]:
        """Pull per-objective scores out of a side_info dict (top-level + per-param)."""
        objective_score: dict[str, float] = {}
        if "scores" in side_info:
            objective_score.update(side_info["scores"])
        for param_name in candidate.keys():
            specific = side_info.get(param_name + "_specific_info") if isinstance(side_info, dict) else None
            if isinstance(specific, dict) and "scores" in specific:
                objective_score.update({f"{param_name}::{k}": v for k, v in specific["scores"].items()})
        return objective_score

    def _assemble_no_refiner_batch(
        self,
        candidate: "Candidate",
        batch: list[DataInst],
        raw_results: list[tuple[float, Any, "SideInfo"]],
    ) -> EvaluationBatch:
        """Package raw ``(score, _, side_info)`` evaluator results into an EvaluationBatch.

        Shared by :meth:`evaluate` (no-refiner path) and
        :meth:`_batch_evaluate_parallel_fallback` so both produce identical
        output, update best-evals history, and count metric calls the same way.
        """
        # Package outputs as (score, candidate, side_info) tuples.
        eval_output = [(score, (score, candidate, side_info), side_info) for score, _, side_info in raw_results]
        for example, (score, _, side_info) in zip(batch, eval_output, strict=True):
            self._update_best_example_evals(example, score, side_info)

        scores = [score for score, _, _ in eval_output]
        side_infos: list[SideInfo] = [info for _, _, info in eval_output]
        outputs = [out for _, out, _ in eval_output]
        objective_scores = [self._extract_objective_scores(candidate, si) for si in side_infos]

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=side_infos,
            objective_scores=objective_scores,
            num_metric_calls=len(batch),
        )

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
                refined_score, _refined_output, refined_eval_side_info = self._call_evaluator(
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
            for _score, side_info in zip(scores, side_infos, strict=False):
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
