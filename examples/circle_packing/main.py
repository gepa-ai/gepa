#!/usr/bin/env python3
"""
Circle Packing Evolution with Two-Component Optimization.

Problem: Pack N circles inside a unit square [0,1]x[0,1] to maximize sum of radii.

This version optimizes TWO components:
1. code: The actual circle packing algorithm
2. refiner_prompt: Instructions for how to refine code per-problem
"""

import dspy
import os
from typing import Any, Optional
import json
import time
import numpy as np


from examples.circle_packing.utils import (
    execute_code,
    SEED_CODE,
)
from examples.circle_packing.llms import (
    CIRCLE_PACKING_BACKGROUND,
    SEED_REFINEMENT_PROMPT,
    RefinerSignature,
)
from gepa.optimize_anything import (
    EngineConfig,
    GEPAConfig,
    ReflectionConfig,
    SideInfo,
    optimize_anything,
)


# Constants
NUM_CIRCLES = 26
LLM_MODEL = "openai/gpt-5.1"
TIMEOUT = 600  # 10 minutes, same as OpenEvolve


class StateTracker:
    """
    Manager for the state of the optimization process.
    Tracks the best solution found so far, the number of metric calls, and the cache.
    """

    def __init__(self, log_dir: str=None, max_metric_calls: int=200):
        self.max_metric_calls = max_metric_calls
        self.metric_calls = 0
        self.cache = {}
        self.best_solution = None
        self.best_score = 0
        self.best_artifact = None
        self.logs = []
        self.log_dir = log_dir

    def _key_to_str(self, key_items: tuple) -> str:
        # Convert numpy arrays to lists, leave other items as-is
        serialized = [
            item.tolist() if isinstance(item, np.ndarray) else item
            for item in key_items
        ]
        return json.dumps(serialized)

    def get(self, key: tuple) -> tuple[float, Any, SideInfo] | None:
        key_str = self._key_to_str(key)
        if key_str in self.cache:
            # cache hit
            return self.cache[key_str]
        return None

    def set(
        self,
        key: tuple,
        value: tuple[float, Any, SideInfo],
        score: Optional[float] = None,
        solution: Optional[Any] = None,
        artifact: Optional[Any] = None,
    ) -> None:
        """
        Key is a dictionary of the inputs to the evaluation function.
            For example, for a code, the key is (code, best_solution)
            For a refinement prompt, the key is (prompt, code, best_solution)
        Value is a tuple of (score, output, side_info), same as the output of the evaluation function for the key.
        Score is the score of the candidate. Provide this if you want to track the best score found so far.
        Solution is the solution found by the candidate. Provide this if you want to track the best solution found so far.
        """
        key_str = self._key_to_str(key)
        self.cache[key_str] = value
        self.metric_calls += 1

        if score is not None and solution is not None:
            self._update_best_solution(solution, score, artifact)

        self.log_state()

        if self.metric_calls >= self.max_metric_calls:
            print("Max metric calls reached!")

    def get_best_solution(self) -> tuple[float, Any]:
        return self.best_score, self.best_solution

    def _update_best_solution(
        self,
        solution,
        score,
        artifact,
    ):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_solution = solution
            self.best_artifact = artifact
            print(f"New best solution found: {score:.4f}")

    def log_state(self) -> None:
        print("Logging state...")
        print(f"Best score: {self.best_score:.4f}")
        print(f"Best solution: {self.best_solution}")
        log = {
            "metric_calls": self.metric_calls,
            "best_score": self.best_score,
            "best_solution": json.dumps(self.best_solution.tolist()) if self.best_solution is not None else None,
        }
        if self.best_artifact is None:
            self.logs.append(log)
            self.save_logs()
            return
        for key, value in self.best_artifact.items():
            if isinstance(value, np.ndarray):
                log[f"best_artifact_{key}"] = json.dumps(value.tolist())
            else:
                log[f"best_artifact_{key}"] = value
        self.logs.append(log)
        self.save_logs()

    def save_logs(self) -> None:
        if self.log_dir is None:
            return
        
        with open(os.path.join(self.log_dir, "state_tracker_logs.json"), "w") as f:
            json.dump(self.logs, f, indent=2)
        print(
            f"State tracker logs saved to: {os.path.join(self.log_dir, 'state_tracker_logs.json')}"
        )


def compute_multiple_metrics(
    global_best_score: float, all_scores: list[float]
) -> dict[str, float]:
    candidate_best_score = max(all_scores)
    alpha_fixed = 0.1
    ema_fixed = all_scores[0]
    for s in all_scores[1:]:
        ema_fixed = alpha_fixed * s + (1 - alpha_fixed) * ema_fixed
    alpha_adaptive = 2.0 / (len(all_scores) + 1)
    ema_adaptive = all_scores[0]
    for s in all_scores[1:]:
        ema_adaptive = alpha_adaptive * s + (1 - alpha_adaptive) * ema_adaptive

    return {
        "max_score": max(all_scores),
        "mean_score": sum(all_scores) / len(all_scores),
        "ema_score_fixed": ema_fixed,
        "ema_score_adaptive": ema_adaptive,
        "score_improvement_from_previous_best": candidate_best_score
        - global_best_score,
    }


def refine_code(
    code: str,
    code_score: float,
    code_side_info: dict,
    refiner_prompt: str,
    refiner_predictor,
    refiner_lm,
    timeout: int,
    state_tracker: StateTracker,
):
    assert refiner_predictor is not None
    assert refiner_lm is not None
    assert refiner_prompt is not None

    # Check cache first
    global_best_score, global_best_solution = state_tracker.get_best_solution()
    # cache_key = (code, refiner_prompt, global_best_solution)
    cache_key = (code, refiner_prompt)
    cache_results = state_tracker.get(cache_key)
    if cache_results is not None:
        return cache_results

    try:
        with dspy.context(lm=refiner_lm):
            refined_result = refiner_predictor(
                refiner_prompt=refiner_prompt,
                code_to_improve=code,
                code_results=code_side_info,
            )

        refined_code = (
            refined_result.refined_code.strip()
            .replace("```python", "")
            .replace("```", "")
        )

        res = execute_code(refined_code, timeout, global_best_solution)
        refined_circles = None

        if res["success"]:
            refined_circles = res["result"]["circles"]
            multiple_metrics = compute_multiple_metrics(
                global_best_score, res["result"]["all_scores"]
            )
            refiner_score = res["result"]["validation_details"]["sum_radii"]
            refiner_improvement_rate = (
                (refiner_score - code_score) / code_score
                if code_score > 0
                else refiner_score
            )

            refined_side_info = {
                "scores": {
                    **multiple_metrics,
                    "refiner_improvement_rate": refiner_improvement_rate,
                },
                "Initial code": code,
                "Refined code": refined_code,
                "Global best circles at the time of evaluation": global_best_solution,
                "Circles": refined_circles,
                "Stdout": res.get("stdout", ""),
            }
        else:
            refiner_score = 0.0
            refined_side_info = {
                "scores": {
                    "sum_radii": 0.0,
                    "max_score": 0.0,
                    "refiner_improvement_rate": 0.0,
                },
                "Initial code": code,
                "Refined code": refined_code,
                "Error": res.get("error", "Unknown error"),
                "Traceback": res.get("traceback", ""),
                "Validation Details": res.get("result", {}).get(
                    "validation_details", {}
                ),
                "Stdout": res.get("stdout", ""),
            }

        print("Refined side info:")
        print(refined_side_info)

        # Set cache
        state_tracker.set(
            cache_key,
            (refiner_score, refined_code, refined_side_info),
            score=refiner_score,
            solution=refined_circles,
            artifact={
                "refined_code": refined_code,
                "refiner_prompt": refiner_prompt,
                "arg_current_best_solution": global_best_solution,
                "validation details": res.get("validation_details"),
            },
        )
        return refiner_score, refined_code, refined_side_info

    except Exception as e:
        raise Exception(f"Refinement failed: {e}")


def create_fitness_function(
    state_tracker=None,
    timeout=TIMEOUT,
    refiner_lm=None,
):
    """
    Create fitness function that evaluates code with optional refinement.
    """
    refiner_predictor = dspy.Predict(RefinerSignature)

    def fitness_fn(
        candidate: dict[str, str], **kwargs
    ) -> list[tuple[float, Any, SideInfo]]:
        """
        Evaluate code candidate on batch of problems with optional refinement.
        """
        code_candidate = candidate["code"]

        # Code candidate evaluation
        global_best_score, global_best_solution = state_tracker.get_best_solution()
        cache_key = code_candidate
        code_candidate_cache = state_tracker.get(cache_key)

        if code_candidate_cache is not None:
            code_score, code_side_info = code_candidate_cache
        else:
            code_result = execute_code(code_candidate, timeout, global_best_solution)
            circles = None

            if code_result["success"]:
                circles = code_result["result"]["circles"]
                all_scores = code_result["result"]["all_scores"]
                code_score = code_result["result"]["validation_details"]["sum_radii"]
                code_side_info = {
                    "scores": compute_multiple_metrics(global_best_score, all_scores),
                    "Code": code_candidate,
                    "Circles": circles,
                    "Global best circles at the time of evaluation": global_best_solution,
                    "Stdout": code_result["stdout"],
                }
            else:
                code_score = 0.0
                code_side_info = {
                    "scores": {"sum_radii": 0.0},
                    "Code": code_candidate,
                    "Error": code_result["error"],
                    "Traceback": code_result.get("traceback", ""),
                    "Stdout": code_result["stdout"],
                    "Validation Details": code_result.get("validation_details"),
                }

            # Cache after computing values
            state_tracker.set(
                cache_key,
                (code_score, code_side_info),
                score=code_score,
                solution=circles,
                artifact={
                    "code": code_candidate,
                    "arg_current_best_solution": global_best_solution,
                    "validation details": code_result.get("validation_details"),
                },
            )

        print("Code candidate side info:")
        print(code_side_info)

        # Refiner prompt evaluation
        # Now that we've got the code's results, we can set a cache key as (prompt, code, best_solution)
        # the refiner will receive the code, the
        print("Refining code...")

        refiner_prompt_candidate = candidate["refiner_prompt"]
        global_best_score, global_best_solution = state_tracker.get_best_solution()

        # Refine code for this problem
        (
            refiner_score,
            refiner_code,
            refiner_side_info,
        ) = refine_code(
            code=code_candidate,
            code_score=code_score,
            code_side_info=code_side_info,
            refiner_prompt=refiner_prompt_candidate,
            refiner_predictor=refiner_predictor,
            refiner_lm=refiner_lm,
            timeout=timeout,
            state_tracker=state_tracker,
        )

        if refiner_score > code_score:
            best_score = refiner_score
            best_code = refiner_code
            best_circles = refiner_side_info.get("Circles", None)
        else:
            best_score = code_score
            best_code = code_candidate
            best_circles = code_side_info.get("Circles", None)

        if best_circles is not None:
            best_circles = best_circles.tolist()

        output = {
            "best_score": best_score,
            "best_code": best_code,
            "best_circles": best_circles,
            "code_candidate": code_candidate,
            "code_score": code_score,
            "refiner_prompt": refiner_prompt_candidate,
            "refiner_code": refiner_code,
            "refiner_score": refiner_score,
        }

        side_info = {
            "scores": {
                "best_score_from_code_and_refiner": max(code_score, refiner_score),
                "initial_code": code_score,
                "refiner_prompt": refiner_score,
            },
            "Input": {
                "Timeout (s)": timeout,
            },
            "code_specific_info": code_side_info,
            "refiner_prompt_specific_info": refiner_side_info,
        }

        return (best_score, output, side_info)

    return fitness_fn


def main():
    # Parse arguments
    max_metric_calls = 200
    log_dir = f"outputs/artifacts/circle_packing/{time.strftime('%y%m%d_%H:%M:%S')}"
    os.makedirs(log_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("Circle Packing Evolution - Two-Component Optimization")
    print("=" * 70)
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Problem size: N={NUM_CIRCLES}")
    print(f"Max metric calls: {max_metric_calls}")
    print(f"Log directory: {log_dir}")
    print("=" * 70 + "\n")

    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    # Create trackers
    state_tracker = StateTracker(
        log_dir=log_dir, max_metric_calls=max_metric_calls
    )

    # Create seed candidate
    seed_candidate = {
        "code": SEED_CODE,
        "refiner_prompt": SEED_REFINEMENT_PROMPT,
    }

    # Create refiner LM
    refiner_lm = dspy.LM(
        LLM_MODEL,
        temperature=1.0,
        max_tokens=32000,
        api_key=api_key,
        cache=True,
    )

    gepa_config = GEPAConfig(
        engine=EngineConfig(
            run_dir=log_dir,
            max_metric_calls=max_metric_calls,
            track_best_outputs=True,
            frontier_type="objective",
            cache_evaluation=True,
        ),
        reflection=ReflectionConfig(
            reflection_minibatch_size=1,
            reflection_lm="openai/gpt-5",
        ),
    )

    # Run GEPA optimization
    print("\n" + "=" * 70)
    print("Running GEPA Two-Component Optimization")
    print("=" * 70 + "\n")

    fitness_fn = create_fitness_function(
        state_tracker=state_tracker,
        timeout=TIMEOUT,
        refiner_lm=refiner_lm,
    )

    result = optimize_anything(
        seed_candidate=seed_candidate,
        fitness_fn=fitness_fn,
        config=gepa_config,
        objective="Optimize circle packing code and refiner prompt to maximize sum of circle radii within a unit square for N=26 circles.",
        background=CIRCLE_PACKING_BACKGROUND,
    )

    # Save results
    print("\n" + "=" * 70)
    print("Optimization Complete!")
    print("=" * 70)
    print("Results: ", result)


if __name__ == "__main__":
    main()
