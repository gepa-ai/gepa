# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
GEPA Optimize-Anything API

A simple, flexible interface for evolving any system with string components.
Just provide:
1. Your seed strings (prompts, code, configs, etc.)
2. Your training workloads
3. An evaluate function that scores candidates
4. (Optional) Custom reflection prompts

GEPA handles the rest.
"""

from typing import Any, Callable, TypedDict

from gepa.core.adapter import EvaluationBatch, GEPAAdapter


class WorkloadResult(TypedDict):
    """Result from evaluating a single workload instance.

    - score: Numeric score for this workload (higher is better)
    - context_and_feedback: Arbitrary context dict containing:
        - inputs: The inputs to the system for this workload
        - outputs: What the system generated
        - feedback: Human-readable feedback on performance
        - Any other relevant execution traces, context, or metadata
    """

    score: float
    context_and_feedback: dict[str, Any]


# Type for the user's evaluate function
EvaluateFn = Callable[[dict[str, str], list[Any]], list[WorkloadResult]]


class _OptimizeAdapter(GEPAAdapter[Any, Any, Any]):
    """Internal adapter that bridges the simple optimize API to GEPAAdapter."""

    evaluate_fn: EvaluateFn
    reflection_prompt: str | dict[str, str] | None
    failure_score: float

    def __init__(
        self,
        evaluate_fn: EvaluateFn,
        reflection_prompt: str | dict[str, str] | None = None,
        failure_score: float = 0.0,
    ):
        self.evaluate_fn = evaluate_fn
        self.reflection_prompt = reflection_prompt
        self.failure_score = failure_score

    def evaluate(
        self,
        batch: list[Any],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[Any, Any]:
        """Run user's evaluate function and convert to EvaluationBatch."""
        try:
            results = self.evaluate_fn(candidate, batch)

            outputs = [r["context_and_feedback"] for r in results]
            scores = [r["score"] for r in results]
            trajectories = results if capture_traces else None

            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories,
            )
        except Exception as e:
            # On failure, return zero scores but preserve exception info
            return EvaluationBatch(
                outputs=[{"error": str(e)} for _ in batch],
                scores=[self.failure_score for _ in batch],
                trajectories=[{"error": str(e), "workload": w} for w in batch] if capture_traces else None,
            )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[Any, Any],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        """Build reflective dataset from context_and_feedback returned by evaluate."""
        ret_d: dict[str, list[dict[str, Any]]] = {}

        # Each component gets all the context_and_feedback from the batch
        for component_name in components_to_update:
            items = []

            for traj in eval_batch.trajectories or []:
                if "error" in traj:
                    # Skip or include error cases
                    continue

                context = traj.get("context_and_feedback", {})

                # Build a reflective example
                # Users can include any keys they want in context_and_feedback
                # Standard keys we look for: inputs, outputs, feedback
                item = {
                    "Inputs": context.get("inputs", context.get("input", "")),
                    "Generated Outputs": context.get("outputs", context.get("output", "")),
                    "Feedback": context.get("feedback", ""),
                }

                # Include any other context keys the user provided
                for key, val in context.items():
                    if key not in ["inputs", "input", "outputs", "output", "feedback"]:
                        item[key] = val

                items.append(item)

            if len(items) > 0:
                ret_d[component_name] = items

        if len(ret_d) == 0:
            raise Exception("No valid workload results found. Check your evaluate function.")

        return ret_d

    def get_reflection_prompt_template(self) -> str | None:
        """Return the custom reflection prompt template if provided."""
        # For now, we assume a single prompt for all components
        # TODO: Support per-component prompts
        if isinstance(self.reflection_prompt, dict):
            # Use the first prompt or a 'default' key
            return self.reflection_prompt.get("default", next(iter(self.reflection_prompt.values()), None))
        return self.reflection_prompt


def evolve(
    seed_candidate: dict[str, str],
    trainset: list[Any],
    evaluate: EvaluateFn,
    *,
    reflection_prompt: str | dict[str, str] | None = None,
    failure_score: float = 0.0,
    num_iterations: int = 50,
    minibatch_size: int = 25,
    teacher_lm: str = "openai/gpt-4o",
    random_seed: int | None = None,
    output_dir: str | None = None,
    verbose: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Evolve any system with string components using GEPA.

    This is a simple, flexible API for evolving prompts, code, configs, documents,
    or any other text-based system components. Just provide your strings, workloads,
    and an evaluation function - GEPA handles the rest.

    Parameters
    ----------
    seed_candidate : dict[str, str]
        Initial strings to evolve, as a dict mapping component_name -> initial_string.
        Examples:
        - Prompt optimization: {"system_prompt": "You are a helpful assistant..."}
        - Code optimization: {"merge_sort": "def merge_sort(arr): ...", "merge": "def merge(left, right): ..."}
        - Multi-component: {"prompt": "...", "tool_description": "...", "schema": "..."}

    trainset : list[Any]
        List of workload instances. Can be anything your evaluate function understands.
        Examples:
        - Prompt optimization: List of (question, answer) tuples
        - Code optimization: List of test cases or benchmarking configs
        - Agent optimization: List of task traces
        - RAG optimization: List of retrieval queries

    evaluate : Callable[[dict[str, str], list[Any]], list[WorkloadResult]]
        Function that evaluates a candidate on a batch of workloads.

        Takes:
        - candidate: dict[str, str] - The current string values for each component
        - batch: list[Any] - Subset of trainset to evaluate on

        Returns:
        - list[WorkloadResult] - One result per workload in batch, where each result is:
            {
                "score": float,  # Higher is better
                "context_and_feedback": {
                    "inputs": ...,    # What went into the system
                    "outputs": ...,   # What the system produced
                    "feedback": ...,  # Human-readable feedback
                    # ... any other context you want to include for reflection
                }
            }

        The evaluate function has complete flexibility:
        - Execute code, call APIs, run simulations, etc.
        - Include dynamic context (only relevant code/docs/tools for this workload)
        - Capture arbitrary trajectories and metadata
        - Handle failures gracefully (return score=0.0 with error feedback)

    reflection_prompt : str | dict[str, str] | None, optional
        Custom reflection prompt(s) for instruction proposal.
        - If str: Same prompt used for all components
        - If dict: Map component_name -> reflection prompt
        - If None: Use GEPA's default instruction proposal prompt

        The reflection prompt guides how GEPA proposes improvements based on
        the context_and_feedback from your evaluate function.

    failure_score : float, default=0.0
        Score to assign when evaluation fails.

    num_iterations : int, default=50
        Number of optimization iterations to run.

    minibatch_size : int, default=25
        Size of minibatches for candidate acceptance testing.

    minibatch_full_eval_steps : int, default=10
        Evaluate on full trainset every N steps (for tracking progress).

    num_threads : int, default=10
        Number of threads for parallel evaluation.

    teacher_lm : str, default="openai/gpt-4o"
        Language model to use for proposing improved strings.
        Supports any model compatible with LiteLLM.

    random_seed : int | None, optional
        Random seed for reproducibility.

    output_dir : str | None, optional
        Directory to save optimization results and checkpoints.

    verbose : bool, default=True
        Whether to print progress information.

    **kwargs
        Additional arguments passed to GEPA optimizer.

    Returns
    -------
    result : dict
        Optimization results containing:
        - "best_candidate": dict[str, str] - Best evolved strings
        - "best_score": float - Score on full trainset
        - "history": List of iteration results
        - "pareto_frontier": List of non-dominated candidates

    Examples
    --------
    **Example 1: Prompt Optimization**

    ```python
    def evaluate_prompt(candidate, batch):
        results = []
        for question, answer in batch:
            # Run LLM with the prompt
            response = llm(candidate["prompt"], question)
            score = 1.0 if answer in response else 0.0
            results.append({
                "score": score,
                "context_and_feedback": {
                    "inputs": question,
                    "outputs": response,
                    "feedback": f"Correct answer: {answer}"
                }
            })
        return results

    result = evolve(
        seed_candidate={"prompt": "Answer questions accurately."},
        trainset=[(q, a) for q, a in qa_pairs],
        evaluate=evaluate_prompt,
        num_iterations=100,
    )
    ```

    **Example 2: Code Optimization**

    ```python
    def evaluate_sorting(candidate, batch):
        results = []
        for test_case in batch:
            # Compile and run the code
            exec_globals = {}
            exec(candidate["sort_function"], exec_globals)

            try:
                result = exec_globals["sort"](test_case["input"])
                correct = result == test_case["expected"]
                score = 1.0 if correct else 0.0

                results.append({
                    "score": score,
                    "context_and_feedback": {
                        "inputs": f"Array: {test_case['input']}",
                        "outputs": f"Got: {result}",
                        "feedback": f"Expected: {test_case['expected']}"
                    }
                })
            except Exception as e:
                results.append({
                    "score": 0.0,
                    "context_and_feedback": {
                        "inputs": f"Array: {test_case['input']}",
                        "outputs": f"Error: {str(e)}",
                        "feedback": "Code failed to execute correctly."
                    }
                })
        return results

    result = evolve(
        seed_candidate={"sort_function": "def sort(arr): return sorted(arr)"},
        trainset=test_cases,
        evaluate=evaluate_sorting,
        reflection_prompt="You are optimizing a sorting algorithm. Focus on performance and correctness.",
        num_iterations=200,
    )
    ```

    **Example 3: Multi-Component Agent Optimization**

    ```python
    def evaluate_agent(candidate, batch):
        results = []
        for task in batch:
            # Run agent with current prompt and tool descriptions
            agent = Agent(
                system_prompt=candidate["system_prompt"],
                tool_descriptions=candidate["tool_descriptions"]
            )
            trace = agent.run(task)

            # Score based on task completion
            score = compute_task_score(trace, task)

            # Include dynamic context (only tools that were called)
            relevant_tools = [t for t in trace.tool_calls]

            results.append({
                "score": score,
                "context_and_feedback": {
                    "inputs": task,
                    "outputs": trace.final_answer,
                    "feedback": analyze_trace(trace, task),
                    "relevant_tools": relevant_tools,  # Dynamic context!
                    "trace": trace.steps,
                }
            })
        return results

    result = evolve(
        seed_candidate={
            "system_prompt": "You are a helpful agent...",
            "tool_descriptions": "Tools: search, calculate, ..."
        },
        trainset=agent_tasks,
        evaluate=evaluate_agent,
        reflection_prompt={
            "system_prompt": "Improve the agent's system prompt for better task completion.",
            "tool_descriptions": "Improve tool descriptions for clearer usage."
        },
        num_iterations=150,
    )
    ```

    Notes
    -----
    - The evaluate function has complete flexibility to execute any system
    - GEPA works at the granularity of individual workloads, not just aggregate scores
    - Dynamic context allows including only relevant information for each workload
    - Not tied to filesystem - compile and execute code in-memory, call APIs, etc.
    - Supports arbitrary trajectories and metadata in context_and_feedback
    - Can evolve multiple components together with coupled updates
    """
    # Create the internal adapter
    adapter = _OptimizeAdapter(
        evaluate_fn=evaluate,
        reflection_prompt=reflection_prompt,
        failure_score=failure_score,
    )

    # Import the actual GEPA optimize function from api
    from gepa.api import optimize as gepa_optimize

    # Handle stop condition - either max_metric_calls or num_iterations
    max_metric_calls_val = kwargs.get("max_metric_calls")
    if max_metric_calls_val is None and "stop_callbacks" not in kwargs:
        # Approximate: num_iterations * minibatch_size * 2 (for proposal + acceptance)
        max_metric_calls_val = num_iterations * minibatch_size * 2

    # Run GEPA optimization
    result = gepa_optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=None,  # Use trainset for validation too
        adapter=adapter,
        reflection_lm=teacher_lm,
        skip_perfect_score=True,
        reflection_minibatch_size=minibatch_size,
        perfect_score=1.0,
        reflection_prompt_template=adapter.get_reflection_prompt_template(),
        seed=random_seed if random_seed is not None else 0,
        run_dir=output_dir,
        display_progress_bar=verbose,
        max_metric_calls=max_metric_calls_val,
        **kwargs,
    )

    # Format results - GEPAResult has best_candidate and val_aggregate_scores
    best_score = result.val_aggregate_scores[result.best_idx]

    return {
        "best_candidate": result.best_candidate,
        "best_score": best_score,
        "history": result,  # Return full GEPAResult for detailed inspection
        "pareto_frontier": result.per_val_instance_best_candidates,
        "output_dir": output_dir,
    }
