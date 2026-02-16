<span id="appendix-e-aime-prompt-optimization"></span>
??? example "AIME Prompt Optimization"

    Generalization mode for prompt optimization. The evaluator runs a math-solving LLM with the candidate prompt and checks whether the answer is correct.

    ```python
    import dspy
    from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig
    from examples.aime_math.dataset import load_math_dataset

    def evaluate(candidate: dict[str, str], example) -> tuple[float, SideInfo]:
        prediction = run_llm(example, candidate["prompt"])
        metric_result = math_metric(example, prediction)
        score = metric_result.score
        feedback = metric_result.feedback

        side_info = {
            "Score": score,
            "Input": example.input,
            "Prompt": candidate["prompt"],
            "Output": prediction.answer,
            "Reasoning": getattr(prediction, "reasoning", ""),
            "ExecutionFeedback": feedback,
        }

        return score, side_info

    trainset, valset, testset = load_math_dataset()

    result = optimize_anything(
        seed_candidate={"prompt": "Solve the math problem carefully. "
                                  "Break down the steps and provide the final answer."},
        evaluator=evaluate,
        dataset=trainset,
        valset=valset,
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=600, parallel=True, max_workers=32, cache_evaluation=True),
        ),
    )
    ```

    <!-- @luke:
    A few things to note:
    - Artifact
    - Dataset = None -> objective suffices
    - Background : your suggestions / constraints / etc.
    - frontier type = objective
    - refiner
    - ASI -->
