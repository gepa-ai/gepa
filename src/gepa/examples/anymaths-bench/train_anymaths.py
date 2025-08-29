def init_dataset(anymaths_dset_name: str = "openai/gsm8k"):
    import random

    from datasets import load_dataset

    train_split = []
    test_split = []
    match anymaths_dset_name:
        case "openai/gsm8k":
            train_load_dataset = load_dataset(anymaths_dset_name, "main", split="train")
            for item in train_load_dataset:
                answer = item["answer"].split("####")[-1].strip()
                solution = item["answer"].split("####")[0].strip()
                question = item["question"]

                train_split.append({"input": question, "additional_context": {"solution": solution}, "answer": answer})

            random.Random(0).shuffle(train_split)

            test_load_dataset = load_dataset(anymaths_dset_name, "main", split="test")
            for item in test_load_dataset:
                answer = item["answer"].split("####")[-1].strip()
                solution = item["answer"].split("####")[0].strip()
                question = item["question"]

                test_split.append({"input": question, "answer": answer})

        case "MathArena/aime_2025":
            train_load_dataset = load_dataset("AI-MO/aimo-validation-aime", "default", split="train")
            for item in train_load_dataset:
                question = item["problem"]
                solution = item["solution"]
                answer = item["answer"]

                train_split.append({"input": question, "additional_context": {"solution": solution}, "answer": answer})

            random.Random(0).shuffle(train_split)

            test_load_dataset = load_dataset("MathArena/aime_2025", "default", split="train")
            for item in test_load_dataset:
                question = item["problem"]
                answer = item["answer"]

                test_split.append({"input": question, "answer": answer})
        case _:
            raise ValueError(f"Unknown dataset name: {anymaths_dset_name}")

    trainset = train_split[: len(train_split) // 2]
    valset = train_split[len(train_split) // 2 :]
    testset = test_split

    return trainset, valset, testset


if __name__ == "__main__":
    import argparse
    from functools import partial
    from pathlib import Path

    import litellm

    from gepa import optimize
    from gepa.adapters.anymaths_adapter import AnyMathsAdapter

    parser = argparse.ArgumentParser()
    parser.add_argument("--anymaths_dset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--train_size", type=int, default=1, help="The size of the training set to use.")
    parser.add_argument("--val_size", type=int, default=1, help="The size of the validation set to use.")
    parser.add_argument("--test_size", type=int, default=1, help="The size of the test set to use.")
    parser.add_argument("--base_lm", type=str, default="ollama/qwen3:4b")
    parser.add_argument("--use_api_base", action="store_true", help="Use API base URL")
    parser.add_argument("--api_base_url", type=str, default="http://localhost:11434")
    parser.add_argument(
        "--reflection_lm", type=str, default="ollama/qwen3:8b", help="The name of the reflection LM to use."
    )
    parser.add_argument("--use_api_reflection", action="store_true", help="Use API reflection URL")
    parser.add_argument(
        "--api_reflection_url",
        type=str,
        default="http://localhost:11434",
        help="The API base URL for the reflection LM.",
    )
    parser.add_argument(
        "--reflection_minibatch_size", type=int, default=8, help="The size of the minibatch for the reflection LM."
    )
    parser.add_argument("--max_litellm_workers", type=int, default=10)
    parser.add_argument("--budget", type=int, default=500, help="The budget for the optimization process.")
    parser.add_argument(
        "--seed", type=int, default=0, help="The seed for the random number generator for reproducibility."
    )
    args = parser.parse_args()

    INSTRUCTION_PROMPT_PATH = Path(__file__).parent / "prompt-templates/instruction_prompt.txt"

    seed_instruction = INSTRUCTION_PROMPT_PATH.read_text()

    trainset, valset, testset = init_dataset(args.anymaths_dset_name)

    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size

    for size in map(int, [train_size, val_size, test_size]):
        if size <= 0:
            raise ValueError("Train, val, and test sizes must be positive integers.")

    trainset = trainset[:train_size]
    valset = valset[:val_size]
    testset = testset[:test_size]

    print("-" * 100)
    print(f"Using dataset: {args.anymaths_dset_name}")
    print(f"Training set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")
    print(f"Test set size: {len(testset)}")
    print("-" * 100)

    base_lm = args.base_lm

    reflection_lm_name = args.reflection_lm

    _reflection = {"model": reflection_lm_name}

    use_api_base = args.use_api_base
    use_api_reflection = args.use_api_reflection

    if use_api_base:
        api_base = args.api_base_url
    else:
        api_base = None

    if use_api_reflection:
        api_reflection = args.api_reflection_url
        _reflection["base_url"] = api_reflection
    else:
        api_reflection = None

    _reflection_completion = partial(litellm.completion, **_reflection)

    def reflection_lm(prompt: str):
        """Call the reflection language model with the given prompt and return its content string."""
        response = _reflection_completion(messages=[{"role": "user", "content": prompt}])
        return response.choices[0].message.content

    max_litellm_workers = args.max_litellm_workers
    budget = args.budget
    reflection_minibatch_size = args.reflection_minibatch_size
    seed = args.seed

    print(f"Using base LM: {base_lm}")
    print(f"Using reflection LM: {reflection_lm_name}")
    print(f"Using API base URL: {api_base}")
    print(f"Using API reflection URL: {api_reflection}")
    print(f"Reflection minibatch size: {reflection_minibatch_size}")
    print(f"Max LiteLLM workers: {max_litellm_workers}")
    print(f"Budget: {budget}")
    print(f"Seed: {seed}")
    print("-" * 100)

    optimized_results = optimize(
        seed_candidate={"instruction_prompt": seed_instruction},
        trainset=trainset,
        valset=valset,
        adapter=AnyMathsAdapter(model=base_lm, api_base=api_base, max_litellm_workers=max_litellm_workers),
        reflection_lm=reflection_lm,
        reflection_minibatch_size=reflection_minibatch_size,
        perfect_score=1,
        skip_perfect_score=False,
        use_wandb=False,
        max_metric_calls=budget,
        seed=seed,
        display_progress_bar=True,
    )

    print("-" * 100)
    print(f"Best prompt >>> {optimized_results.best_candidate}")
    print("-" * 100)
