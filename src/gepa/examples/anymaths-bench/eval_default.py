from train_anymaths import init_dataset

from gepa.adapters.anymaths_adapter.anymaths_adapter import AnyMathsAdapter

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--anymaths_dset_name", type=str, default="openai/gsm8k")
    parser.add_argument("--model", type=str, default="ollama/qwen3:4b", help="The model to evaluate.")
    parser.add_argument("--use_api_url", action="store_true", help="Whether to use the API URL.")
    parser.add_argument("--api_url", type=str, default="http://localhost:11434", help="The API URL to use.")
    parser.add_argument("--batch_size", type=int, default=8, help="The batch size for evaluation.")
    parser.add_argument(
        "--max_litellm_workers", type=int, default=1, help="The maximum number of LiteLLM workers to use."
    )
    parser.add_argument(
        "--which_prompt",
        type=str,
        default="seed",
        choices=["seed", "optimized"],
        help="The prompt to use for evaluation.",
    )

    args = parser.parse_args()

    dataset = args.anymaths_dset_name

    use_api_url = args.use_api_url
    if not use_api_url:
        api_url = ""
    else:
        api_url = args.api_url

    model = args.model
    max_litellm_workers = args.max_litellm_workers
    adapter = AnyMathsAdapter(model=model, api_base=api_url, max_litellm_workers=max_litellm_workers)

    _, _, testset = init_dataset(dataset)

    if args.which_prompt == "seed":
        INSTRUCTION_PROMPT_PATH = Path(__file__).parent / "prompt-templates/instruction_prompt.txt"
    else:
        INSTRUCTION_PROMPT_PATH = Path(__file__).parent / "prompt-templates/optimal_prompt.txt"

    instruction = INSTRUCTION_PROMPT_PATH.read_text()

    batched_testset = []
    batch_size = args.batch_size

    for i in range(0, len(testset), batch_size):
        batched_testset.append(testset[i : i + batch_size])

    total_score = 0.0

    print("-" * 100)
    print(f"Evaluating model: {model}")
    print(f"Using API URL: {api_url if api_url else 'No API URL'}")
    print(f"Batch size: {batch_size}")
    print(f"Max LiteLLM workers: {max_litellm_workers}")
    print(f"Using prompt: {args.which_prompt}")
    print("-" * 100)

    with tqdm(total=len(testset), desc="Evaluating") as pbar:
        for batch in batched_testset:
            evaluation = adapter.evaluate(batch, {"system": instruction})
            total_score += sum(evaluation.scores)

            pbar.update(len(batch))
            pbar.set_postfix({"Score": f"{total_score} / {len(testset):.4f}"})

    print("-" * 100)
    print(f"Final score >> {total_score} / {len(testset):.4f}")
    print("-" * 100)
