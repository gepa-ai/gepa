from train_anymaths import init_dataset

from gepa.adapters.anymaths_adapter.anymaths_adapter import AnyMathsStructuredOutput

if __name__ == "__main__":
    import argparse
    import ast
    from pathlib import Path

    import litellm
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
            litellm_requests = []

            for item in batch:
                user_content = f"{item['input']}"
                messages = [{"role": "system", "content": instruction}, {"role": "user", "content": user_content}]

                litellm_requests.append(messages)

            try:
                responses = litellm.batch_completion(
                    model=model,
                    messages=litellm_requests,
                    api_base=api_url,
                    max_workers=max_litellm_workers,
                    format=AnyMathsStructuredOutput.model_json_schema(),
                    response_format={
                        "type": "json_object",
                        "response_schema": AnyMathsStructuredOutput.model_json_schema(),
                        "enforce_validation": True,
                    },
                )
            except litellm.exceptions.JSONSchemaValidationError as e:
                raise e

            for response, item in zip(responses, batch, strict=False):
                correct_output_format = True
                try:
                    assistant_response = ast.literal_eval(response.choices[0].message.content.strip())
                    assistant_final_answer = assistant_response["final_answer"]
                    ground_truth = item["answer"]
                    score = 1.0 if ground_truth in assistant_final_answer else 0.0
                    total_score += score
                except Exception:
                    correct_output_format = False
                    continue

            pbar.update(len(batch))
            pbar.set_postfix({"Score": f"{total_score} / {len(testset):.4f}"})

    print("-" * 100)
    print(f"Final score >> {total_score} / {len(testset):.4f}")
    print("-" * 100)
