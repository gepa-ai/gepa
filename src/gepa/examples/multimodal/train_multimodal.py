from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any
import gepa

from gepa.adapters.multimodal_adapter import MultimodalAdapter
from gepa.examples.multimodal.chartqa.ds_chartqa_helper import init_dataset as load_chartqa


def init_multimodal_dataset(
    multimodal_dset_name: str = "chartqa",
    limit: int | None = None,
    train_limit: int | None = None,
    val_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Return (train, val, test) lists of MultiModalDataInst-style dicts:
      {input, images, answer, additional_context}
    Currently only 'chartqa' supported (wraps helper).
    """
    match multimodal_dset_name.lower():
        case "chartqa":
            train_raw, val_raw, test_raw = load_chartqa(
                limit=limit,
                train_limit=train_limit,
                val_limit=val_limit,
                test_limit=test_limit,
            )

            def convert(rows):
                out = []
                for r in rows:
                    out.append(
                        {
                            "input": r["prompt"],
                            "images": r.get("images", []),
                            "answer": r.get("answer", ""),
                            "additional_context": r.get("additional_context", {}),
                        }
                    )
                return out

            return convert(train_raw), convert(val_raw), convert(test_raw)
        case _: # Future work can introduce new multimodal benchmarks here
            raise ValueError(f"Unsupported multimodal dataset: {multimodal_dset_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multimodal_dset_name", type=str, default="chartqa")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--api_base", type=str)
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--reflection_api_key", type=str, default="EMPTY")
    parser.add_argument("--max_litellm_workers", type=int, default=32)
    parser.add_argument("--budget", type=int, default=150, help="The budget for the optimization process.")
    parser.add_argument("--reflection_lm", type=str)
    parser.add_argument("--reflection_api_base")
    parser.add_argument("--reflection_minibatch_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--prompt", type=str)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--val_limit", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    log = logging.getLogger("train_multimodal")

    # Load system prompt template
    prompt_base = Path(__file__).parent / "prompt_templates"
    candidate_paths = [
        prompt_base / args.multimodal_dset_name / f"{args.prompt}.txt",
        prompt_base / f"{args.prompt}.txt",
    ]
    for p in candidate_paths:
        if p.is_file():
            system_prompt = p.read_text(encoding="utf-8")
            break
    else:
        raise FileNotFoundError(
            "Could not find prompt template. Tried:\n" + "\n".join(str(p) for p in candidate_paths)
        )

    log.info("Loading multimodal dataset...")
    trainset, valset, testset = init_multimodal_dataset(
        multimodal_dset_name=args.multimodal_dset_name,
        limit=args.limit,
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        test_limit=args.test_limit,
    )
    log.info(
        "Dataset ready: train=%d val=%d test=%d",
        len(trainset),
        len(valset),
        len(testset),
    )

    # Adapter (strict structured output assumption like AnyMathsAdapter)
    adapter = MultimodalAdapter(
        model=args.model_name,
        api_base=args.api_base,
        api_key=args.api_key,
        max_litellm_workers=args.max_litellm_workers,
    )

    # Reflection LM wrapper (simple completion interface)
    def reflection_lm(prompt: str) -> str:
        # Re-use same API route; reflection model can be different base URL
        import litellm

        resp = litellm.completion(
            model=args.reflection_lm,
            api_base=args.reflection_api_base,
            api_key=args.reflection_api_key,
            messages=[{"role": "user", "content": prompt}],
        )
        return (resp.choices[0].message.content or "").strip()

    log.info("Starting GEPA optimization...")
    result = gepa.optimize(
        seed_candidate={"system_prompt": system_prompt},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=args.reflection_minibatch_size,
        perfect_score=1,
        skip_perfect_score=False,
        use_wandb=False,
        max_metric_calls=args.budget,
        seed=args.seed,
        display_progress_bar=True,
    )

    print("GEPA Optimized System Prompt:\n", result.best_candidate["system_prompt"])


if __name__ == "__main__":
    main()