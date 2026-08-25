"""CLI and low-level GEPA engine wiring for Glean prompt optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

from gepa.core.state import FrontierType
from gepa.logging.experiment_tracker import create_experiment_tracker
from gepa.logging.logger import StdOutLogger
from glean_gepa.adapter_types import ALDataInst, JudgingMode
from glean_gepa.al_adapter import (
    MODULES,
    ALRunner,
    Judge,
    ModuleSpec,
    Thresholds,
)
from glean_gepa.api import optimize
from glean_gepa.bigquery_client import BigQueryClient
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.evalset_policy import UnseenEvalSetPolicy
from glean_gepa.evolutionary_proposer import EvolutionaryProposer
from glean_gepa.openai_client import create_qe_openai_client, format_exception_chain, get_perfeval_secret
from glean_gepa.prompt import WRITING_CODE_KEY
from glean_gepa.single_model_adapter import SingleModelAdapter
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter


def _load_seed_candidate(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"seed_candidate file not found: {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not raw:
        raise SystemExit("seed_candidate must be a non-empty JSON object")

    if set(raw) != {WRITING_CODE_KEY}:
        raise SystemExit(f"seed_candidate must contain only {WRITING_CODE_KEY!r}")
    writing_code = raw[WRITING_CODE_KEY]
    if not isinstance(writing_code, str):
        raise SystemExit(f"{WRITING_CODE_KEY} must be a string. Got type={type(writing_code)}")
    return {WRITING_CODE_KEY: writing_code}


def _make_reflection_lm(
    model: str,
    *,
    qe_project: str,
    qe_instance: str,
    authenticated_email: str,
    max_tokens: int = 4096,
) -> Callable[[str], str]:
    client = create_qe_openai_client(qe_instance)
    call_count = 0

    def reflection_lm(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        print(f"QE reflection LLM call {call_count}: requesting model={model}, prompt_chars={len(prompt)}")
        print(f"QE reflection LLM call {call_count} prompt:\n{prompt}")
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_tokens,
                extra_body={
                    "perf_eval_secret": get_perfeval_secret(qe_project),
                    "source_info": {
                        "clientInitiator": "USER",
                        "feature": "INTEGRATION_TEST",
                    },
                    "authenticated_email": authenticated_email,
                },
            )
            response_text = response.output_text.strip()
            if not response_text:
                raise RuntimeError("QE reflection LLM returned an empty response")
            print(f"QE reflection LLM call {call_count}: received response_chars={len(response_text)}")
            print(f"QE reflection LLM call {call_count} response:\n{response_text}")
            return response_text
        except Exception as exc:
            message = format_exception_chain(exc)
            raise RuntimeError(f"QE reflection LLM call failed: {message}") from exc

    return reflection_lm


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Glean prompts with GEPA's low-level engine.")
    parser.add_argument("--seed_candidate", required=True, type=Path)
    parser.add_argument("--max_metric_calls", type=int, default=10)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--student_model", default="gpt")
    parser.add_argument("--teacher_model", default="gpt")
    parser.add_argument("--reflection_lm_model", default="OPEN_AI:GPT5_LATEST")
    parser.add_argument("--qe_project", default="dev-sandbox-334901")
    parser.add_argument("--qe_instance", default="glean-dev")
    parser.add_argument("--qe_authenticated_email", default="cathy.chen@glean.com")
    parser.add_argument("--global_token_cap", type=int, default=4096)
    parser.add_argument("--evalcli", default=None)
    parser.add_argument("--shell_error_lookback_days", type=int, default=7)
    parser.add_argument("--bigquery_project", default=None)
    parser.add_argument("--eval_versions", default="20260806,20260727")
    parser.add_argument(
        "--judging_mode",
        choices=["teacher_student", "single_model"],
        default="single_model",
    )
    args = parser.parse_args()

    seed_candidate = _load_seed_candidate(args.seed_candidate)
    versions = [version.strip() for version in args.eval_versions.split(",") if version.strip()]
    if not versions:
        raise SystemExit("At least one eval version is required")

    evalset: list[ALDataInst] = [
        {
            "eval_set_name": "Glean Chat V2 Medium",
            "eval_set_version": version,
            "deployment_ids": ["scio-prod"],
            "status": "active",
        }
        for version in versions
    ]
    judging_mode = cast(JudgingMode, args.judging_mode)
    evalcli = EvalCliClient(binary=args.evalcli)
    adapter_kwargs = {
        "runner": ALRunner(evalcli=evalcli),
        "thresholds": Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        "student_model": args.student_model,
        "cache_file": "~/eval_cache.json",
    }
    if judging_mode == "teacher_student":
        adapter = TeacherStudentAdapter(**adapter_kwargs, teacher_model=args.teacher_model, judge=Judge(evalcli))
    else:
        adapter = SingleModelAdapter(
            **adapter_kwargs,
            bigquery_client=BigQueryClient(project_id=args.bigquery_project),
            shell_error_lookback_days=args.shell_error_lookback_days,
        )

    logger = StdOutLogger()
    tracker = create_experiment_tracker()
    module_specs = {name: ModuleSpec(name, "free_text", 1024) for name in MODULES}
    evalset_policy = UnseenEvalSetPolicy() if judging_mode == "single_model" else None
    proposer = EvolutionaryProposer(
        logger=logger,
        trainset=evalset,
        al_adapter=adapter,
        reflection_llm=_make_reflection_lm(
            args.reflection_lm_model,
            qe_project=args.qe_project,
            qe_instance=args.qe_instance,
            authenticated_email=args.qe_authenticated_email,
        ),
        experiment_tracker=tracker,
        model=args.student_model,
        module_specs=module_specs,
        global_token_cap=args.global_token_cap,
        baseline_prompt_hash=hashlib.md5(json.dumps(seed_candidate, sort_keys=True).encode()).hexdigest(),
        evalset_policy=evalset_policy,
    )
    optimize(
        seed_candidate=seed_candidate,
        trainset=evalset,
        valset=evalset,
        adapter=adapter,
        proposer=proposer,
        logger=logger,
        experiment_tracker=tracker,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(args.run_dir) if args.run_dir else None,
        frontier_type=cast(FrontierType, adapter.default_frontier_type),
        val_evaluation_policy=evalset_policy,
    )


if __name__ == "__main__":
    main()
