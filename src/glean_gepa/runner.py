"""CLI and low-level GEPA engine wiring for Glean prompt optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from datetime import date, timedelta
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
from glean_gepa.debug import set_debug
from glean_gepa.evalcli_client import EvalCliClient
from glean_gepa.evalset_policy import UnseenEvalSetPolicy
from glean_gepa.evolutionary_proposer import EvolutionaryProposer
from glean_gepa.fake_flow import build_fake_flow_components
from glean_gepa.openai_client import create_qe_openai_client, format_exception_chain, get_perfeval_secret
from glean_gepa.prompt import WRITING_CODE_KEY
from glean_gepa.single_model_adapter import SingleModelAdapter
from glean_gepa.teacher_student_adapter import TeacherStudentAdapter

CACHE_DIRECTORY_NAME = "cache"
ADAPTER_CACHE_FILENAME = "glean_adapter_cache.json"
EVAL_RUN_CACHE_FILENAME = "glean_eval_run_cache.json"
CHILDREN_CACHE_FILENAME = "glean_children_cache.json"


def _default_cache_file(run_dir: Path | None, filename: str) -> Path | None:
    """Return a run-local cache path, moving a legacy root-level cache if needed."""
    if run_dir is None:
        return None

    cache_file = run_dir / CACHE_DIRECTORY_NAME / filename
    legacy_file = run_dir / filename
    if legacy_file.exists() and not cache_file.exists():
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        legacy_file.replace(cache_file)
    return cache_file


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


def _parse_reflection_samples(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        sample_count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("reflection_samples must be a positive integer or 'all'") from exc
    if sample_count <= 0:
        raise argparse.ArgumentTypeError("reflection_samples must be a positive integer or 'all'")
    return sample_count


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parse_eval_versions(value: str, *, argument_name: str) -> list[str]:
    versions = [version.strip() for version in value.split(",") if version.strip()]
    if not versions:
        raise SystemExit(f"{argument_name} must contain at least one eval version")
    return versions


def _make_evalset(versions: list[str]) -> list[ALDataInst]:
    return [
        {
            "eval_set_name": "Glean Chat V2 Medium",
            "eval_set_version": version,
            "deployment_ids": ["scio-prod"],
            "status": "active",
        }
        for version in versions
    ]


def _select_recent_train_and_val_versions(
    version_rows: list[dict[str, object]], *, today: date, lookback_days: int, valset_size: int
) -> tuple[list[str], list[str]]:
    """Reserve the newest one or two versions for validation and schedule older ones for training."""
    earliest = today - timedelta(days=lookback_days)
    recent_versions: set[tuple[date, str]] = set()
    for row in version_rows:
        raw_version = row.get("version") or row.get("evalSetVersion")
        if not isinstance(raw_version, str) or not re.fullmatch(r"\d{8}", raw_version):
            continue
        try:
            version_date = date.fromisoformat(f"{raw_version[:4]}-{raw_version[4:6]}-{raw_version[6:]}")
        except ValueError:
            continue
        if earliest <= version_date <= today:
            recent_versions.add((version_date, raw_version))

    ordered_versions = [version for _version_date, version in sorted(recent_versions)]
    if len(ordered_versions) < 2:
        raise SystemExit(
            f"Need at least two scio-prod eval versions dated {earliest.isoformat()} through {today.isoformat()}; "
            f"found {len(ordered_versions)}."
        )
    actual_valset_size = min(valset_size, len(ordered_versions) - 1)
    return ordered_versions[:-actual_valset_size], ordered_versions[-actual_valset_size:]


def _resolve_eval_version_split(args: argparse.Namespace, evalcli: EvalCliClient) -> tuple[list[str], list[str]]:
    if bool(args.train_eval_versions) != bool(args.val_eval_versions):
        raise SystemExit("Set both --train_eval_versions and --val_eval_versions, or neither for automatic selection.")
    if args.train_eval_versions:
        train_versions = _parse_eval_versions(args.train_eval_versions, argument_name="--train_eval_versions")
        val_versions = _parse_eval_versions(args.val_eval_versions, argument_name="--val_eval_versions")
    else:
        rows = evalcli.list_eval_set_versions(
            eval_set_name="Glean Chat V2 Medium", deployment_ids=["scio-prod"]
        )
        train_versions, val_versions = _select_recent_train_and_val_versions(
            rows,
            today=date.today(),
            lookback_days=args.eval_version_lookback_days,
            valset_size=args.val_eval_version_count,
        )
        print(
            "[Eval set schedule] Auto-selected "
            f"train versions={','.join(train_versions)} and val versions={','.join(val_versions)}"
        )

    overlapping_versions = set(train_versions) & set(val_versions)
    if overlapping_versions:
        overlap = ", ".join(sorted(overlapping_versions))
        raise SystemExit(f"Train and validation eval versions must not overlap: {overlap}")
    if not 1 <= len(val_versions) <= 2:
        raise SystemExit("Validation must contain one or two eval versions.")
    return train_versions, val_versions


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize Glean prompts with GEPA's low-level engine.")
    parser.add_argument("--seed_candidate", type=Path)
    parser.add_argument("--max_metric_calls", type=int, default=10)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--student_model", default="gpt")
    parser.add_argument("--teacher_model", default="gpt")
    parser.add_argument("--reflection_lm_model", default="OPEN_AI:GPT5_LATEST")
    parser.add_argument("--qe_project", default="dev-sandbox-334901")
    parser.add_argument("--qe_instance", default="glean-dev")
    parser.add_argument("--qe_authenticated_email", default="cathy.chen@glean.com")
    parser.add_argument("--global_token_cap", type=int, default=4096)
    parser.add_argument(
        "--reflection_samples",
        type=_parse_reflection_samples,
        default=8,
        help="Number of reflective examples per module, or 'all' for every available example.",
    )
    parser.add_argument(
        "--reflection_hamming_distance_k",
        type=_nonnegative_int,
        default=None,
        help="Drop later examples whose isolated execution errors are within Hamming distance k.",
    )
    parser.add_argument("--evalcli", default=None)
    parser.add_argument(
        "--eval_run_timeout_sec",
        type=_nonnegative_int,
        default=21600,
        help="Maximum time to wait for one Cortex eval run (default: 6 hours).",
    )
    parser.add_argument(
        "--cache_file",
        type=Path,
        default=None,
        help="Persistent adapter-analysis cache. Defaults to <run_dir>/cache/glean_adapter_cache.json.",
    )
    parser.add_argument(
        "--eval_run_cache_file",
        type=Path,
        default=None,
        help="Persistent Cortex eval-run ID cache. Defaults to <run_dir>/cache/glean_eval_run_cache.json.",
    )
    parser.add_argument(
        "--children_cache_file",
        type=Path,
        default=None,
        help="Persistent generated-child cache. Defaults to <run_dir>/cache/glean_children_cache.json.",
    )
    parser.add_argument("--shell_error_lookback_days", type=int, default=7)
    parser.add_argument("--bigquery_project", default=None)
    parser.add_argument(
        "--train_eval_versions",
        help="Optional comma-separated override for training versions; set with --val_eval_versions.",
    )
    parser.add_argument(
        "--val_eval_versions",
        help="Optional comma-separated override for held-out validation versions.",
    )
    parser.add_argument(
        "--eval_version_lookback_days",
        type=_nonnegative_int,
        default=14,
        help="Automatically select scio-prod versions from this many days ago through today.",
    )
    parser.add_argument(
        "--val_eval_version_count",
        type=int,
        choices=[1, 2],
        default=2,
        help="Number of newest eligible versions reserved for validation (default: 2).",
    )
    parser.add_argument(
        "--judging_mode",
        choices=["teacher_student", "single_model"],
        default="single_model",
    )
    parser.add_argument(
        "--fake_flow",
        action="store_true",
        help="Run deterministic fake eval data and prompt iterations without external Glean services.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show eval-set payloads and shell-tool action/error details.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    set_debug(args.debug)

    if args.fake_flow:
        _run_fake_flow(args)
        return

    if args.seed_candidate is None:
        raise SystemExit("--seed_candidate is required unless --fake_flow is set")
    seed_candidate = _load_seed_candidate(args.seed_candidate)
    evalcli = EvalCliClient(binary=args.evalcli)
    train_versions, val_versions = _resolve_eval_version_split(args, evalcli)
    trainset = _make_evalset(train_versions)
    valset = _make_evalset(val_versions)
    judging_mode = cast(JudgingMode, args.judging_mode)
    cache_file = args.cache_file or _default_cache_file(args.run_dir, ADAPTER_CACHE_FILENAME)
    eval_run_cache_file = args.eval_run_cache_file or _default_cache_file(args.run_dir, EVAL_RUN_CACHE_FILENAME)
    children_cache_file = args.children_cache_file or _default_cache_file(args.run_dir, CHILDREN_CACHE_FILENAME)
    adapter_kwargs = {
        "runner": ALRunner(
            evalcli=evalcli,
            cache_file=str(eval_run_cache_file) if eval_run_cache_file else None,
            eval_run_timeout_sec=args.eval_run_timeout_sec,
        ),
        "thresholds": Thresholds(quality_min=0.7, tools_min=0.7, max_student_tokens=100000),
        "student_model": args.student_model,
        "cache_file": str(cache_file) if cache_file else None,
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
    proposer = EvolutionaryProposer(
        logger=logger,
        trainset=trainset,
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
        reflect_k=args.reflection_samples,
        reflection_hamming_distance_k=args.reflection_hamming_distance_k,
        baseline_prompt_hash=hashlib.md5(json.dumps(seed_candidate, sort_keys=True).encode()).hexdigest(),
        evalset_policy=UnseenEvalSetPolicy(),
        children_cache_file=children_cache_file,
    )
    optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        proposer=proposer,
        logger=logger,
        experiment_tracker=tracker,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(args.run_dir) if args.run_dir else None,
        frontier_type=cast(FrontierType, adapter.default_frontier_type),
    )


def _run_fake_flow(args: argparse.Namespace) -> None:
    """Execute the real GEPA lifecycle with in-memory fake evaluations."""
    seed_candidate, trainset, valset, adapter, module_specs = build_fake_flow_components()
    print(
        "[FAKE FLOW] Starting offline Glean GEPA flow with separate train and val sets; no external services will be called."
    )
    logger = StdOutLogger()
    tracker = create_experiment_tracker()
    proposer = EvolutionaryProposer(
        logger=logger,
        trainset=trainset,
        al_adapter=adapter,
        reflection_llm=lambda _prompt: "fake reflection is supplied by FakeFlowAdapter",
        experiment_tracker=tracker,
        model="fake-model",
        module_specs=module_specs,
        global_token_cap=4096,
        reflect_k=3,
        baseline_prompt_hash=hashlib.md5(json.dumps(seed_candidate, sort_keys=True).encode()).hexdigest(),
        evalset_policy=UnseenEvalSetPolicy(),
        children_cache_file=args.children_cache_file
        or _default_cache_file(args.run_dir, CHILDREN_CACHE_FILENAME),
    )
    result = optimize(
        seed_candidate=seed_candidate,
        trainset=trainset,
        valset=valset,
        adapter=adapter,  # type: ignore[arg-type]
        proposer=proposer,
        logger=logger,
        experiment_tracker=tracker,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(args.run_dir) if args.run_dir else None,
        frontier_type="objective",
    )
    print(
        f"[FAKE FLOW] Complete: iterations={result.num_candidates - 1}, "
        f"metric_calls={result.total_evals}, best_score={result.best_score:.2f}"
    )


if __name__ == "__main__":
    main()
