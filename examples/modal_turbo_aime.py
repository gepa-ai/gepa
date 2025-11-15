"""
Distributed TurboGEPA benchmark orchestrated by Modal.

Runs the AIME benchmark configuration across multiple Modal workers,
each sharing the same cache/log/migration/control directories on a
mounted Modal Volume. Workers stop as soon as one of them hits the
target quality threshold, mirroring the local benchmark behaviour.

Usage:
    modal run examples/modal_turbo_aime.py --num-workers 2 --dataset-size 16

Environment:
    Configure the following environment variables locally before running:
        OPENROUTER_API_KEY
        OPENROUTER_API_BASE (optional; defaults to OpenRouter endpoint)
    These are forwarded to Modal via a Secret created from the local env.
"""

from __future__ import annotations

import os
import time
import shutil
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Sequence

try:
    import modal
except ImportError as exc:  # pragma: no cover - example dependency
    raise SystemExit(
        "This example requires the Modal client. Install via `pip install modal-client`."
    ) from exc

import gepa
import json

from turbo_gepa.adapters import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config
from turbo_gepa.distributed.runner import run_worker_from_factory


APP_NAME = "turbo-gepa-aime-modal"
VOLUME_MOUNT = "/mnt/turbogepa"
CACHE_DIR = f"{VOLUME_MOUNT}/cache"
LOG_DIR = f"{VOLUME_MOUNT}/logs"
MIGRATION_DIR = f"{VOLUME_MOUNT}/migrations"
CONTROL_ROOT = f"{VOLUME_MOUNT}/control"
HF_CACHE_ROOT = f"{VOLUME_MOUNT}/hf_cache"
DATASET_SNAPSHOT = "/root/turbo_gepa/data/aime_train.jsonl"
DEFAULT_SEEDS = [
    "You are a meticulous math tutor. Think step by step and end with ### <answer>.",
    "You are an AIME champion. Explain reasoning clearly, then output ### <answer>.",
]
REPO_ROOT = Path(__file__).resolve().parents[1]


def prefetch_aime_dataset(path: str = DATASET_SNAPSHOT) -> None:
    """Download and persist the train split so Modal workers skip HF downloads."""
    trainset, _, _ = gepa.examples.aime.init_dataset()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in trainset:
            handle.write(json.dumps(row))
            handle.write("\n")


MODAL_IMAGE = (
    modal.Image.debian_slim()
    .add_local_dir(REPO_ROOT / "src", remote_path="/root/turbo_gepa/src", copy=True)
    .add_local_dir(REPO_ROOT / "examples", remote_path="/root/turbo_gepa/examples", copy=True)
    .add_local_file(REPO_ROOT / "pyproject.toml", remote_path="/root/turbo_gepa/pyproject.toml", copy=True)
    .add_local_file(REPO_ROOT / "README.md", remote_path="/root/turbo_gepa/README.md", copy=True)
    .add_local_file(REPO_ROOT / "LICENSE", remote_path="/root/turbo_gepa/LICENSE", copy=True)
    .run_commands("cd /root/turbo_gepa && pip install .[full]")
    .run_function(prefetch_aime_dataset, kwargs={"path": DATASET_SNAPSHOT})
    .env({"PYTHONPATH": "/root/turbo_gepa", "AIME_DATASET_PATH": DATASET_SNAPSHOT})
)

volume = modal.Volume.from_name("turbo-gepa-cache", create_if_missing=True)
app = modal.App(APP_NAME, image=MODAL_IMAGE, secrets=[])


def _build_local_secret() -> modal.Secret:
    """Forward selected env vars as a Modal secret when running locally."""
    keys = ["OPENROUTER_API_KEY", "OPENROUTER_API_BASE"]
    payload = {k: v for k, v in os.environ.items() if k in keys and v}
    if not payload:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required in the environment to create a Modal secret."
        )
    return modal.Secret.from_dict(payload)


if modal.is_local():
    OPENROUTER_SECRET = _build_local_secret()
else:  # pragma: no cover - while running on Modal
    # Fall back to a pre-created secret if running in the cloud
    OPENROUTER_SECRET = modal.Secret.from_name("openrouter-api")


def _load_dataset(limit: int) -> list[DefaultDataInst]:
    snapshot_path = Path(os.getenv("AIME_DATASET_PATH", DATASET_SNAPSHOT))
    if snapshot_path.exists():
        rows = [json.loads(line) for line in snapshot_path.read_text().splitlines() if line.strip()]
    else:
        rows, _, _ = gepa.examples.aime.init_dataset()
    trimmed = rows[:limit]
    dataset: list[DefaultDataInst] = []
    for idx, example in enumerate(trimmed):
        dataset.append(
            DefaultDataInst(
                input=example["input"],
                answer=example["answer"],
                additional_context=example.get("additional_context"),
                id=f"aime_{idx}",
            )
        )
    return dataset


def _build_config(dataset_size: int, eval_concurrency: int, target_quality: float) -> Config:
    cfg = Config(
        n_islands=4,
        eval_concurrency=eval_concurrency,
        shards=(0.4, 1.0),
        max_mutations_per_round=max(8, eval_concurrency),
        queue_limit=max(16, eval_concurrency * 2),
        target_quality=target_quality,
        target_shard_fraction=1.0,
        migration_backend="volume",
        migration_path=MIGRATION_DIR,
        cache_path=CACHE_DIR,
        log_path=LOG_DIR,
        control_dir=os.environ.get("TURBOGEPA_CONTROL_PATH"),
    )
    cfg.auto_scale_eval_concurrency = True
    cfg.max_final_shard_inflight = max(2, eval_concurrency // 2)
    cfg.batch_size = min(dataset_size, eval_concurrency)
    cfg.max_optimization_time_seconds = 600
    return cfg


def adapter_factory() -> tuple[DefaultAdapter, Sequence[str]]:
    dataset_size = int(os.environ.get("AIME_DATASET_SIZE", "30"))
    eval_concurrency = int(os.environ.get("AIME_EVAL_CONCURRENCY", "20"))
    target_quality = float(os.environ.get("AIME_TARGET_QUALITY", "0.73"))
    dataset = _load_dataset(dataset_size)
    config = _build_config(dataset_size, eval_concurrency, target_quality)
    adapter = DefaultAdapter(
        dataset=dataset,
        task_lm=os.environ.get("TASK_LM", "openrouter/openai/gpt-oss-20b:nitro"),
        reflection_lm=os.environ.get("REFLECTION_LM", "openrouter/x-ai/grok-4-fast"),
        config=config,
        auto_config=False,
    )
    return adapter, tuple(DEFAULT_SEEDS)


def _prepare_env(run_id: str) -> str:
    cache_dir = Path(CACHE_DIR)
    log_dir = Path(LOG_DIR)
    migration_dir = Path(MIGRATION_DIR)
    control_dir = Path(CONTROL_ROOT) / run_id
    hf_cache = Path(HF_CACHE_ROOT)
    state_dir = Path(VOLUME_MOUNT) / ".turbo_gepa"

    for path in (cache_dir, log_dir, migration_dir, control_dir, hf_cache, state_dir):
        path.mkdir(parents=True, exist_ok=True)

    # Ensure all relative .turbo_gepa/* writes land on the shared volume.
    os.chdir(VOLUME_MOUNT)

    os.environ["TURBOGEPA_CACHE_PATH"] = str(cache_dir)
    os.environ["TURBOGEPA_LOG_PATH"] = str(log_dir)
    os.environ["TURBOGEPA_CONTROL_PATH"] = str(control_dir)
    os.environ["AIME_MIGRATION_PATH"] = str(migration_dir)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")
    return str(control_dir)


@app.function(
    volumes={VOLUME_MOUNT: volume},
    secrets=[OPENROUTER_SECRET],
    timeout=900,
)
def run_worker(
    worker_id: int,
    worker_count: int,
    run_id: str,
    dataset_size: int,
    eval_concurrency: int,
    target_quality: float,
    max_rounds: int | None,
    max_evaluations: int | None,
) -> dict:
    control_dir = _prepare_env(run_id)
    os.environ["AIME_DATASET_SIZE"] = str(dataset_size)
    os.environ["AIME_EVAL_CONCURRENCY"] = str(eval_concurrency)
    os.environ["AIME_TARGET_QUALITY"] = str(target_quality)
    payload = run_worker_from_factory(
        factory="examples.modal_turbo_aime:adapter_factory",
        package="examples",
        worker_id=worker_id,
        worker_count=worker_count,
        islands_per_worker=None,
        seeds=None,
        max_rounds=max_rounds,
        max_evaluations=max_evaluations,
        display_progress=False,
        enable_auto_stop=True,
        control_dir=control_dir,
        run_id=run_id,
    )
    # Persist cache/log/migration updates for other workers/runs.
    try:
        volume.commit()
    except Exception:
        pass
    return payload


@app.function(volumes={VOLUME_MOUNT: volume}, timeout=120)
def clear_volume() -> None:
    """Remove cached artifacts from the shared volume before a new benchmark run."""
    base = Path(VOLUME_MOUNT)
    for child in base.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)  # type: ignore[attr-defined]
    try:
        volume.commit()
    except Exception:
        pass


@dataclass
class Summary:
    worker_id: int
    best_quality: float
    time_to_target: float | None
    prompt: str | None


def _summarize_worker(worker_id: int, payload: dict) -> Summary:
    meta = payload.get("run_metadata")
    if not meta:
        per_island = [item for item in payload.get("run_metadata_per_island", []) if isinstance(item, dict)]
        if per_island:
            meta = max(per_island, key=lambda item: float(item.get("best_quality") or 0.0))
    if not meta:
        meta = {}
    best_quality = float(meta.get("best_quality") or 0.0)
    metrics = meta.get("metrics") or {}
    time_to_target = metrics.get("time_to_target_seconds")
    if time_to_target is None:
        time_to_target = meta.get("time_to_target_seconds")
    prompt = meta.get("best_prompt")
    return Summary(worker_id=worker_id, best_quality=best_quality, time_to_target=time_to_target, prompt=prompt)


def _print_report(run_id: str, summaries: list[Summary]) -> None:
    print(f"\n=== TurboGEPA Modal Run ({run_id}) ===")
    summaries.sort(key=lambda s: s.best_quality, reverse=True)
    best = summaries[0]
    for summary in summaries:
        print(
            f"Worker {summary.worker_id}: quality={summary.best_quality:.3f}, "
            f"time_to_target={summary.time_to_target or 0:.1f}s"
        )
    print("\n--- Best Prompt ---")
    print(best.prompt or "<empty>")


@app.local_entrypoint()
def main(
    num_workers: int = 2,
    dataset_size: int = 16,
    eval_concurrency: int = 12,
    target_quality: float = 0.73,
    max_rounds: int | None = 6,
    max_evaluations: int | None = 200,
) -> None:
    """Launch Modal workers and aggregate the final TurboGEPA results."""
    run_id = f"modal-{int(time.time())}"
    # Reset shared cache/log directories so the run starts from a clean slate.
    clear_volume.remote()
    worker_args = [
        (wid, num_workers, run_id, dataset_size, eval_concurrency, target_quality, max_rounds, max_evaluations)
        for wid in range(num_workers)
    ]
    # Fully realize the Modal generator so we drain all worker futures before summarizing.
    results = list(run_worker.starmap(worker_args))
    summaries = [_summarize_worker(wid, payload) for wid, payload in enumerate(results)]
    _print_report(run_id, summaries)
