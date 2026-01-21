import argparse
import itertools
import json
import os
import shutil
import multiprocessing
from typing import Any, Dict, List

import filelock
import numpy as np
import pandas as pd
import ray
import tqdm

from sky_spot import env as env_lib
from sky_spot import traces
from sky_spot import simulate
from sky_spot import task as task_lib
from sky_spot.strategies import strategy as strategy_lib

NICKNAMES = {
    "restart_overhead_hours": "restart",
    "num_slices": "slice",
}


@ray.remote(num_cpus=1)
def run_one_exp(
    config: Dict[str, Any], search_configs, strategy_name: str, dump_history=False
):
    args = argparse.Namespace(**config)

    # Ensure restart_overhead_hours is a list for strategy creation
    if not isinstance(args.restart_overhead_hours, list):
        args.restart_overhead_hours = [args.restart_overhead_hours]

    # Add missing inter_task_overhead attribute with default value
    if not hasattr(args, "inter_task_overhead"):
        args.inter_task_overhead = [0.0]

    envs = env_lib.TraceEnv.create_env(config["trace_file"], env_start_hours=0)
    strategy = strategy_lib.Strategy.get(strategy_name)(args)

    # Create task object
    task = task_lib.SingleTask({"duration": args.task_duration_hours})

    stats = simulate.simulate(
        envs,
        strategy,
        task,
        args.trace_file,
        args.deadline_hours,
        args.restart_overhead_hours,
        args.env_start_hours,
        args.output_dir,
        vars(args),
        output_filename=args.output_filename,
        silent=True,
        dump_history=dump_history,
    )
    return {
        "strategy": strategy_name,
        "avg_spot_hours": config["avg_spot_hours"],
        "avg_wait_hours": config["avg_wait_hours"],
        # 'costs': stats['costs'],
        "avg_cost": np.mean(stats["costs"]),
        "std_cost": np.std(stats["costs"]),
        **search_configs,
    }


def run_jobs(
    strategy_name,
    spot_gaps,
    wait_gaps,
    search_configs,
    gap_seconds,
    exp_path,
    result_path,
    kwargs,
):
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    exps = []
    job_submission = tqdm.tqdm(desc="Job submission")
    for spot_gap, wait_gap in zip(spot_gaps, wait_gaps):
        # Generate traces
        exp_dir = os.path.abspath(
            os.path.join(
                exp_path,
                "two_exp",
                f"gap={gap_seconds}",
                f"spot={spot_gap:.2f}-wait={wait_gap:.2f}",
            )
        )
        trace_path = os.path.join(exp_dir, "traces")
        # Generate random start traces
        random_start_trace_path = os.path.join(trace_path, "random_start")
        os.makedirs(trace_path, exist_ok=True)
        with filelock.FileLock(os.path.join(trace_path, ".lock")):
            if not os.path.exists(os.path.join(trace_path, "raw")):
                traces.generate.TwoExponentialGenerator(
                    trace_folder=trace_path,
                    output_folder="raw",
                    gap_seconds=gap_seconds,
                    length=24 * 60 * 60 * 30 * 100 // gap_seconds,
                    alive_scale=spot_gap,
                    wait_scale=wait_gap,
                ).generate(1)

                num_random_traces = kwargs.pop("num_random_traces", 300)
                traces.random_start.generate_random_traces(
                    os.path.join(trace_path, "raw", "0.json"),
                    output_dir=random_start_trace_path,
                    num_traces=num_random_traces,
                )

        # Generate experiment config
        config = {
            "avg_spot_hours": spot_gap * gap_seconds / 3600,
            "avg_wait_hours": wait_gap * gap_seconds / 3600,
            "exp_dir": exp_dir,
            "deadline_hours": 52,
            "task_duration_hours": 48,
            "restart_overhead_hours": 0,
            "strategy": "time_sliced",
            "use_avg_gain": False,
            "env": "trace",
            "env_start_hours": 0,
            "trace_file": random_start_trace_path,
        }
        config.update(kwargs)
        df = pd.DataFrame(
            columns=[
                "strategy",
                "avg_spot_hours",
                "avg_wait_hours",
                *search_configs.keys(),
                "avg_cost",
            ]
        )
        if os.path.exists(result_path):
            new_df = pd.read_csv(result_path)
            # Add new columns
            for col in df.columns:
                if col not in new_df.columns:
                    new_df[col] = np.nan
            df = new_df
        iter_prod = itertools.product(*search_configs.values())
        for prod in iter_prod:
            search_config = dict(zip(search_configs.keys(), prod))
            config.update(search_config)
            name = "-".join(
                [
                    f"{NICKNAMES[k]}={v}" if k in NICKNAMES else f"{k}={v:.1f}"
                    for k, v in search_config.items()
                ]
            )
            config["output_dir"] = os.path.join(exp_dir, name)
            config["output_filename"] = f"result.json"
            os.makedirs(config["output_dir"], exist_ok=True)
            if len(df) > 0:
                same_config = df[
                    (np.isclose(df["avg_spot_hours"], config["avg_spot_hours"]))
                    & np.isclose(df["avg_wait_hours"], config["avg_wait_hours"])
                ]
                conditions = [
                    np.isclose(same_config[k], v) for k, v in search_config.items()
                ]
                same_config = same_config[np.logical_and.reduce(conditions)]
                if len(same_config) > 0:
                    # Skip if has been done
                    continue
            job_submission.update(1)
            with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
                json.dump(config, f, indent=2)
            exps.append(run_one_exp.remote(config, search_config, strategy_name))
    job_submission.close()

    pbar = tqdm.tqdm(total=len(exps), desc="Job completion")
    while len(exps) > 0:
        done, exps = ray.wait(exps, num_returns=1)
        pbar.update(1)
        result = ray.get(done[0])
        df = pd.concat([df, pd.DataFrame.from_records([result])])
        df.to_csv(result_path, index=False)
    return df


def run_real_jobs(
    strategy_with_configs: Dict[str, Dict[str, List[Any]]],
    env_path,
    exp_path,
    result_path,
    kwargs,
    dump_history=False,
    num_random_traces=300,
    trace_length=None,
    force_rerun=False,
    force_strategy: str | None = None,
):
    random_start = num_random_traces > 1
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    exps = []
    job_submission = tqdm.tqdm(desc="Job submission")
    # Generate traces
    filename = os.path.basename(env_path)
    exp_dir = os.path.abspath(os.path.join(exp_path, "real", filename))
    trace_path = os.path.join(exp_dir, "traces")
    # Generate random start traces
    if random_start:
        random_start_trace_path = os.path.join(trace_path, "random_start")
    else:
        random_start_trace_path = env_path
    os.makedirs(trace_path, exist_ok=True)
    trace_length_hours = 92
    for strategy_name, search_configs in strategy_with_configs.items():
        trace_length_hours = max(
            max(search_configs.get("deadline_hours", [52])), trace_length_hours
        )
    print("Max ddl hours", trace_length_hours)
    if trace_length is not None:
        trace_length_hours = trace_length
    if random_start:
        with filelock.FileLock(os.path.join(trace_path, ".lock")):
            if not os.path.exists(random_start_trace_path):
                traces.random_start.generate_random_traces(
                    env_path,
                    output_dir=random_start_trace_path,
                    trace_length_hours=trace_length_hours,
                    num_traces=num_random_traces,
                )

    with open(env_path, "r") as f:
        content = json.load(f)
        availability_trace = 1 - np.array(content["data"])
        gap_seconds = content["metadata"]["gap_seconds"]

    # Get the lengths of all the consecutive 1's
    arr = np.concatenate(([0], availability_trace, [0]))
    arr = np.diff(arr)
    starts = np.where(arr == 1)[0]
    ends = np.where(arr == -1)[0]
    lengths = ends - starts
    avg_spot_hours = np.mean(lengths) * gap_seconds / 3600

    # Get the lengths of all the consecutive 0's
    arr = np.concatenate(([0], 1 - availability_trace, [0]))
    arr = np.diff(arr)
    starts = np.where(arr == 1)[0]
    ends = np.where(arr == -1)[0]
    lengths = ends - starts
    avg_wait_hours = np.mean(lengths) * gap_seconds / 3600

    # Generate experiment config
    config = {
        "avg_spot_hours": avg_spot_hours,
        "avg_wait_hours": avg_wait_hours,
        "origin_trace": env_path,
        "exp_dir": exp_dir,
        "deadline_hours": 52,
        "task_duration_hours": 48,
        "restart_overhead_hours": 0,
        "use_avg_gain": False,
        "env": "trace",
        "env_start_hours": 0,
        "trace_file": random_start_trace_path,
    }
    config.update(kwargs)
    search_keys = set()
    for d in strategy_with_configs.values():
        search_keys.update(d.keys())
    df = pd.DataFrame(
        columns=[
            "strategy",
            "env_path",
            "avg_spot_hours",
            "avg_wait_hours",
            *search_keys,
            "avg_cost",
            "costs",
        ]
    )
    if os.path.exists(result_path) and not force_rerun:
        existing_df = pd.read_csv(result_path)
        if force_strategy:
            existing_df = existing_df[existing_df['strategy'] != force_strategy]
        # Add new columns
        for col in df.columns:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
        df = existing_df

    for strategy_name in strategy_with_configs:
        if force_strategy and strategy_name != force_strategy:
            continue
        search_configs = strategy_with_configs[strategy_name]
        iter_prod = itertools.product(*search_configs.values())
        for prod in iter_prod:
            search_config = dict(zip(search_configs.keys(), prod))
            config.update(search_config)
            name = "-".join(
                [strategy_name]
                + [
                    f"{NICKNAMES[k]}={v}" if k in NICKNAMES else f"{k}={v:.1f}"
                    for k, v in search_config.items()
                ]
            )
            config["output_dir"] = os.path.join(exp_dir, name)
            config["output_filename"] = f"result.json"
            os.makedirs(config["output_dir"], exist_ok=True)
            if len(df) > 0 and not force_rerun and not force_strategy:
                same_config = df[df["strategy"] == strategy_name]
                conditions = [
                    np.isclose(same_config[k].astype(float), v)
                    for k, v in search_config.items()
                    if np.isscalar(v)
                ]
                same_config = same_config[np.logical_and.reduce(conditions)]
                if len(same_config) > 0:
                    # Skip if has been done
                    continue
            job_submission.update(1)
            with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
                json.dump(config, f, indent=2)
            if "ilp" in strategy_name:
                exps.append(
                    run_one_exp.options(
                        num_cpus=min(4, multiprocessing.cpu_count())
                    ).remote(config, search_config, strategy_name, dump_history)
                )
            else:
                exps.append(
                    run_one_exp.remote(
                        config, search_config, strategy_name, dump_history
                    )
                )
    job_submission.close()

    pbar = tqdm.tqdm(total=len(exps), desc=f"Job completion ({result_path})")
    while len(exps) > 0:
        done, exps = ray.wait(exps, num_returns=1)
        pbar.update(1)
        try:
            result = ray.get(done[0])
        except Exception as e:
            print("job failed with error", e)
            continue
        df = pd.concat([df, pd.DataFrame.from_records([result])])
        df.to_csv(result_path, index=False)
    return df
