import argparse
import logging
import os
import sys

import numpy as np
import ray
import pandas as pd

from sky_spot import job_runner

USE_COMPLETE_DATA = True
USE_COMPLETE_DATA = False


def main(args):
    assert args.task is not None or args.ddl is not None, (
        "Must specify either task duration or deadline"
    )

    DDL = args.ddl if args.ddl is not None else "search"
    TASK = args.task if args.task is not None else "search"
    if isinstance(TASK, float) and TASK.is_integer():
        TASK = int(TASK)

    RESTART_OVERHEAD_HOURS = args.overhead

    complete_str = "_complete" if USE_COMPLETE_DATA else ""
    if args.real_preemption:
        assert args.date is not None
        DATE = f"real_preemption/{args.date}/"
        REAL_DATA_PATH = os.path.abspath(
            f"data/real/real_preemption/{args.date}/parsed"
        )
    elif args.date is None:
        DATE = ""
        REAL_DATA_PATH = os.path.abspath(f"data/real/ping_based{complete_str}")
    else:
        DATE = f"{args.date}/"
        REAL_DATA_PATH = os.path.abspath(f"data/real/availability/{DATE}/parsed")
    # REAL_DATA_PATH = os.path.abspath(f'data/real/real_preemption/parsed')
    os.makedirs(args.result_dir, exist_ok=True)
    RESULTS_DIR = os.path.abspath(
        f"{args.result_dir}/real{complete_str}/{DATE}ddl={DDL}+task={TASK}+overhead={RESTART_OVERHEAD_HOURS:.2f}"
    )
    EXP_DIR = os.path.abspath(
        f"{args.exp_dir}/real{complete_str}/{DATE}ddl={DDL}+task={TASK}+overhead={RESTART_OVERHEAD_HOURS:.2f}"
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if args.task is None:
        TASK_DURATIONS = list(range(4, DDL, 4))
    else:
        TASK_DURATIONS = [TASK]

    if args.ddl is None:
        DDL_DURATIONS = list(range(TASK + 4, int(TASK * 2), 4))
        # TIGHTER_DDL_DURATIONS = list(range(TASK + 4, int(TASK * 1.5), 4))
        TIGHTER_DDL_DURATIONS = DDL_DURATIONS
        # DDL_DURATIONS = list(range(int(TASK * 1.6 // 4 * 4), int(TASK * 2), 4))
    else:
        DDL_DURATIONS = [args.ddl]
        TIGHTER_DDL_DURATIONS = [args.ddl]

    STRATEGY_KWARGS = {
        # 'deadline_hours': DDL,
        "restart_overhead_hours": RESTART_OVERHEAD_HOURS,
        "max_total_slacks": None,
        "max_slice_slacks": None,
    }

    SLICES = list(range(1, 10)) + list(range(10, max(DDL_DURATIONS), 4))
    # SLICES = list(range(1, 10))

    STRATEGIES_WITH_SEARCH_CONFIGS = {
        "strawman": {
            "task_duration_hours": TASK_DURATIONS,
            "deadline_hours": DDL_DURATIONS,
        },
        "on_demand": {
            "deadline_hours": DDL_DURATIONS,
            "task_duration_hours": TASK_DURATIONS,
        },
        # 'time_sliced_by_num': {
        #     'task_duration_hours':
        #     TASK_DURATIONS,
        #     'deadline_hours':
        #     DDL_DURATIONS,
        #     'num_slices': SLICES,
        # },
        # 'loose_time_sliced_by_num': {
        #     'task_duration_hours':
        #     TASK_DURATIONS,
        #     'deadline_hours':
        #     DDL_DURATIONS,
        #     'num_slices': SLICES,
        # },
        # 'loose_time_sliced_vdt_by_num': {
        #     'task_duration_hours':
        #     TASK_DURATIONS,
        #     'deadline_hours':
        #     DDL_DURATIONS,
        #     'num_slices': SLICES,
        # },
        # 'ideal_ilp_overhead': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': TIGHTER_DDL_DURATIONS,
        # },
        "quick_optimal": {
            "deadline_hours": DDL_DURATIONS,
            "task_duration_hours": TASK_DURATIONS,
        },
        "quick_optimal_paranoid": {
            "deadline_hours": DDL_DURATIONS,
            "task_duration_hours": TASK_DURATIONS,
        },
        # 'quick_optimal_sliced_by_num': {
        #     'deadline_hours': DDL_DURATIONS,
        #     'task_duration_hours': TASK_DURATIONS,
        #     'num_slices': [2, 4, 8, 16],
        # },
        "quick_optimal_more_sliced_by_num": {
            "deadline_hours": DDL_DURATIONS,
            "task_duration_hours": TASK_DURATIONS,
            "num_slices": [2, 4, 8, 16],
        },
        # 'rc_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        #     'keep_on_demand': [True]
        # },
        # 'rc_slack_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        # 'rc_vd_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        # 'rc_vdt_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        # 'rc_v2dt_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        "rc_cr_threshold": {
            "task_duration_hours": TASK_DURATIONS,
            "deadline_hours": DDL_DURATIONS,
        },
        "rc_cr_threshold_no_condition2": {
            "task_duration_hours": TASK_DURATIONS,
            "deadline_hours": DDL_DURATIONS,
        },
        # 'rc_cr_no_keep_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        # 'rc_1cr_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        "rc_next_spot_threshold": {
            "task_duration_hours": TASK_DURATIONS,
            "deadline_hours": DDL_DURATIONS,
        },
        # 'rc_next_spot_single_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        # 'rc_next_wait_spot_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        # 'rc_vdt_allow_idle_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        # 'ideal_ilp_overhead_sliced': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': TIGHTER_DDL_DURATIONS,
        #     'slice_interval_hours': [6, 12, 24],
        # },
        # 'ideal_ilp_overhead_sliced_by_num': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': TIGHTER_DDL_DURATIONS,
        #     'num_slices': [2, 4, 8, 16, 32]
        # },
        # 'rc_lw_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        # 'rc_ec_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        # 'rc_gec_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        # 'rc_vd_no_k_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        # },
        # 'rc_dc_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        #     'keep_on_demand': [True, False],
        #     'dc_scale': [1.0],
        #     # 'dc_scale': np.arange(0.2, 1.1, 0.2)
        # },
        # 'rc_dd_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        #     # 'keep_on_demand': [True, False],
        #     'keep_on_demand': [True],
        #     # 'dd_scale': np.arange(1.0, 7.2, 0.5),
        #     'dd_scale': [4.5],
        # },
        # 'rc_dda_threshold': {
        #     'task_duration_hours': TASK_DURATIONS,
        #     'deadline_hours': DDL_DURATIONS,
        #     'keep_on_demand': [True, False],
        # },
    }
    if args.no_ilp:
        STRATEGIES_WITH_SEARCH_CONFIGS.pop("ideal_ilp_overhead", None)
        STRATEGIES_WITH_SEARCH_CONFIGS.pop("ideal_ilp_overhead_sliced_by_num", None)
    if args.strategy is not None:
        STRATEGIES_WITH_SEARCH_CONFIGS = {
            "on_demand": STRATEGIES_WITH_SEARCH_CONFIGS["on_demand"],
            args.strategy: STRATEGIES_WITH_SEARCH_CONFIGS[args.strategy],
        }

    env_paths = []
    for env_name in os.listdir(REAL_DATA_PATH):
        if env_name.endswith(".json"):
            env_paths.append(os.path.join(REAL_DATA_PATH, env_name))

    jobs = []
    for env_path in env_paths:
        env_name = os.path.basename(env_path).rpartition(".")[0]
        result_path = os.path.join(RESULTS_DIR, f"{env_name}.csv")
        jobs.append(
            ray.remote(job_runner.run_real_jobs).remote(
                STRATEGIES_WITH_SEARCH_CONFIGS,
                env_path,
                EXP_DIR,
                result_path,
                STRATEGY_KWARGS,
                dump_history=args.dump_history,
                num_random_traces=args.num_traces,
                trace_length=None if args.trace_length is None else args.trace_length,
                force_rerun=args.force,
                force_strategy=args.update_strategy,
            )
        )
    ray.get(jobs)


def none_or_int(value):
    if value.lower() == "none":
        return None
    return int(value)


if __name__ == "__main__":
    import logging

    cpu_count = os.cpu_count()
    print(f"🚀 Initializing Ray, using {cpu_count} CPU cores")
    ray.init(num_cpus=cpu_count)

    root_logger = logging.getLogger("sky_spot")

    def setup_logger():
        logging_level = os.environ.get("LOG_LEVEL", "DEBUG")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
        handler.setLevel(logging_level)
        root_logger.addHandler(handler)

    setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=float, default=None)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--overhead", type=float, default=0.3)
    parser.add_argument("--ddl", type=int, default=None)
    parser.add_argument("--no-ilp", action="store_true")
    parser.add_argument("--strategy", type=str, default=None)
    parser.add_argument("--result-dir", type=str, default="results")
    parser.add_argument("--exp-dir", type=str, default="exp")
    parser.add_argument("--dump-history", action="store_true")
    parser.add_argument("--num-traces", type=int, default=300)
    parser.add_argument("--trace-length", type=none_or_int, default=None)
    parser.add_argument("--real-preemption", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--update-strategy", type=str, default=None)
    args = parser.parse_args()

    main(args)
