import logging
import os
import sys

import numpy as np
import ray
import tqdm
import wandb
import pandas as pd

from sky_spot import job_runner

# TODO(zhwu): try smaller gaps
ORIGINAL_GAP_SECONDS = 600
GAP_SECONDS = 180
ORIGINAL_SPOT_GAPS = 24.5 * ORIGINAL_GAP_SECONDS / GAP_SECONDS
ORIGINAL_WAIT_GAPS = 6.4 * ORIGINAL_GAP_SECONDS / GAP_SECONDS

ORIGIN_SPOT_FRACTION = ORIGINAL_SPOT_GAPS / (ORIGINAL_SPOT_GAPS +
                                             ORIGINAL_WAIT_GAPS)

DDL = 52

STRATEGY = 'strawman'
RESTART_OVERHEAD_HOURS = 0.1
STRATEGY_KWARGS = {
    'deadline_hours': DDL,
    'restart_overhead_hours': RESTART_OVERHEAD_HOURS,
    'max_total_slacks': None,
    'max_slice_slacks': None
}

EXP_DIR = os.path.abspath('exp/f_vs_gap')
RESULTS_PATH = os.path.abspath(
    f'results/greedy-optimal/restart={RESTART_OVERHEAD_HOURS:.2f}/f_vs_gap_two_exp-{GAP_SECONDS}.csv'
)

wandb.init(mode="disabled")
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    import logging

    ray.init()

    root_logger = logging.getLogger('sky_spot')

    def setup_logger():
        logging_level = os.environ.get('LOG_LEVEL', 'DEBUG')
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                '%(asctime)s %(levelname)s %(name)s %(message)s'))
        handler.setLevel(logging_level)
        root_logger.addHandler(handler)

    setup_logger()

    spot_fractions = [ORIGIN_SPOT_FRACTION]
    deltas = [i / 100 for i in range(10, 35, 10)]
    for delta in deltas:
        if ORIGIN_SPOT_FRACTION + delta <= 1:
            spot_fractions.append(ORIGIN_SPOT_FRACTION + delta)
        spot_fractions.append(ORIGIN_SPOT_FRACTION - delta)

    job_submission = tqdm.tqdm(desc='Job submission')
    exps = []
    spot_gaps, wait_gaps = [], []
    for spot_fraction in spot_fractions:
        wait_gap = ORIGINAL_WAIT_GAPS
        wait_gaps.append(wait_gap)
        spot_gap = int(wait_gap * spot_fraction / (1 - spot_fraction))
        spot_gaps.append(spot_gap)
    futures = []
    tmp_result_paths = []
    for spot_gap, wait_gap in zip(spot_gaps, wait_gaps):
        spot_fraction = spot_gap / (spot_gap + wait_gap)
        min_task_fraction = max(spot_fraction - 0.2, 0)
        task_fraction = list(np.arange(0.1, 0.5, 0.1)) + list(
            np.arange(min_task_fraction, 0.99, 0.025)) + [
                0.99, 0.95, 0.96, 0.97, 0.978, 0.98
            ] + [(52 - RESTART_OVERHEAD_HOURS) / 52 - 5e-4]
        search_configs = {
            'task_duration_hours': [DDL * t_f for t_f in task_fraction],
            'num_slices': list(range(1, 53)),
        }
        # for strategy in ['strawman', 'ideal_ilp_overhead']:
        for strategy in ['time_sliced_by_num', 'loose_time_sliced_by_num']:
            result_path = RESULTS_PATH.replace(
                '.csv', f'_{strategy}_{spot_gap:.2f}_{wait_gap:.2f}.csv.tmp')
            futures.append(
                ray.remote(job_runner.run_jobs).remote(
                    strategy, [spot_gap], [wait_gap], search_configs,
                    GAP_SECONDS, EXP_DIR, result_path, STRATEGY_KWARGS))
    ray.get(futures)
    dfs = []

    result_dir = os.path.dirname(RESULTS_PATH)
    for tmp_result_path in os.listdir(os.path.dirname(RESULTS_PATH)):
        if tmp_result_path.endswith('.csv.tmp'):
            dfs.append(pd.read_csv(os.path.join(result_dir, tmp_result_path)))

    df = pd.concat(dfs).drop_duplicates()
    df.to_csv(RESULTS_PATH, index=False)
