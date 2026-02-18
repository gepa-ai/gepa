import logging

import ray
import wandb

from sky_spot import job_runner

# TODO(zhwu): try smaller gaps
ORIGINAL_GAP_SECONDS = 600
GAP_SECONDS = 180

RESTART_OVERHEAD_HOURS = [0.05] + [i / 100 for i in range(6, 51, 1)]
# RESTART_OVERHEAD_HOURS = [0.01] + [i / 100 for i in range(2, 18)]
NUM_SLICES = list(range(1, 53))

# RESTART_OVERHEAD_HOURS = RESTART_OVERHEAD_HOURS[:1]
NUM_SLICES = NUM_SLICES[::3]

RESTART_OVERHEAD_HOURS = [0.1]

STRATEGY = 'time_sliced_by_num'
STRATEGY_KWARGS = {
    # 'max_total_slacks': None,
    # 'max_slice_slacks': None,
}

EXP_DIR = 'exp-time-sliced-test/n_vs_d'
RESULT_PATH = 'results-new-sliced-test/n_vs_d_two_exp.csv'
if GAP_SECONDS != ORIGINAL_GAP_SECONDS:
    RESULT_PATH = RESULT_PATH.replace('.csv', f'-{GAP_SECONDS}.csv')

wandb.init(mode="disabled")
logger = logging.getLogger(__name__)



if __name__ == '__main__':
    import logging

    ray.init()

    original_spot_gaps = 24.5 * ORIGINAL_GAP_SECONDS / GAP_SECONDS
    original_wait_gaps = 6.4 * ORIGINAL_GAP_SECONDS / GAP_SECONDS

    s_w = original_spot_gaps / original_wait_gaps

    spot_alive_gaps = [original_spot_gaps]
    INTERVAL = 4 * ORIGINAL_GAP_SECONDS // GAP_SECONDS
    for delta in list(range(INTERVAL, 3 * INTERVAL, INTERVAL)):
        spot_alive_gaps.append(original_spot_gaps + delta)
        spot_alive_gaps.append(original_spot_gaps - delta)

    spot_gaps, wait_gaps = [], []
    for spot_gap in spot_alive_gaps:
        wait_gap = spot_gap / s_w
        spot_gaps.append(spot_gap)
        wait_gaps.append(wait_gap)
    search_configs = {
        'restart_overhead_hours': RESTART_OVERHEAD_HOURS,
        'num_slices': NUM_SLICES,
    }
    job_runner.run_jobs(STRATEGY, spot_gaps, wait_gaps, search_configs, GAP_SECONDS, EXP_DIR, RESULT_PATH, STRATEGY_KWARGS)
