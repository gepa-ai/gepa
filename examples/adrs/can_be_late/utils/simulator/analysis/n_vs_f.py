import logging
import os
import sys

import ray
import tqdm
import wandb

from sky_spot import job_runner

# TODO(zhwu): try smaller gaps
ORIGINAL_GAP_SECONDS = 600
GAP_SECONDS = 180
ORIGINAL_SPOT_GAPS = 24.5 * ORIGINAL_GAP_SECONDS / GAP_SECONDS
ORIGINAL_WAIT_GAPS = 6.4 * ORIGINAL_GAP_SECONDS / GAP_SECONDS

ORIGIN_SPOT_FRACTION = ORIGINAL_SPOT_GAPS / (ORIGINAL_SPOT_GAPS + ORIGINAL_WAIT_GAPS)

RESTART_OVERHEAD_HOURS = [0.06, 0.1, 0.2, 0.3]
NUM_SLICES = list(range(1, 53))
# NUM_SLICES = NUM_SLICES[2:4]

STRATEGY = 'loose_time_sliced_by_num'
STRATEGY_KWARGS = {
    'max_total_slacks': None,
    'max_slice_slacks': None,
}

EXP_DIR = 'exp-time-sliced/n_vs_f'
RESULTS_PATH = 'results-new-sliced/n_vs_f_two_exp.csv'

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
    deltas = [i / 100 for i in range(2, 55, 2)]
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
    search_configs = {
        'restart_overhead_hours': RESTART_OVERHEAD_HOURS,
        'num_slices': NUM_SLICES,
    }
    job_runner.run_jobs(STRATEGY, spot_gaps, wait_gaps, search_configs, GAP_SECONDS, EXP_DIR, RESULTS_PATH, STRATEGY_KWARGS)
