import pathlib

EXP_DIR = pathlib.Path('../../exp-new-sliced')
RESULT_DIR = pathlib.Path('../../results-new-sliced')


TRACE = 'two_exp'
TRACE_DIRS = {
    'real': pathlib.Path('../../data/real/ping_based/random_start_time'),
    'two_exp': pathlib.Path(
    '../../data/two_exp/gap_600-real_mean')
}

TRACE_NAMES = {
    'real': 'real_overhead={overhead}',
    'two_exp': 'exp-gap_600-real_mean_over={overhead}'
}

TRACE_TYPES = {
    'real': 'us-west-2a_v100_1',
    'two_exp': 'exp_gap_600-real_mean'
}
