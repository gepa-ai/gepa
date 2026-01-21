import json
import pandas as pd
import os
import numpy as np

from sky_spot.env import TraceEnv

EXP_PATH = '../../exp-for-plot/real/ddl=search+task=48+overhead=0.20/real'
# exp_path = '../../exp-for-plot/real/2023-02-15/ddl=search+task=48+overhead=0.20/real'

df = pd.DataFrame(columns=[
    'trace', 'strategy', 'deadline_hours', 'task_hLimited Deficit', 'gap_seconds',
    'avg_spot_hLimited Deficit', 'avg_wait_hLimited Deficit', 'cost', 'num_slices'
])


def get_consecutive_1s(arr):
    padded = np.concatenate(([0], arr, [0]))
    diff = padded[1:] - padded[:-1]
    lengths = np.where(diff == -1)[0] - np.where(diff == 1)[0]
    return lengths

def get_dfs_from_exp(exp_path=EXP_PATH):
    results = []
    for env_dir in os.listdir(exp_path):
        trace_path = os.path.join(exp_path, env_dir, 'traces', 'random_start')
        envs = TraceEnv.create_env(trace_path, 0)
        for exp in os.listdir(os.path.join(exp_path, env_dir)):
            if 'deadline_hours' not in exp:
                continue
            result_path = os.path.join(exp_path, env_dir, exp, 'result.json')
            if not os.path.exists(result_path):
                continue
            with open(result_path) as f:
                exp_result = json.load(f)
            strategy = exp_result['strategy']['name']
            deadline_hours = exp_result['strategy']['deadline'] / 3600
            task_hours = exp_result['strategy']['task_duration'] / 3600
            gap_seconds = exp_result['env']['metadata']['gap_seconds']

            for i, cost in enumerate(exp_result['costs']):
                env = envs[i]
                preemption_trace = np.array(
                    env.get_trace_before_end(deadline_hours * 3600))
                spot_hours = get_consecutive_1s(
                    1 - preemption_trace) * gap_seconds / 3600
                avg_spot_hours = spot_hours.mean()
                avg_wait_hours = get_consecutive_1s(
                    preemption_trace).mean() * gap_seconds / 3600
                results.append({
                    'trace':
                    os.path.join(trace_path, f'{i}.json'),
                    'env':
                    env_dir.replace('.json', ''),
                    'strategy':
                    strategy,
                    'deadline_hours':
                    deadline_hours,
                    'task_hours':
                    task_hours,
                    'task_fraction':
                    task_hours / deadline_hours,
                    'gap_seconds':
                    gap_seconds,
                    'avg_spot_hours':
                    avg_spot_hours,
                    'avg_wait_hours':
                    avg_wait_hours,
                    'num_slices':
                    exp_result['strategy'].get('num_pairs', None),
                    'spot_fraction':
                    spot_hours.sum() / deadline_hours,
                    'cost':
                    cost,
                })

    df = pd.DataFrame(results)
    return df


def add_bar_annotations(ax, errors=None, value_precision=0, error_precision=1):
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        error_value = ''
        if error_precision is not None:
            error_value = f" ± {errors[i]:.{error_precision}f}" if errors is not None else ""
        annotation = f"{height:.{value_precision}f}{error_value}"
        ax.annotate(
            annotation,
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )

def add_arrow_and_annotation(ax, x1, x2, text_shift, y1, y2, text, color):
    print(x1, x2, y1, y2, text, color)
    ax.annotate("", xy=(x1, y2), xytext=(x2, y1), arrowprops=dict(arrowstyle="->", color=color, zorder=100))
    ax.text(x2 + text_shift, y1 + (y2 - y1)/2 + 1.5, text, ha="right", va="center", fontsize=7, color=color, zorder=100)


OUR_NAME = 'Uniform Progress'
