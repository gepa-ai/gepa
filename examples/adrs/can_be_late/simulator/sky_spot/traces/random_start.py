import argparse
import pathlib
import json
from typing import Optional
import numpy as np

# TRACE = '../../data/two_exp/gap_600-alive_0.4202000000080931-wait_0.2366999999982129/0.json'
# TRACE = 'data/two_gamma/gap_600-alive_0.1919_0.008700000000002435-wait_0.358_0.06579999999981707/0.json'
# TRACE = 'data/two_gamma/gap_600-alive_5.0_0.1-wait_5.0_0.1/0.json'
# TRACE = 'data/two_exp/gap_600-alive_0.02-wait_0.02/0.json'
# TRACE = 'data/two_exp/gap_600-alive_0.04082965866405357-wait_0.1559405943681992/0.json'
# TRACE = 'data/two_exp/gap_600-preemption_rate_0.2/0.json'
# TRACE = 'data/two_exp/gap_600-preemption_rate_0.9/0.json'
# TRACE = 'data/two_exp/gap_600-real_mean-longer/0.json'
TRACE = 'traces/2022-10-26T22-05/us-west-2a_v100_1.txt'
# TRACE = 'traces/2022-10-26T22-05/us-west-2b_v100_1.txt'
# TRACE = 'traces/2022-10-26T22-05/us-west-2a_k80_1.txt'
# TRACE = 'traces/2022-10-26T22-05/us-west-2b_k80_1.txt'


def generate_random_traces(trace_str: str,
                           *,
                           num_traces: int = 1000,
                           trace_length_hours: int = 52,
                           output_dir: Optional[str] = None):
    """Generate random traces from a given trace.
    """
    trace_path = pathlib.Path(trace_str).expanduser().absolute()
    if output_dir is None:
        output_dir_path = trace_path.parent / f'{trace_path.parent.name}_random'
    else:
        output_dir_path = pathlib.Path(output_dir).expanduser().absolute()
    output_dir_path.mkdir(exist_ok=True)

    with trace_path.open('r', encoding='utf-8') as f:
        trace = json.load(f)

    avail_traces_month = np.array(trace['data'])
    prices = trace.get('prices')
    # print(len(avail_traces_month))

    gap_seconds = trace['metadata']['gap_seconds']
    trace_length = trace_length_hours * 3600 // gap_seconds + 2

    for i in range(num_traces):
        np.random.seed(i)
        start_time = np.random.randint(0,
                                       len(avail_traces_month) - trace_length)
        end_time = start_time + trace_length
        # print(start_time, end_time)
        output_path = output_dir_path / f'{i}.json'
        data = avail_traces_month[start_time:end_time]
        trace = {
            'generator': 'random_start_time',
            'original_trace': str(trace_path.relative_to(pathlib.Path.cwd())),
            'metadata': {
                'gap_seconds': gap_seconds,
                'start_time': start_time,
                'end_time': end_time,
            },
            'data': data.tolist(),
        }
        if prices is not None:
            trace['prices'] = prices[start_time:end_time]
        with output_path.open('w') as f:
            json.dump(trace, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', type=str, default=TRACE)
    parser.add_argument('--num-traces', default=1000, type=int)
    parser.add_argument('--output-dir', default='exp', type=str)
    args = parser.parse_args()
    generate_random_traces(args.trace,
                           num_traces=args.num_traces,
                           output_dir=args.output_dir)
