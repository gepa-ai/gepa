# %%
import json
import pandas as pd
import pathlib
TRACE = 'origin_traces/2022-10-26T22-05'

trace_path = pathlib.Path(TRACE).expanduser().absolute()
traces = {}
if trace_path.is_file():
    traces = {trace_path.stem: pd.read_csv(trace_path)}
else:
    for path in trace_path.glob('*.txt'):
        trace = pd.read_csv(path, names=['index', 'time', 'preempted'])
        traces[path.stem] = trace

# %%
for trace_name, trace in traces.items():
    data = {
        'metadata': {'gap_seconds': 600},
        'data': trace['preempted'].tolist(),
    }
    with pathlib.Path(__file__).parent.joinpath(f'{trace_name}.json').open('w') as f:
        json.dump(data, f)
