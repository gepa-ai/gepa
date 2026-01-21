import json
import pathlib
import numpy as np

NUM_NODES = 16

data_dir = pathlib.Path(f'data/real/multinode/{NUM_NODES}-nodes').absolute()
print(data_dir)

total_binary_cnt = 0
total_cnt = 0
for path in data_dir.glob('*.json'):
    with path.open('r') as f:
        data = json.load(f)
        avail_nodes = np.array(data['data'])
        print('path:', path)
        binary_cnt = sum(avail_nodes == 0) + sum(avail_nodes == NUM_NODES)
        total_binary_cnt += binary_cnt
        total_cnt += len(avail_nodes)
        print(f'binary percentage: {binary_cnt / len(avail_nodes):.1%}')
        print('-' * 20)


print(f'total binary percentage: {total_binary_cnt / total_cnt:.1%}')
