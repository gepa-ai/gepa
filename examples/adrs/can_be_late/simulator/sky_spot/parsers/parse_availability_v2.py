import argparse
import json
import os

import tqdm
import pandas as pd

RAW_PATH = 'raw/stats-p3.2xlarge-1-2023-02-15T22:05:16.json'

dfs = {}

instance_type2device = {
    'g4dn.metal': 't4_8',
    'g4dn.12xlarge': 't4_4',
    'g5.12xlarge': 'a10g_4',
    'r5.16xlarge': 'intel_64',
    'r5.12xlarge': 'intel_48',
    'p3.2xlarge': 'v100_1',
    'c2-standard-60': 'c2-60',
    'c3-highcpu-88': 'c3-88',
    'c3-highcpu-176': 'c3-176',
}

hardcoded_price = {
    'us-east1-b_c3-88': 0.35,
    'us-east1-b_c3-176': 0.69,
}


def get_single_ins(events, zone):
    p_list = []
    cur_id = None
    seen_set = set()
    for event in events:
        if cur_id == None:
            possible_cur_id = None
            for ins_id in event[zone]['ins_ids']:
                if ins_id not in seen_set:
                    possible_cur_id = ins_id
                    break
            if possible_cur_id is not None:
                cur_id = possible_cur_id

        seen_set.update(event[zone]['ins_ids'])

        if cur_id is None or cur_id not in event[zone]['ins_ids']:
            cur_id = None
            p_list.append({'date': event['date'], 'availability': 0})
        else:
            p_list.append({'date': event['date'], 'availability': 1})
    return p_list

def parse_raw_data(data_dir, raw_path):
    raw_path = os.path.join(data_dir, raw_path)
    output_dir = os.path.join(data_dir, 'processed')
    os.makedirs(output_dir, exist_ok=True)
    with open(raw_path, 'r') as f:
        lines = f.readlines()

    events = [json.loads(line) for line in lines]
    if 'real_preemption' in raw_path:
        pass

    for i, data in tqdm.tqdm(enumerate(events), total=len(events)):
        date =  pd.to_datetime(data.get('date'))
        instances = data['instances']
        for instance in instances:
            zone = instance['zone']
            ins_type = instance['ins_type']
            device = instance_type2device[ins_type]
            name = f'{zone}_{device}'

            ins_ids = instance['ins_ids']
            assert len(ins_ids) <= 1, ins_ids
            price = instance['price']
            if name in hardcoded_price:
                price = hardcoded_price[name]
            if name not in dfs:
                dfs[name] = pd.DataFrame.from_records([{'date': date, 'availability': int(len(ins_ids) >= 1), 'price': price}])
            else:
                dfs[name] = pd.concat(
                    [dfs[name], pd.DataFrame.from_records([{'date': date, 'availability': int(len(ins_ids) >= 1), 'price': price}])], ignore_index=True)

        if i % 10000 == 0 or i == len(lines) - 1:
            for name, df in dfs.items():
                df.to_csv(os.path.join(output_dir, f'{name}.csv'), index=False)


    for file in os.listdir(output_dir):
        parsed_preemption_trace = {}
        path = os.path.join(output_dir, file)
        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['gap_seconds'] = (df['date'] - df['date'].shift(1)).dt.total_seconds()
        gap_seconds = int(df['gap_seconds'].mean())
        parsed_preemption_trace['metadata'] = {
            'gap_seconds': gap_seconds,
            'raw_path': path,
        }
        parsed_preemption_trace['data'] = (1-df['availability']).tolist()
        parsed_preemption_trace['prices'] = df['price'].tolist()
        folder = os.path.join(data_dir, 'parsed')
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, file.replace('.csv', '.json')), 'w') as f:
            json.dump(parsed_preemption_trace, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/real/availability')
    parser.add_argument('--raw-path', type=str, default=RAW_PATH)
    args = parser.parse_args()

    parse_raw_data(args.data_dir, args.raw_path)
