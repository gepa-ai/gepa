import argparse
import os
import pandas as pd

REMOVE_STRATEGIES = [
    # 'strawman',
    # 'time_sliced_by_num',
    # 'loose_time_sliced_by_num',
    # 'rc_threshold',
    # 'rc_lw_threshold',
    # 'only_spot'
    # 'rc_dc_threshold',
    # 'rc_dd_threshold',
    # 'ideal_ilp_overhead',
    # 'rc_next_spot_threshold',
    # 'rc_next_wait_spot_threshold',
    # 'quick_optimal_more_sliced_by_num',
    'rc_cr_threshold',
]
    
KEEP_STRATEGIES = [
    'ideal_ilp_overhead',
    'loose_time_sliced_by_num',
    'ideal_ilp_overhead_sliced_by_num',
    'on_demand',
]


def migrate(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if not file.endswith('.csv') and not file.endswith('.csv.tmp'):
                continue
            df = pd.read_csv(os.path.join(root, file))
            try:
                df = df[~df['strategy'].isin(REMOVE_STRATEGIES)]
            except KeyError:
                print(file)
                raise
            # df = df[df['strategy'].isin(KEEP_STRATEGIES)]
            new_root = root.replace(input_dir, output_dir)
            os.makedirs(new_root, exist_ok=True)
            df.drop_duplicates(inplace=True)
            # df = df.sort_values(by=[
            #     'strategy', 'avg_spot_hours', 'avg_wait_hours',
            #     'task_duration_hours', 'num_slices'
            # ])
            df.to_csv(os.path.join(new_root, file), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    migrate(args)
