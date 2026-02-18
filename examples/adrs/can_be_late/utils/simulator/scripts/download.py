import argparse
import os
import subprocess
import tempfile

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('--strategy', type=str, default=None)
    args = parser.parse_args()
    tmp_dir = tempfile.mkdtemp()
    subprocess.run(['rsync', '-Pavz', args.source, tmp_dir])
    for root, dirs, files in os.walk(tmp_dir):
        print(root, dirs, files)
        for file in files:
            if not file.endswith('.csv'):
                continue
            file_path = os.path.join(root, file)
            new_df = pd.read_csv(file_path)
            if args.strategy is not None:
                new_df = new_df[new_df['strategy'].str.startswith(args.strategy)]
            try:
                origin_df = pd.read_csv(file_path.replace(tmp_dir, args.target))
            except (FileNotFoundError, pd.errors.EmptyDataError):
                origin_df = pd.DataFrame()
            new_df = pd.concat([origin_df, new_df])
            new_df.drop_duplicates(inplace=True)
            target_file_path = file_path.replace(tmp_dir, args.target)
            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
            new_df.to_csv(target_file_path, index=False)
            print(target_file_path)

