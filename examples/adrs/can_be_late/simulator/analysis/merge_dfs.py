import os

import pandas as pd

RESULTS_PATH = os.path.abspath(
    f'results/greedy-optimal/restart=0.10/f_vs_gap_two_exp-180.csv')

dfs = []
result_dir = os.path.dirname(RESULTS_PATH)
for tmp_result_path in os.listdir(os.path.dirname(RESULTS_PATH)):
    if tmp_result_path.endswith('.csv.tmp'):
        dfs.append(pd.read_csv(os.path.join(result_dir, tmp_result_path)))

df = pd.concat(dfs).drop_duplicates()
print(len(df))
df.drop_duplicates(inplace=True)
print(len(df))
df.to_csv(RESULTS_PATH, index=False)
