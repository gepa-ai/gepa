import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=int, help="Dataset size", default=30)
parser.add_argument("concurrencies", nargs="+", type=int)
parser.add_argument("--runtime", type=int, default=120)
parser.add_argument("--evals", type=int, default=120)
parser.add_argument("--target-quality", type=float, default=0.5)
parser.add_argument("--target-shard", type=float, default=0.4)
args = parser.parse_args()

for conc in args.concurrencies:
    print(f"\n=== Running conc={conc} ===")
    cmd = [
        "bash","-lc",
        f"rm -rf .turbo_gepa && mkdir -p .turbo_gepa && "
        f"source .envrc && source .venv/bin/activate && "
        f"/usr/bin/time -p python examples/aime_benchmark_v2.py --mode turbo --dataset-size {args.dataset} "
        f"--task-lm openrouter/openai/gpt-oss-20b:nitro --reflection-lm openrouter/x-ai/grok-4-fast "
        f"--turbo-target-shard {args.target_shard} --turbo-target-quality {args.target_quality} "
        f"--turbo-max-runtime {args.runtime} --turbo-max-evaluations {args.evals} "
        f"--no-turbo-auto-scale --turbo-eval-concurrency {conc}"
    ]
    subprocess.run(cmd, check=False)
    metrics_files = sorted(Path(".turbo_gepa/metrics").glob("metrics_*.txt"))
    if metrics_files:
        print(metrics_files[-1].read_text())
