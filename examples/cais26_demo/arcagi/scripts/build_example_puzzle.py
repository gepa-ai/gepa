"""One-shot prep: run the evolved agent on a single ARC puzzle and bundle
puzzle + seed prediction + evolved prediction into artifacts/example_puzzle.json.

The notebook then loads this file in reviewer mode (no network).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GEPA_EXAMPLES = ROOT.parent.parent / "gepa" / "examples" / "arc_agi"
sys.path.insert(0, str(GEPA_EXAMPLES))

from utils import run_agent  # noqa: E402

PROBLEM_ID = "bda2d7a6"
TASK_DIR = "task_1"
MODEL_ID = "openrouter/google/gemini-3-flash-preview"

valset_dir = ROOT / "artifacts" / "generated_best_outputs_valset"
if not valset_dir.exists():
    import zipfile
    zip_path = valset_dir.with_suffix(".zip")
    print(f"Unzipping {zip_path.name} (one-time)...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(valset_dir.parent)

seed_cache_path = valset_dir / TASK_DIR / "iter_0_prog_0.json"
seed_cache = json.loads(seed_cache_path.read_text())
assert seed_cache["problem_id"] == PROBLEM_ID

train_in = [ex["input"] for ex in seed_cache["train_examples"]]
train_out = [ex["gold"] for ex in seed_cache["train_examples"]]
test_in = [ex["input"] for ex in seed_cache["test_examples"]]
test_out = [ex["gold"] for ex in seed_cache["test_examples"]]

best_agent_code = (ROOT / "best_agent.py").read_text()
seed_agent_code = seed_cache["agent_code"]

print(f"Running evolved agent on puzzle {PROBLEM_ID} ({len(train_in)} train, {len(test_in)} test)...")
result = run_agent(
    agent_code=best_agent_code,
    train_in=train_in,
    train_out=train_out,
    test_in=test_in,
    test_out=test_out,
    model_id=MODEL_ID,
    max_llm_calls=10,
)

llms = result.pop("llms")
trace = llms.get_traces()

evolved_predictions_train = [ex["prediction"] for ex in result["train_examples"]]
evolved_predictions_test = [ex["prediction"] for ex in result["test_examples"]]
seed_predictions_train = [ex["prediction"] for ex in seed_cache["train_examples"]]
seed_predictions_test = [ex["prediction"] for ex in seed_cache["test_examples"]]

bundle = {
    "problem_id": PROBLEM_ID,
    "model_id": MODEL_ID,
    "puzzle": {
        "train_inputs": train_in,
        "train_outputs": train_out,
        "test_inputs": test_in,
        "test_outputs": test_out,
    },
    "seed": {
        "agent_code": seed_agent_code,
        "training_score": seed_cache["training_score"],
        "test_score": seed_cache["test_score"],
        "predictions_train": seed_predictions_train,
        "predictions_test": seed_predictions_test,
        "train_examples_detail": seed_cache["train_examples"],
        "test_examples_detail": seed_cache["test_examples"],
        "llm_calls": seed_cache["llm_calls"],
        "total_cost": seed_cache["total_cost"],
        "trajectory": seed_cache.get("trajectory", []),
    },
    "evolved": {
        "agent_code": best_agent_code,
        "training_score": result["training_score"],
        "test_score": result["test_score"],
        "predictions_train": evolved_predictions_train,
        "predictions_test": evolved_predictions_test,
        "llm_calls": trace["llm_calls"],
        "total_cost": trace["total_cost"],
        "trajectory": trace["trajectory"],
        "error": result["error"],
    },
}

out_path = ROOT / "artifacts" / "example_puzzle.json"
out_path.write_text(json.dumps(bundle, indent=2))
print(f"  seed   : train={seed_cache['training_score']:.2f} test={seed_cache['test_score']:.2f}")
print(f"  evolved: train={result['training_score']:.2f} test={result['test_score']:.2f}")
print(f"  evolved cost: ${trace['total_cost']:.4f}, calls={trace['llm_calls']}")
print(f"Wrote {out_path}")
