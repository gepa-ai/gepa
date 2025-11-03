import os
import shutil
from pathlib import Path

Path(".turbo_gepa/").exists() and shutil.rmtree(".turbo_gepa/")

os.environ["LITELLM_LOG"] = "ERROR"
os.environ["LITELLM_DISABLE_LOGGING_WORKER"] = "True"

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

trainset, _, _ = gepa.examples.aime.init_dataset()
dataset = [
    DefaultDataInst(input=ex['input'], answer=ex['answer'], id=f'aime_{i}')
    for i, ex in enumerate(trainset[:10])
]

config = Config(
    eval_concurrency=4,
    n_islands=1,
    shards=(0.6, 1.0),
    batch_size=4,
    max_mutations_per_round=4,
    mutation_buffer_min=3,
    queue_limit=16,
    log_level='INFO',
    adaptive_shards_enabled=False,
    max_optimization_time_seconds=120,
)

adapter = DefaultAdapter(
    dataset=dataset,
    task_lm='openrouter/openai/gpt-oss-20b:nitro',
    reflection_lm='openrouter/x-ai/grok-4-fast',
    config=config,
    auto_config=False,
)

# Monkey patch to add debug logging
original_promote_ready = adapter.orchestrator.scheduler.promote_ready
def debug_promote_ready():
    result = original_promote_ready()
    if result:
        print(f"\n🔍 DEBUG: promote_ready() returned {len(result)} candidates")
        for c in result:
            print(f"   - {c.fingerprint[:12]}...")
    return result
adapter.orchestrator.scheduler.promote_ready = debug_promote_ready

original_stream_launch_ready = adapter.orchestrator._stream_launch_ready
async def debug_stream_launch_ready(window_id, max_evals):
    pq_size_before = len(adapter.orchestrator._priority_queue)
    if pq_size_before > 0:
        print(f"\n🔍 DEBUG: _stream_launch_ready() called, priority_queue has {pq_size_before} candidates")
    result = await original_stream_launch_ready(window_id, max_evals)
    pq_size_after = len(adapter.orchestrator._priority_queue)
    if pq_size_before > 0 or result > 0:
        print(f"🔍 DEBUG: Launched {result} candidates, priority_queue now has {pq_size_after}")
    return result
adapter.orchestrator._stream_launch_ready = debug_stream_launch_ready

print('Starting with debug logging...\n')
result = adapter.optimize(
    seeds=['You are a helpful assistant. Answer the math question and provide your final answer in the format "### <answer>"'],
    max_rounds=2,
    enable_auto_stop=False,
    display_progress=False,
)

pareto = result.get('pareto_entries', [])
rungs = sorted(set(e.result.shard_fraction for e in pareto))
print(f'\n=== RESULT ===')
print(f'Rungs: {[f"{int(r*100)}%" for r in rungs]}')
