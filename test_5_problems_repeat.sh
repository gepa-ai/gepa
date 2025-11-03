#!/bin/bash
for i in 1 2 3; do
  echo "=== Run $i ==="
  rm -rf .turbo_gepa
  PYTHONPATH=src .venv/bin/python -c "
import os
os.environ['LITELLM_LOG'] = 'ERROR'

import gepa
from turbo_gepa.adapters.default_adapter import DefaultAdapter, DefaultDataInst
from turbo_gepa.config import Config

trainset, _, _ = gepa.examples.aime.init_dataset()
dataset = [DefaultDataInst(input=ex['input'], answer=ex['answer'], id=f'aime_{i}') for i, ex in enumerate(trainset[:5])]

config = Config(eval_concurrency=2, n_islands=1, shards=(0.6, 1.0), batch_size=2, max_mutations_per_round=1, log_level='INFO', adaptive_shards_enabled=False, max_optimization_time_seconds=45)
adapter = DefaultAdapter(dataset=dataset, task_lm='openrouter/openai/gpt-oss-20b:nitro', reflection_lm='openrouter/x-ai/grok-4-fast', config=config, auto_config=False)

result = adapter.optimize(seeds=['Answer with ### answer'], max_rounds=2, enable_auto_stop=False, display_progress=False)

rungs = sorted(set(e.result.shard_fraction for e in result.get('pareto_entries', [])))
print(f'Rungs: {[f\"{int(r*100)}%\" for r in rungs]}')
if 1.0 in rungs:
    print('✅ SUCCESS - Reached 100%')
else:
    print('❌ FAILED - Did not reach 100%')
" 2>&1 | grep -E "Rungs|SUCCESS|FAILED"
  echo
done
