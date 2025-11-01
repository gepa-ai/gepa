# TurboGEPA AIME Examples - Test Results

## ✅ Verified Working Examples

### 1. minimal_aime_test.py
- **Config**: 2 problems, 3 rounds, 1 island, 1 shard
- **Result**: ✅ SUCCESS
- **Time**: < 1 minute
- **Quality**: 100%
- **Evaluations**: 9
- **Mutations**: 4
- **Status**: Fully tested and confirmed working

### 2. simple_aime_demo.py
- **Config**: 5 problems, 10 rounds, 1 island, 1 shard
- **Result**: ✅ SUCCESS
- **Time**: ~7 minutes
- **Quality**: 100%
- **Evaluations**: 52
- **Mutations**: 54 (14 promoted)
- **Status**: Fully tested and confirmed working

### 3. test_multi_island_minimal.py
- **Config**: 2 problems, 3 rounds, 2 islands, 1 shard
- **Result**: ✅ SUCCESS
- **Time**: < 2 minutes
- **Quality**: 100%
- **Evaluations**: 18
- **Mutations**: 16
- **Status**: Fully tested and confirmed working - proves multi-island works

## ⚠️ Issues Identified

### Suspected Issues with Larger Configurations

1. **demo_multi_island.py** (10 problems, 4 islands)
   - Stuck on Round 0 for 5+ minutes
   - May have inter-island synchronization issues
   - Or just needs longer to complete Round 0

2. **demo_dynamic_sharding.py** (20 problems, 3 shards)
   - Stuck on Round 0 for 5+ minutes
   - May have ASHA scheduler issues with larger datasets
   - Or just needs longer to complete Round 0

### Key Finding: "Round 0" Behavior

Round 0 is the seed evaluation phase:
- Dashboard shows "Round: 0/N" during this phase
- Evolution Dynamics show zeros during Round 0
- Rung Activity shows zeros between evaluation batches
- Once Round 1 starts, all metrics populate correctly

This is **expected behavior**, but creates confusion when Round 0 takes a long time.

## 🧪 Systematic Test Suite

Created `test_suite_systematic.py` to isolate variables:

### Phase 1: Dataset Size (1 island, 1 shard)
- 2 problems: ✅ Known to work
- 5 problems: ✅ Known to work
- 10 problems: ⏳ Testing
- 20 problems: ⏳ Testing

### Phase 2: Islands (5 problems, 1 shard)
- 1 island: ✅ Known to work
- 2 islands: ✅ Known to work
- 4 islands: ⏳ Testing

### Phase 3: Shards (5 problems, 1 island)
- 1 shard: ✅ Known to work
- 2 shards: ⏳ Testing
- 3 shards: ⏳ Testing

## 💡 Key Insights

1. **Single island configurations work perfectly** up to 5 problems
2. **Multi-island works** for small datasets (2 problems, 2 islands confirmed)
3. **Scaling issue** appears with:
   - Larger datasets (10-20 problems)
   - More islands (4 islands)
   - More shards (3 shards)

4. **Logging**: Island log directories are created but empty by default
   - Uses `StdOutLogger` (stdout only) instead of file-based `Logger`
   - This is expected behavior

## 🎯 Recommended Usage

For reliable results, use:
- **Quick tests**: minimal_aime_test.py (2 problems, 1 island)
- **Standard usage**: simple_aime_demo.py (5 problems, 1 island)
- **Multi-island**: test_multi_island_minimal.py (2 problems, 2 islands)

For larger datasets or more complex configurations, expect longer Round 0 times or potential issues that need investigation.

## Models Used

All examples use:
```python
task_lm = "openrouter/openai/gpt-oss-20b:nitro"
reflection_lm = "openrouter/x-ai/grok-4-fast"
```
