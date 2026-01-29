# Fair Comparison: Subprocess Isolation for KernelBench

## Problem Statement

KernelBench evaluates LLM-generated CUDA kernels against baseline PyTorch implementations. For accurate scoring, both must be measured under identical conditions. The question: **Is subprocess isolation the right architecture, and how do we ensure fair baseline comparisons?**

---

## Architecture Decision: Subprocess Isolation

### The Argument

> "Subprocess isolation is superior because it provides crash and memory isolation without sacrificing measurement accuracy. The 1-2 second wall-clock overhead is NOT captured by CUDA event timing."

### Verdict: Correct

| Question | Answer |
|----------|--------|
| **Does process overhead leak into CUDA event timing?** | No. `torch.cuda.Event.elapsed_time()` measures GPU-side time only. Process spawn, Python init, CUDA context creation are CPU-side and complete before `record()`. |
| **Are there kernel performance differences in fresh subprocess?** | Minimal, with proper warm-up. Main concerns are GPU power state (may drop to base clocks if idle) and L2 cache cold start. Both are addressable. |
| **Does subprocess help with resource management?** | Yes, significantly. VRAM fragmentation resets, sticky CUDA errors are isolated, memory leaks cannot accumulate. |
| **Is 1-2s wall-clock penalty justified?** | Yes. For LLM-generated kernels that may crash, hang, or corrupt memory, isolation is mandatory. The overhead doesn't affect measurement precision. |

### Key Technical Details

**CUDA Event Timing Chain:**
```
CPU: spawn → Python init → CUDA context init → ...
GPU:                                            record(start) → kernel → record(end)
                                                ↑________________________↑
                                                   Only this is measured
```

**Risks to Mitigate:**
1. **GPU power state**: Add warm-up kernels before timing to ensure boost clocks
2. **L2 cache cold start**: 5 warmup iterations should suffice; consider explicit cache flush for consistency
3. **Clock drift**: Lock GPU clocks with `nvidia-smi -lgc <freq>` during benchmarking

---

## The Fair Comparison Problem

### Current State

- **LLM kernels**: Evaluated in subprocess via `execute_kernel()` with full isolation
- **Baselines**: Pre-computed JSON files from offline runs (unknown conditions)

This is **not a fair comparison**. The baseline times were measured under different:
- GPU warm-up state
- L2 cache conditions
- Process context
- Potentially different hardware

### Solution: Run Baselines Through Same Subprocess

**Key Insight**: `eval_kernel_against_ref()` compares `Model` (from ref_arch) vs `ModelNew` (from custom code). If we wrap the ref_arch so it provides both classes, we can measure baseline PyTorch through the same path.

```python
def execute_baseline(ref_arch: str, timeout: int = 360, device: int | None = None) -> dict:
    """Run ref_arch PyTorch through same subprocess as LLM kernels.

    No modification to KernelBench repo needed - this is a pure wrapper.
    """
    # Wrap ref_arch so it also defines ModelNew (inherits from Model)
    wrapped_code = f'''{ref_arch}

# For baseline measurement: ModelNew = Model (identical implementation)
class ModelNew(Model):
    pass
'''
    return execute_kernel(code=wrapped_code, ref_arch=ref_arch, timeout=timeout, device=device)
```

**Why This Works:**
1. `eval_kernel_against_ref()` loads `Model` from ref_arch, `ModelNew` from wrapped_code
2. Since `ModelNew` inherits `Model`, outputs are identical → correctness passes
3. Performance timing captures pure PyTorch reference time
4. Same subprocess isolation, GPU acquisition, timeout, warm-up protocol

### Comparison

| Aspect | Pre-computed JSON | Subprocess Baseline |
|--------|-------------------|---------------------|
| Hardware match | Fixed at generation time | Current hardware |
| Measurement conditions | Unknown | Identical to LLM eval |
| L2 cache state | Unknown | Same cold-cache protocol |
| GPU power state | Unknown | Same warm-up protocol |
| Crash isolation | None | Full |
| Reproducibility | Static | Dynamic, hardware-aware |

---

## Implementation Plan

### Phase 1: Add `execute_baseline()` to eval.py

```python
def execute_baseline(ref_arch: str, timeout: int = 360, device: int | None = None) -> dict:
    """Run baseline (ref_arch PyTorch) through same subprocess as LLM kernels."""
    wrapped_code = f'''{ref_arch}

class ModelNew(Model):
    pass
'''
    return execute_kernel(code=wrapped_code, ref_arch=ref_arch, timeout=timeout, device=device)
```

### Phase 2: Update main.py to use dynamic baselines

Option A: **Measure all baselines at startup**
```python
baseline_times = {}
for ex in dataset:
    result = execute_baseline(ex.ref_arch)
    baseline_times[ex.problem_id] = result["PerformanceStatsMean"]
```

Option B: **Lazy measurement** (measure baseline only when evaluating that problem)
```python
# In fitness function, before scoring:
if use_dynamic_baseline:
    baseline_result = execute_baseline(ref_arch)
    baseline = baseline_result["PerformanceStatsMean"]
else:
    baseline = ex.baseline_time  # from JSON
```

### Phase 3: Warm-up Protocol

Add explicit GPU warm-up in subprocess before timing:

```python
def _gpu_warmup(device):
    """Ensure GPU is at boost clocks before timing."""
    torch.cuda.set_device(device)
    for _ in range(10):
        x = torch.randn(1024, 1024, device=f'cuda:{device}')
        y = x @ x
    torch.cuda.synchronize()
```

---

## Summary

1. **Subprocess isolation is correct** for crash safety and doesn't affect CUDA timing precision
2. **Current baselines are unfair** because they were measured under different conditions
3. **Solution**: Wrap ref_arch to provide `ModelNew = Model`, run through same `execute_kernel()` path
4. **No changes to KernelBench repo** required - pure wrapper in eval.py
5. **Result**: Apples-to-apples comparison with identical measurement conditions
