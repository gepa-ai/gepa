# TurboGEPA Promotion & Evolution Stabilization Plan

Goal: eliminate the promotion and queue overload issues observed in the latest OSS‑20B runs, so TurboGEPA reliably converges without flooding the final rung.

---

## A. Promotion Accuracy & Metrics

| Priority | Task | Status | Notes |
|----------|------|--------|-------|
| 🔴 | Track promotion attempts/success per rung | ✅ Done | `Metrics` now records attempts/prunes by rung (Nov 13) |
| 🔴 | Require real parent rung score before comparing | ✅ Done | Scheduler now uses parent_rung_scores before shrinkage (Nov 13) |
| 🟠 | Tighten rung‑0 tolerance / add extra rung | ✅ Done | First rung tolerance now capped at 10% (Nov 13) |
| 🟠 | Re-enable final-rung straggler detaching | ✅ Done | Evaluator no longer skips final-rung straggler cutoff (Nov 13) |

---

## B. Objective Handling

| Priority | Task | Status | Notes |
|----------|------|--------|-------|
| 🟡 | Thread `promote_objective` through evaluator/mutator/strategies | ✅ Done | Evaluator/mutator/strategies honor custom objective (Nov 13) |
| 🟡 | Regression test using alternate objective (e.g., “f1”) | ✅ Done | Live regression scenario + unit test cover non-quality objective (Nov 14) |

---

## C. Archive & Parent Selection

| Priority | Task | Status | Notes |
|----------|------|--------|-------|
| 🔴 | Fix `archive.select_for_generation` call sites (`_select_batch`, `_maybe_migrate`) | ✅ Done | Exploit/explore restored, migration safe (Nov 13) |
| 🔴 | Keep mutation parents mixed until higher rung quorum | ✅ Done | Parent gate relaxed until ≥3 high-rung parents (Nov 13) |
| 🟠 | Soften parent shard gate | ✅ Done | Effective min shard now delayed until quorum of higher-rung parents (Nov 13) |
| 🟡 | Remove “warmup” parent hack | ✅ Done | Removed temp parent injection (Nov 13) |

---

## D. Mutation Throughput & Queue Health

| Priority | Task | Status | Notes |
|----------|------|--------|-------|
| 🔴 | Base mutation budget on inflight capacity / recent promotion rate | ✅ Done | Budget now watches inflight slots and queue cap (Nov 13) |
| 🟠 | Decouple mutation enqueue buffer from eval backlog | ⬜ TODO | Allow streamed mutations to land in a bounded buffer without priority-queue saturation |
| 🟠 | Increase `max_final_shard_inflight` dynamically + auto straggler replays | ⬜ TODO | Keep final rung saturated without backlog |
| 🟠 | Bandit operator throttle for mutation spending | ✅ Done | Operator budgets now follow reward-weighted allocation (Nov 14) |

---

## E. Stop Governor Accuracy

| Priority | Task | Status | Notes |
|----------|------|--------|-------|
| 🔴 | Pass per-epoch evaluation delta to stop governor | ✅ Done | Gov now uses eval delta instead of cumulative total (Nov 13) |
| 🔴 | Track incremental token spend per epoch | ✅ Done | Tokens aggregated per evaluation before hypervolume update (Nov 13) |

---

## F. Multi-Island Reliability

| Priority | Task | Status | Notes |
|----------|------|--------|-------|
| 🟠 | Ensure `_maybe_migrate` works with new archive API | ✅ Done | Verified via multi-island regression scenario (Nov 14) |

---

## G. Persistence & Memory Hygiene

| Priority | Task | Status | Notes |
|----------|------|--------|-------|
| 🟡 | Restore Pareto from cached evals (no forced full re-run) | ✅ Done | Archive rehydrates from stored EvalResults, no re-eval required (Nov 13) |
| 🟡 | Trim `latest_results` with an LRU cap | ✅ Done | OrderedDict-based cap prevents leak during long runs (Nov 13) |

---

## Next Steps

1. Implement bandit-driven operator budget adjustments (DONE Nov 14).
2. Decouple mutation enqueue buffer from eval backlog so streaming never stalls when queue is deep.
3. Add CI hooks for the evolution regression matrix to guard against performance regressions.
4. Expand multi-objective testing (latency, neg_cost) beyond the current neg-cost scenario.

Keep this doc updated as tasks close. Use ✅/⬜ to track progress. For each change, capture before/after metrics (promotion rates, queue size, time-to-target) in the benchmark log.
