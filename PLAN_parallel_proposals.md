# Plan: Revert threaded parallel proposals + implement batch-based parallel proposals

## Context

PR #314 added parallel proposals using `ThreadPoolExecutor` in the engine. The vision is to remove threading from GEPA's core and instead use batch calls — delegating parallelism to adapters (`batch_evaluate`) and LMs (`batch_complete`). This makes the engine sequential and async-ready. Also incorporates PR #279's `batch_evaluator` concept via a `batch_evaluate` method on the adapter protocol.

Branch: `refactor/batch-parallel-proposals` (already created off main)

## Commit 1: Revert #314's threading

### `src/gepa/proposer/reflective_mutation/reflective_mutation.py`

**Remove entirely:**
- `import threading` (line 4)
- `from dataclasses import dataclass, field` (line 6)
- `ProposalContext` dataclass (lines 34-49)
- `ProposalOutput` dataclass (lines 52-63)
- `self._lock = threading.Lock()` (line 102)
- `prepare_proposal()` method (lines 182-237)
- `execute_proposal()` method (lines 239-468)
- `apply_proposal_output()` method (lines 470-475)
- `propose_output()` method (lines 477-488)

**Restore `propose()` as a single method** that does inline what `execute_proposal` + `apply_proposal_output` did:

```python
def propose(self, state: GEPAState) -> CandidateProposal | None:
    i = state.i + 1

    # 1. Select candidate
    curr_prog_id = self.candidate_selector.select_candidate_idx(state)
    curr_prog = state.program_candidates[curr_prog_id]
    state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
    self.logger.log(f"Iteration {i}: Selected program {curr_prog_id} score: {state.program_full_scores_val_set[curr_prog_id]}")

    # Fire on_candidate_selected callback
    notify_callbacks(self.callbacks, "on_candidate_selected", CandidateSelectedEvent(
        iteration=i, candidate_idx=curr_prog_id, candidate=curr_prog,
        score=state.program_full_scores_val_set[curr_prog_id],
    ))

    self.experiment_tracker.log_metrics(
        {"iteration": i, "selected_program_candidate": curr_prog_id, "total_metric_calls": state.total_num_evals},
        step=i,
    )

    # 2. Sample minibatch
    subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
    state.full_program_trace[-1]["subsample_ids"] = subsample_ids
    minibatch = self.trainset.fetch(subsample_ids)

    # Fire on_minibatch_sampled callback
    notify_callbacks(self.callbacks, "on_minibatch_sampled", MinibatchSampledEvent(
        iteration=i, minibatch_ids=subsample_ids, trainset_size=len(self.trainset),
    ))

    curr_parent_ids = [p for p in state.parent_program_for_candidate[curr_prog_id] if p is not None]
    is_seed_candidate = curr_prog_id == 0

    # 3. Evaluate current program with traces
    notify_callbacks(self.callbacks, "on_evaluation_start", EvaluationStartEvent(
        iteration=i, candidate_idx=curr_prog_id, batch_size=len(minibatch),
        capture_traces=True, parent_ids=curr_parent_ids, inputs=minibatch,
        is_seed_candidate=is_seed_candidate,
    ))
    eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
    state.increment_evals(eval_curr.num_metric_calls if eval_curr.num_metric_calls is not None else len(subsample_ids))
    state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores
    notify_callbacks(self.callbacks, "on_evaluation_end", EvaluationEndEvent(
        iteration=i, candidate_idx=curr_prog_id, scores=eval_curr.scores,
        has_trajectories=bool(eval_curr.trajectories), parent_ids=curr_parent_ids,
        outputs=eval_curr.outputs, trajectories=eval_curr.trajectories,
        objective_scores=eval_curr.objective_scores, is_seed_candidate=is_seed_candidate,
    ))

    # Update evaluation cache
    if state.evaluation_cache is not None:
        objective_scores_list = list(eval_curr.objective_scores) if eval_curr.objective_scores else None
        state.evaluation_cache.put_batch(curr_prog, subsample_ids, eval_curr.outputs, eval_curr.scores, objective_scores_list)

    # Check trajectories
    if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
        self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
        notify_callbacks(self.callbacks, "on_evaluation_skipped", EvaluationSkippedEvent(
            iteration=i, candidate_idx=curr_prog_id, reason="no_trajectories",
            scores=eval_curr.scores, is_seed_candidate=is_seed_candidate,
        ))
        return None

    # Check perfect scores
    if (self.skip_perfect_score and self.perfect_score is not None
            and all(s is not None and s >= self.perfect_score for s in eval_curr.scores)):
        self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
        notify_callbacks(self.callbacks, "on_evaluation_skipped", EvaluationSkippedEvent(
            iteration=i, candidate_idx=curr_prog_id, reason="all_scores_perfect",
            scores=eval_curr.scores, is_seed_candidate=is_seed_candidate,
        ))
        return None

    subsample_before = sum(eval_curr.scores)

    # 4. Select components and reflect
    predictor_names_to_update = self.module_selector(
        state, eval_curr.trajectories, eval_curr.scores, curr_prog_id, curr_prog
    )

    # 5. Build reflective dataset and propose
    try:
        reflective_dataset = self.adapter.make_reflective_dataset(curr_prog, eval_curr, predictor_names_to_update)
        reflective_dataset_concrete = {k: [dict(item) for item in v] for k, v in reflective_dataset.items()}

        notify_callbacks(self.callbacks, "on_reflective_dataset_built", ReflectiveDatasetBuiltEvent(
            iteration=i, candidate_idx=curr_prog_id, components=predictor_names_to_update,
            dataset=reflective_dataset_concrete,
        ))
        notify_callbacks(self.callbacks, "on_proposal_start", ProposalStartEvent(
            iteration=i, parent_candidate=curr_prog, components=predictor_names_to_update,
            reflective_dataset=reflective_dataset_concrete,
        ))

        new_texts, prompts, raw_lm_outputs = self.propose_new_texts(
            curr_prog, reflective_dataset, predictor_names_to_update
        )

        notify_callbacks(self.callbacks, "on_proposal_end", ProposalEndEvent(
            iteration=i, new_instructions=new_texts, prompts=prompts, raw_lm_outputs=raw_lm_outputs,
        ))

        _lm_metadata: dict[str, Any] = {}
        for comp in new_texts:
            _lm_metadata[f"prompt:{comp}"] = prompts.get(comp, "")
            _lm_metadata[f"raw_lm_output:{comp}"] = raw_lm_outputs.get(comp, "")

        for pname, text in new_texts.items():
            self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")
    except Exception as e:
        self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
        import traceback
        self.logger.log(traceback.format_exc())
        return None

    # 6. Create new candidate and evaluate
    new_candidate = curr_prog.copy()
    for pname, text in new_texts.items():
        assert pname in new_candidate, f"{pname} missing in candidate"
        new_candidate[pname] = text

    notify_callbacks(self.callbacks, "on_evaluation_start", EvaluationStartEvent(
        iteration=i, candidate_idx=None, batch_size=len(minibatch),
        capture_traces=True, parent_ids=[curr_prog_id], inputs=minibatch,
        is_seed_candidate=False,
    ))

    eval_new = self.adapter.evaluate(minibatch, new_candidate, capture_traces=True)
    new_scores = eval_new.scores
    outputs = eval_new.outputs

    # Populate evaluation cache with new candidate results
    if state.evaluation_cache is not None:
        new_objective_scores_list = list(eval_new.objective_scores) if eval_new.objective_scores else None
        state.evaluation_cache.put_batch(new_candidate, subsample_ids, eval_new.outputs, eval_new.scores, new_objective_scores_list)

    notify_callbacks(self.callbacks, "on_evaluation_end", EvaluationEndEvent(
        iteration=i, candidate_idx=None, scores=new_scores,
        has_trajectories=bool(eval_new.trajectories), parent_ids=[curr_prog_id],
        outputs=outputs, trajectories=eval_new.trajectories,
        objective_scores=list(eval_new.objective_scores) if eval_new.objective_scores else None,
        is_seed_candidate=False,
    ))

    state.increment_evals(eval_new.num_metric_calls if eval_new.num_metric_calls is not None else len(subsample_ids))
    state.full_program_trace[-1]["new_subsample_scores"] = new_scores

    new_sum = sum(new_scores)
    self.experiment_tracker.log_metrics(
        {"subsample/before": subsample_before, "subsample/after": new_sum, "total_metric_calls": state.total_num_evals},
        step=i,
    )

    return CandidateProposal(
        candidate=new_candidate,
        parent_program_ids=[curr_prog_id],
        subsample_indices=subsample_ids,
        subsample_scores_before=eval_curr.scores,
        subsample_scores_after=new_scores,
        eval_before=SubsampleEvaluation(
            scores=eval_curr.scores, outputs=eval_curr.outputs,
            objective_scores=list(eval_curr.objective_scores) if eval_curr.objective_scores else None,
            trajectories=eval_curr.trajectories,
        ),
        eval_after=SubsampleEvaluation(
            scores=new_scores, outputs=outputs,
            objective_scores=list(eval_new.objective_scores) if eval_new.objective_scores else None,
            trajectories=eval_new.trajectories,
        ),
        tag="reflective_mutation",
        metadata=_lm_metadata,
    )
```

**Keep `propose_new_texts()` completely unchanged** (lines 120-180).

### `src/gepa/core/engine.py`

**Remove:**
- `from concurrent.futures import ThreadPoolExecutor, as_completed` (line 7)
- `from gepa.proposer.base import CandidateProposal` (line 33)
- `ProposalOutput` from imports (line 38)
- `num_parallel_proposals: int = 1` param (line 85) and `self.num_parallel_proposals` (line 129)
- `_process_proposal_output()` method (lines 350-375)
- `_run_parallel_reflective_batch()` method (lines 381-452)

**Replace lines 742-749** with:
```python
# 2) Reflective mutation proposer
proposal = self.reflective_proposer.propose(state)
if proposal is None:
    self.logger.log(f"Iteration {state.i + 1}: Reflective mutation did not propose a new candidate")
    continue

proposal_accepted = self._accept_reflective_proposal(proposal, state.i + 1, state)

if proposal_accepted and self.merge_proposer is not None:
    self.merge_proposer.last_iter_found_new_program = True
    if self.merge_proposer.total_merges_tested < self.merge_proposer.max_merge_invocations:
        self.merge_proposer.merges_due += 1
```

**Keep `_accept_reflective_proposal()`** (lines 287-348) — clean extraction.

### `src/gepa/optimize_anything.py`

**Remove:**
- `num_parallel_proposals: int | Literal["auto"] = 1` from EngineConfig (lines 487-495)
- `_resolve_num_parallel_proposals()` function (lines 1098-1106)
- `num_parallel_proposals=_resolve_num_parallel_proposals(...)` from engine construction (lines 1623-1627)

**Keep:** MLflow thread-safety in `experiment_tracker.py` (untouched)

---

## Commit 2: Add `batch_evaluate` to GEPAAdapter + wire PR #279's batch_evaluator

### `src/gepa/core/adapter.py`

Add standalone helper + document on protocol:

```python
def default_batch_evaluate(
    adapter: "GEPAAdapter",
    items: list[tuple[dict[str, str], list]],
    capture_traces: bool = False,
) -> list["EvaluationBatch"]:
    """Default sequential batch_evaluate implementation.
    
    Calls adapter.evaluate() once per (candidate, batch) pair.
    Adapters can implement batch_evaluate() directly for true batching.
    """
    return [adapter.evaluate(batch, candidate, capture_traces=capture_traces)
            for candidate, batch in items]
```

Add to `GEPAAdapter` protocol docstring:
```
4) Optional batch evaluation (batch_evaluate):
   Adapters that support evaluating multiple (candidate, batch) pairs
   in a single call can implement:
   
   batch_evaluate(self, items: list[tuple[dict[str, str], list[DataInst]]], 
                  capture_traces: bool = False) -> list[EvaluationBatch]
   
   The engine uses this for parallel proposals. When not implemented,
   falls back to sequential evaluate() calls via default_batch_evaluate().
```

### `src/gepa/adapters/optimize_anything_adapter/optimize_anything_adapter.py`

Add `batch_evaluator: Callable[..., Any] | None = None` to `__init__`.
Store as `self.batch_evaluator`.

Implement `batch_evaluate()`:
```python
def batch_evaluate(self, items, capture_traces=False):
    if self.batch_evaluator is not None:
        return self._batch_evaluate_via_user_fn(items, capture_traces)
    return [self.evaluate(batch, candidate, capture_traces=capture_traces)
            for candidate, batch in items]

def _batch_evaluate_via_user_fn(self, items, capture_traces):
    """Flatten all (candidate, example) pairs, call batch_evaluator once, repackage."""
    # Build pairs list
    pairs = []
    pair_to_item_idx = []
    for item_idx, (candidate, batch) in enumerate(items):
        for example in batch:
            str_candidate = next(iter(candidate.values())) if self._str_candidate_mode else candidate
            pairs.append((str_candidate, example))
            pair_to_item_idx.append(item_idx)
    
    # Call batch_evaluator
    raw_results = self.batch_evaluator(pairs)
    if len(raw_results) != len(pairs):
        raise ValueError(f"batch_evaluator returned {len(raw_results)} results but expected {len(pairs)}")
    
    # Normalize results and group by item
    # ... (normalize each result to (score, output, side_info) tuple)
    # ... (build EvaluationBatch per item)
```

### `src/gepa/optimize_anything.py`

Add `batch_evaluator: Callable[..., Any] | None = None` to `optimize_anything()` signature.
Pass to `OptimizeAnythingAdapter` constructor.
Add validation: at least one of `evaluator` or `batch_evaluator` must be provided.

---

## Commit 3: Add batch-based parallel proposals

### New file: `src/gepa/proposer/parallel.py`

```python
"""Batch-based parallel proposals for GEPA.

No ThreadPoolExecutor — all parallelism delegated to adapter.batch_evaluate()
and LM.batch_complete().
"""

from dataclasses import dataclass
from typing import Any, Protocol

from gepa.core.adapter import EvaluationBatch, default_batch_evaluate
from gepa.core.data_loader import DataId, DataLoader
from gepa.core.state import GEPAState
from gepa.proposer.base import CandidateProposal, SubsampleEvaluation
from gepa.proposer.reflective_mutation.base import CandidateSelector
from gepa.strategies.acceptance import AcceptanceCriterion
from gepa.strategies.batch_sampler import BatchSampler


@dataclass
class ProposalTask:
    """One (parent, minibatch) pair to propose from."""
    parent_idx: int
    parent_candidate: dict[str, str]
    minibatch_ids: list
    minibatch: list


@dataclass
class ParallelConfig:
    """Configuration for batch-based parallel proposals."""
    sampling_strategy: "ParallelSamplingStrategy"
    selection_strategy: "ProposalSelectionStrategy"


class ParallelSamplingStrategy(Protocol):
    def sample_tasks(
        self, state: GEPAState, candidate_selector: CandidateSelector,
        batch_sampler: BatchSampler, trainset: DataLoader,
    ) -> list[ProposalTask]: ...


class ProposalSelectionStrategy(Protocol):
    def select(
        self, proposals: list[CandidateProposal], state: GEPAState,
        acceptance_criterion: AcceptanceCriterion,
    ) -> list[CandidateProposal]: ...


# --- Built-in sampling strategies ---

class SingleMutationSampling:
    """1 parent, 1 mutation — default GEPA behavior."""
    def sample_tasks(self, state, candidate_selector, batch_sampler, trainset):
        parent_idx = candidate_selector.select_candidate_idx(state)
        mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
        return [ProposalTask(parent_idx, state.program_candidates[parent_idx], mb_ids, trainset.fetch(mb_ids))]


class SameParentSampling:
    """Best-of-N on the same parent with different minibatches."""
    def __init__(self, n: int):
        self.n = n
    def sample_tasks(self, state, candidate_selector, batch_sampler, trainset):
        parent_idx = candidate_selector.select_candidate_idx(state)
        parent = state.program_candidates[parent_idx]
        return [ProposalTask(parent_idx, parent, (ids := batch_sampler.next_minibatch_ids(trainset, state)), trainset.fetch(ids)) for _ in range(self.n)]


class IndependentSampling:
    """N different parents, each with their own minibatch."""
    def __init__(self, n: int):
        self.n = n
    def sample_tasks(self, state, candidate_selector, batch_sampler, trainset):
        tasks = []
        for _ in range(self.n):
            parent_idx = candidate_selector.select_candidate_idx(state)
            mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
            tasks.append(ProposalTask(parent_idx, state.program_candidates[parent_idx], mb_ids, trainset.fetch(mb_ids)))
        return tasks


class PxNSampling:
    """P parents x N mutations each = P*N total tasks."""
    def __init__(self, p: int, n: int):
        self.p, self.n = p, n
    def sample_tasks(self, state, candidate_selector, batch_sampler, trainset):
        tasks = []
        for _ in range(self.p):
            parent_idx = candidate_selector.select_candidate_idx(state)
            parent = state.program_candidates[parent_idx]
            for _ in range(self.n):
                mb_ids = batch_sampler.next_minibatch_ids(trainset, state)
                tasks.append(ProposalTask(parent_idx, parent, mb_ids, trainset.fetch(mb_ids)))
        return tasks


# --- Built-in selection strategies ---

class AllImprovements:
    """Accept all proposals that pass the acceptance criterion."""
    def select(self, proposals, state, acceptance_criterion):
        return [p for p in proposals if acceptance_criterion.should_accept(p, state)]


class BestImprovement:
    """Accept only the single best-improving proposal."""
    def select(self, proposals, state, acceptance_criterion):
        best, best_imp = None, float('-inf')
        for p in proposals:
            if not acceptance_criterion.should_accept(p, state):
                continue
            imp = sum(p.subsample_scores_after or []) - sum(p.subsample_scores_before or [])
            if imp > best_imp:
                best, best_imp = p, imp
        return [best] if best else []


class TopKImprovements:
    """Accept top-K proposals by improvement margin."""
    def __init__(self, k: int):
        self.k = k
    def select(self, proposals, state, acceptance_criterion):
        passing = [(sum(p.subsample_scores_after or []) - sum(p.subsample_scores_before or []), p)
                    for p in proposals if acceptance_criterion.should_accept(p, state)]
        passing.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in passing[:self.k]]


# --- Core batch pipeline ---

def propose_batch(proposer, adapter, state, sampling_strategy, selection_strategy, 
                  acceptance_criterion, logger, experiment_tracker, callbacks=None):
    """4-stage batch pipeline. No ThreadPoolExecutor.
    
    Stage 1: Sample N tasks [sequential]
    Stage 2: Batch evaluate all parents with deduplication [adapter.batch_evaluate]
    Stage 3: Build reflective datasets + propose [sequential per task]
    Stage 4: Batch evaluate all children [adapter.batch_evaluate]
    Stage 5: Filter via selection_strategy
    """
    from gepa.core.adapter import default_batch_evaluate
    from gepa.core.state import _candidate_hash

    # Stage 1: Sample
    tasks = sampling_strategy.sample_tasks(state, proposer.candidate_selector, proposer.batch_sampler, proposer.trainset)
    if not tasks:
        return []

    # Stage 2: Batch evaluate parents (deduplicated)
    unique_keys = {}  # (hash, tuple(mb_ids)) -> (candidate, minibatch)
    task_to_key = []
    for task in tasks:
        key = (_candidate_hash(task.parent_candidate), tuple(task.minibatch_ids))
        unique_keys.setdefault(key, (task.parent_candidate, task.minibatch))
        task_to_key.append(key)

    key_list = list(unique_keys.keys())
    items = [unique_keys[k] for k in key_list]
    
    batch_evaluate_fn = getattr(adapter, 'batch_evaluate', None)
    if batch_evaluate_fn is not None:
        parent_evals = batch_evaluate_fn(items, capture_traces=True)
    else:
        parent_evals = default_batch_evaluate(adapter, items, capture_traces=True)

    key_to_eval = dict(zip(key_list, parent_evals))

    total_parent_evals = sum(
        (e.num_metric_calls if e.num_metric_calls is not None else len(items[i][1]))
        for i, e in enumerate(parent_evals)
    )
    state.increment_evals(total_parent_evals)

    # Stage 3: Reflect + propose (sequential per task)
    children = []  # list of (task, new_candidate, parent_eval, lm_metadata) or None
    for task, key in zip(tasks, task_to_key):
        eval_curr = key_to_eval[key]
        
        if not eval_curr.trajectories:
            children.append(None)
            continue
        
        if (proposer.skip_perfect_score and proposer.perfect_score is not None
                and all(s >= proposer.perfect_score for s in eval_curr.scores)):
            children.append(None)
            continue

        # Select components
        predictor_names = proposer.module_selector(
            state, eval_curr.trajectories, eval_curr.scores, task.parent_idx, task.parent_candidate
        )

        # Build reflective dataset and propose
        try:
            reflective_dataset = adapter.make_reflective_dataset(task.parent_candidate, eval_curr, predictor_names)
            new_texts, prompts, raw_outputs = proposer.propose_new_texts(
                task.parent_candidate, reflective_dataset, predictor_names
            )
            
            _lm_metadata = {}
            for comp in new_texts:
                _lm_metadata[f"prompt:{comp}"] = prompts.get(comp, "")
                _lm_metadata[f"raw_lm_output:{comp}"] = raw_outputs.get(comp, "")

            new_candidate = task.parent_candidate.copy()
            for name, text in new_texts.items():
                new_candidate[name] = text
            
            children.append((task, new_candidate, eval_curr, _lm_metadata))
        except Exception as e:
            logger.log(f"Parallel proposal failed for parent {task.parent_idx}: {e}")
            children.append(None)

    # Stage 4: Batch evaluate children
    valid_children = [(i, c) for i, c in enumerate(children) if c is not None]
    if not valid_children:
        return []

    child_items = [(c[1], c[0].minibatch) for _, c in valid_children]  # (new_candidate, minibatch)
    if batch_evaluate_fn is not None:
        child_evals = batch_evaluate_fn(child_items, capture_traces=True)
    else:
        child_evals = default_batch_evaluate(adapter, child_items, capture_traces=True)

    total_child_evals = sum(
        (e.num_metric_calls if e.num_metric_calls is not None else len(child_items[i][1]))
        for i, e in enumerate(child_evals)
    )
    state.increment_evals(total_child_evals)

    # Stage 5: Build proposals and filter
    proposals = []
    for (orig_idx, (task, new_candidate, eval_curr, _lm_metadata)), child_eval in zip(valid_children, child_evals):
        proposal = CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[task.parent_idx],
            subsample_indices=task.minibatch_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=child_eval.scores,
            eval_before=SubsampleEvaluation(
                scores=eval_curr.scores, outputs=eval_curr.outputs,
                objective_scores=list(eval_curr.objective_scores) if eval_curr.objective_scores else None,
                trajectories=eval_curr.trajectories,
            ),
            eval_after=SubsampleEvaluation(
                scores=child_eval.scores, outputs=child_eval.outputs,
                objective_scores=list(child_eval.objective_scores) if child_eval.objective_scores else None,
                trajectories=child_eval.trajectories,
            ),
            tag="parallel_mutation",
            metadata=_lm_metadata,
        )
        proposals.append(proposal)

    return selection_strategy.select(proposals, state, acceptance_criterion)
```

### `src/gepa/core/engine.py` (Commit 3 additions)

Add to `__init__`:
```python
parallel_config: "ParallelConfig | None" = None,
```
Store as `self.parallel_config`.

In main loop, replace the reflective mutation section:
```python
# 2) Reflective mutation proposer
if self.parallel_config is not None:
    from gepa.proposer.parallel import propose_batch
    accepted_proposals = propose_batch(
        proposer=self.reflective_proposer,
        adapter=self.adapter,
        state=state,
        sampling_strategy=self.parallel_config.sampling_strategy,
        selection_strategy=self.parallel_config.selection_strategy,
        acceptance_criterion=self.acceptance_criterion,
        logger=self.logger,
        experiment_tracker=self.experiment_tracker,
        callbacks=self.callbacks,
    )
    for prop in accepted_proposals:
        new_idx, _ = self._run_full_eval_and_add(
            new_program=prop.candidate, state=state,
            parent_program_idx=prop.parent_program_ids,
        )
        proposal_accepted = True
        self._log_proposal_lm_calls(state.i + 1, prop, candidate_idx=new_idx)
        notify_callbacks(self.callbacks, "on_candidate_accepted", CandidateAcceptedEvent(
            iteration=state.i + 1, new_candidate_idx=new_idx,
            new_score=sum(prop.subsample_scores_after or []),
            parent_ids=prop.parent_program_ids,
        ))
    if proposal_accepted and self.merge_proposer is not None:
        self.merge_proposer.last_iter_found_new_program = True
        if self.merge_proposer.total_merges_tested < self.merge_proposer.max_merge_invocations:
            self.merge_proposer.merges_due += 1
else:
    # Sequential path (default, unchanged)
    proposal = self.reflective_proposer.propose(state)
    if proposal is None:
        self.logger.log(...)
        continue
    proposal_accepted = self._accept_reflective_proposal(proposal, state.i + 1, state)
    if proposal_accepted and self.merge_proposer is not None:
        ...
```

### `src/gepa/api.py`

Add optional params:
```python
parallel_sampling_strategy: "ParallelSamplingStrategy | None" = None,
parallel_selection_strategy: "ProposalSelectionStrategy | None" = None,
```

Build ParallelConfig when provided:
```python
parallel_config = None
if parallel_sampling_strategy is not None:
    from gepa.proposer.parallel import ParallelConfig, AllImprovements
    parallel_config = ParallelConfig(
        sampling_strategy=parallel_sampling_strategy,
        selection_strategy=parallel_selection_strategy or AllImprovements(),
    )
```

Pass to engine constructor.

### `src/gepa/optimize_anything.py`

Add to EngineConfig:
```python
parallel_sampling_strategy: "ParallelSamplingStrategy | None" = None
parallel_selection_strategy: "ProposalSelectionStrategy | None" = None
```

Wire to engine.

### Tests: `tests/test_parallel_proposals.py`

Test classes:
- `TestSamplingStrategies` — unit tests for each sampling strategy
- `TestSelectionStrategies` — unit tests for each selection strategy  
- `TestProposeBatch` — integration tests with mock adapter
- `TestBatchEvaluate` — tests for default_batch_evaluate and adapter.batch_evaluate
- `TestEngineIntegration` — verify default behavior unchanged

## Verification

```bash
uv run pyright src/gepa/core/engine.py src/gepa/core/adapter.py src/gepa/proposer/parallel.py src/gepa/proposer/reflective_mutation/reflective_mutation.py src/gepa/api.py src/gepa/optimize_anything.py
uv run pytest tests/ -x -q
uv run ruff check src/gepa/
```
