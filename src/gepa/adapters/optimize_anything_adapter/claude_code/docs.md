# Claude Code Adapter Guide

This directory implements the Claude Code specific mutation path for `optimize_anything`.

It is not a general-purpose Claude integration layer. Its job is narrower:

1. Run Claude Code as the proposer LLM inside GEPA's optimization loop.
2. Materialize evaluation history on disk so Claude Code can inspect it with filesystem tools.
3. Maintain a persistent reflection journal (`note.md`) across iterations.
4. Turn Claude Code's final fenced code block back into the next candidate string.

If you want to understand where this fits, start from [`gepa.optimize_anything.optimize_anything`](../../../optimize_anything.py), not from these files in isolation.

## What this mode is

Normal `optimize_anything` reflection works by:

- evaluating a candidate,
- building an in-memory reflective dataset,
- prompting an LLM directly with that dataset,
- getting back new text for one or more candidate fields.

This Claude Code mode does something different:

- it records evaluation artifacts to a run directory,
- launches `claude -p` as an external subprocess,
- gives Claude access to that run directory via `--add-dir`,
- tells Claude to inspect eval files itself,
- asks Claude to return the improved candidate in a fenced code block.

So the core design choice is:

- default reflective mutation: GEPA curates the reflection context for the model,
- Claude Code mutation: GEPA gives the model tools, files, and a journal, then lets it do its own investigation.

## Files in this directory

### `agent.py`

This is the main implementation. It contains:

- `AgenticProposer`: the proposer used by the GEPA engine in Claude Code mode.
- `EvalRecorderCallback`: writes evaluation artifacts to disk under `run_dir/evals/`.
- `GlobalNote`: append-only thread-safe wrapper around `note.md`.
- `NoteUpdater`: uses a second Claude call to summarize each iteration into the global note.
- `build_mutation_prompt()`: creates the prompt that tells Claude where to look and what to improve.
- small helpers for fenced-block extraction and JSON serialization.

### `runtime.py`

This wraps the Claude CLI:

- builds the subprocess environment,
- runs `claude -p --output-format stream-json`,
- parses JSONL output,
- extracts assistant text from the streamed event format,
- returns a plain `LanguageModel` callable compatible with GEPA.

### `prompts.py`

This stores the system prompts and the note-update prompt builder:

- `CC_RLM_SYSTEM_PROMPT`: teaches Claude how to spawn recursive sub-agents with `python -m rlm.cli`.
- `CC_AGENTIC_MUTATION_SYSTEM_PROMPT`: system prompt for the mutation proposer.
- `build_note_update_prompt()`: prompt for writing one new journal entry after each mutation.

### `__init__.py`

Just re-exports the public API of this subpackage.

## How it gets activated

The entry point is in [`gepa.optimize_anything.optimize_anything`](../../../optimize_anything.py).

Claude Code mode is enabled when:

- `config.reflection.reflection_lm == "claude_code"`

During setup, `optimize_anything()`:

1. converts `"claude_code"` into a callable via `make_claude_code_lm()`,
2. adds `EvalRecorderCallback` so evaluations are persisted to disk,
3. creates two Claude LM callables:
   - one for proposing mutations,
   - one for updating `note.md`,
4. creates `GlobalNote` at `run_dir/note.md`,
5. constructs `AgenticProposer`,
6. passes that proposer into the normal GEPA engine.

This means the GEPA engine still drives the overall optimization loop. Only the proposal mechanism changes.

## High-level control flow

The core method is `AgenticProposer.propose(state)`.

A single iteration looks like this:

1. Pick a parent candidate.
2. Pick a minibatch of training examples.
3. Evaluate the parent on that minibatch.
4. Write callback artifacts for the parent evaluation.
5. Build a prompt pointing Claude to `run_dir/evals/cXXXXX/tasks/`.
6. Run Claude Code on that prompt.
7. Extract the last fenced code block from Claude's reply.
8. Replace the candidate text with that new string.
9. Evaluate the child candidate on the same minibatch.
10. Update `note.md` with a short learned lesson from the parent/child comparison.
11. Return a `CandidateProposal` to the engine.

The engine then decides whether that proposal is accepted into the frontier, just like any other GEPA proposer.

## The main object: `AgenticProposer`

`AgenticProposer` is intentionally close to `ReflectiveMutationProposer` in `src/gepa/proposer/reflective_mutation/reflective_mutation.py`, but with a different reflection path.

It receives:

- the `trainset`,
- the active `OptimizeAnythingAdapter`,
- a Claude-backed `LanguageModel` callable (`cc_lm`),
- `global_note` and `note_updater`,
- objective/background strings,
- the run directory,
- selection/sampling strategy objects,
- optional perfect-score skipping config,
- callbacks/logger/tracking/rng.

### What it assumes about candidates

This is an important constraint:

- Claude Code mode only supports a candidate dictionary with exactly one text field.

That is enforced here:

- if `len(curr_prog) != 1`, it raises `ValueError`.

The code prefers the key `"current_candidate"` if present; otherwise it uses the single existing key.

Why this restriction exists:

- the prompt asks Claude to rewrite one text artifact,
- the output parser extracts one fenced block,
- there is no per-field structured proposal format in this mode.

If you need multi-field candidates, this proposer has to be redesigned rather than lightly patched.

## Candidate selection modes

There are two selection paths.

### Standard mode

Used when `task_aligned == False`.

- parent candidate comes from `candidate_selector.select_candidate_idx(state)`,
- minibatch comes from `batch_sampler.next_minibatch_ids(...)`,
- no reference tasks are provided.

### Task-aligned mode

Used when:

- not in single-instance mode,
- and `valset is None`.

In this case `_task_aligned_select()`:

1. samples one task id from `state.program_at_pareto_front_valset`,
2. samples one parent candidate from the Pareto front for that task,
3. samples a few other task ids as reference tasks.

The mutation prompt then says:

- improve the candidate specifically for the sampled target task,
- optionally inspect reference-task eval files for transferable ideas.

This is the key multi-task behavior. Instead of optimizing only by global average, the proposer is sometimes pointed at a task-specific frontier survivor.

## Evaluation flow inside `propose()`

There are two evaluations per mutation attempt.

### 1. Parent evaluation

The selected parent candidate is evaluated on the chosen minibatch with:

- `capture_traces=False`

That is a meaningful departure from the standard reflective proposer, which often evaluates with traces so it can build a reflective dataset. In Claude Code mode, the model is expected to inspect persisted eval files instead of receiving in-memory trajectories.

After this evaluation:

- `state.increment_evals(...)` is called,
- scores are written into `state.full_program_trace[-1]`,
- `EvaluationEndEvent` is emitted.

### 2. Child evaluation

After Claude proposes a new candidate string, the child is evaluated on the same minibatch using:

- `state.cached_evaluate_full(...)`

That matters because repeated candidate/example pairs can reuse GEPA's evaluation cache.

The proposer builds:

- `outputs_by_id`,
- `scores_by_id`,
- `objective_by_id`,
- `actual_evals`

and only increments the evaluation budget by `actual_evals`, not by minibatch size unconditionally.

## Disk layout and why it matters

Claude Code does not receive raw eval objects directly. It reads files from disk. That makes `EvalRecorderCallback` central to this design.

It writes under:

```text
{run_dir}/evals/
  c00000/
    skill.md
    meta.json
    tasks/
      task_00007_eval_00000.json
      task_00012_eval_00000.json
  c00001/
    ...
  proposed_00000/
    ...
```

### Candidate directories

For candidates already in GEPA state:

- directory name is `c{candidate_idx:05d}`

For not-yet-accepted proposals evaluated during mutation:

- directory name is `proposed_{n:05d}`

### `skill.md`

Written once per candidate directory by `_write_skill_md()`.

It serializes the candidate dict as sections like:

```md
## current_candidate

...candidate text...
```

This gives Claude a human-readable snapshot of the candidate associated with those evals.

### `meta.json`

Contains:

- `candidate_idx`
- `parents`
- `iteration`
- `average_score`

This is lightweight metadata, mostly for inspection and tooling.

### Task eval JSON files

One file is written per `(task_id, eval_id)` pair.

Each JSON contains:

- `eval_id`
- `candidate_idx`
- `task_id`
- `score`
- `iteration`
- optional `side_info`

The `side_info` field comes from `_extract_side_info(output)`, which only pulls data if the evaluator output is a tuple of length at least 3 and the third element is a dict.

This is another critical contract:

- if your evaluator returns useful debugging information in `side_info`, Claude can inspect it from these files,
- if not, Claude mostly sees scalar scores and candidate text.

The better your evaluator's `side_info`, the more useful this mode becomes.

## Prompting strategy

There are two kinds of prompts here.

### 1. Mutation prompt

Built by `build_mutation_prompt()`.

Its structure is roughly:

- objective
- background
- accumulated reflection notes
- path to the parent's eval files
- optional target task id and reference task ids
- parent candidate text
- concrete instruction to read eval history and output an improved candidate

A notable design decision:

- the prompt does not inline eval contents,
- it points Claude at a directory and relies on tool use plus `rlm.cli` for exploration.

This keeps prompt size stable even when eval history grows.

### 2. Note update prompt

Built by `build_note_update_prompt()`.

It compares:

- parent average score,
- child average score,
- parent/child candidate text,
- truncated parent/child outputs,
- objective.

It asks Claude to append one short entry beginning with `## Iteration N`.

This creates a persistent learned-memory document across the run.

## `note.md`: long-term memory for the proposer

`GlobalNote` is intentionally simple:

- `read()` returns the current note contents,
- `append()` appends text under a lock.

`NoteUpdater` calls a second Claude model after each mutation attempt and appends the result.

This note is then fed back into future mutation prompts through the `## Reflection notes` section.

Conceptually:

- eval files are local episodic evidence,
- `note.md` is compressed cross-iteration memory.

The code is append-only. It never rewrites or summarizes old note entries. That keeps implementation simple, but it also means `note.md` can grow large over long runs.

## Claude CLI wrapper: `runtime.py`

`make_claude_code_lm()` returns a normal GEPA `LanguageModel` callable, but internally it runs a subprocess:

```bash
claude -p --output-format stream-json --dangerously-skip-permissions
```

It optionally adds:

- `--model ...`
- `--max-budget-usd ...`
- `--append-system-prompt ...`
- `--add-dir ...`

### Why `stream-json`

Claude Code streams structured JSON events instead of only returning plain text. The runtime:

1. collects JSONL events,
2. stores them in `events`,
3. pulls `assistant` text blocks when available,
4. falls back to the `result` event text otherwise.

That logic lives in:

- `parse_stream_json()`
- `extract_text_from_stream_json()`

### Environment sanitization

`build_claude_code_env()` removes a few parent Claude-session variables:

- `CLAUDECODE`
- `CLAUDE_CODE_SSE_PORT`
- `CLAUDE_CODE_ENTRYPOINT`

Then it sets:

- `RLM_DEPTH=0`
- `RLM_BUDGET_FILE=/tmp/rlm_budget_<pid>.txt`
- `PYTHONPATH` to include the repo source tree

This matters because nested Claude Code subprocesses reject some inherited session variables, and the RLM tooling needs a clean environment.

## Recursive sub-agents: `CC_RLM_SYSTEM_PROMPT`

The system prompt teaches Claude how to use:

```bash
python -m rlm.cli
```

for cheaper recursive reads and summaries.

The intended pattern is:

- use cheaper Haiku sub-agents to inspect many eval files,
- use the main Sonnet process to synthesize and write the final mutation.

This is not enforced in Python code. It is a prompt-level capability. So if behavior degrades, one place to inspect is prompt quality, not just control flow.

## Output parsing and its fragility

After Claude replies, `AgenticProposer` calls `_extract_last_fenced_block(raw_reply)`.

This helper:

- finds the last fenced code block in the reply,
- strips a language tag if present,
- returns the block contents,
- falls back to raw trimmed text in some malformed cases.

This is one of the more brittle parts of the design.

The proposer assumes:

- the final useful answer is the last fenced block,
- that block contains the full new candidate text,
- nothing after the closing fence should matter.

That assumption is reinforced by `CC_AGENTIC_MUTATION_SYSTEM_PROMPT`, which explicitly says:

- write the improved candidate in a single fenced code block at the end,
- do not write anything after the closing fence.

If Claude stops following that format, mutation quality can silently degrade.

## Callback integration

This subpackage uses the same callback system as the rest of GEPA.

`AgenticProposer.propose()` emits:

- `on_candidate_selected`
- `on_minibatch_sampled`
- `on_evaluation_start`
- `on_evaluation_end`
- `on_evaluation_skipped`
- `on_proposal_start`
- `on_proposal_end`

`EvalRecorderCallback` listens to `on_evaluation_end` and writes the artifact files that future Claude calls depend on.

This means the artifact-writing mechanism is observational and decoupled:

- proposer emits events,
- recorder persists them,
- future proposer calls read the resulting files indirectly through Claude.

## How this interacts with `OptimizeAnythingAdapter`

The active adapter is still `OptimizeAnythingAdapter` in [`../optimize_anything_adapter.py`](../optimize_anything_adapter.py).

Its role here is unchanged:

- call the user evaluator,
- manage caching,
- package results as scores/outputs/objective scores.

For Claude Code mode, the most important adapter behavior is that `evaluate()` returns outputs whose third tuple element may contain structured `side_info`. That `side_info` gets persisted into the task JSON files and becomes the main diagnostic material Claude can inspect.

So if you are debugging weak mutations, inspect the evaluator contract first:

- Are scores meaningful?
- Is `side_info` actually populated?
- Is the persisted JSON enough for an external agent to diagnose failures?

## Key invariants and design constraints

These are the main assumptions that are easy to violate by accident.

### 1. Single-field candidates only

`AgenticProposer` rejects multi-key candidate dicts.

### 2. A run directory is required

This mode depends on:

- `run_dir/evals/...`
- `run_dir/note.md`

Without a stable run directory, Claude would have nothing to inspect.

### 3. Eval artifacts must be readable by Claude

`make_claude_code_lm(..., add_dirs=[run_dir])` is what grants Claude access to the run directory. If that disappears, the prompts still mention paths, but Claude cannot inspect them.

### 4. Mutation output must end in a fenced block

If the model does not return the candidate in the expected format, parsing becomes unreliable.

### 5. Useful `side_info` strongly affects performance

This architecture is only as good as the evaluator diagnostics written to disk.

### 6. `note.md` growth is unbounded

No pruning or summarization is implemented.

### 7. Note update failures are non-fatal

`NoteUpdater.update()` is wrapped in `try/except` and only logs a warning. Optimization continues even if journal updates fail.

## Important differences from standard reflective mutation

Compared with `ReflectiveMutationProposer`, this mode:

- does not build a reflective dataset,
- does not rely on captured trajectories,
- does not propose per-field updates,
- does not need prompt templates validated by `InstructionProposalSignature`,
- does rely on filesystem artifacts and external tool use,
- does add a persistent journal between iterations.

That makes it more agentic and more flexible, but also more operationally fragile because good behavior depends on CLI execution, prompt compliance, file layout, and external tool access.

## Failure modes to check first

If this code behaves poorly, check these in order.

### Claude subprocess failures

Look at `make_claude_code_lm()` first:

- missing `claude` binary,
- timeout,
- non-zero exit code,
- env issues with nested Claude invocations.

### Empty or malformed model output

Look at:

- `extract_text_from_stream_json()`
- `_extract_last_fenced_block()`

### Claude not using the eval history well

Check:

- was `EvalRecorderCallback` attached?
- are files actually being written under `run_dir/evals/`?
- is `side_info` present in task JSON?
- is `--add-dir` pointing at the right run dir?

### Weak long-term learning across iterations

Check:

- whether `note.md` exists,
- whether entries are appended after each iteration,
- whether note updates are failing and getting swallowed by the warning path.

## Safe extension points

If you want to improve this module without changing its overall architecture, the safest places are:

1. enrich the task JSON schema written by `EvalRecorderCallback`,
2. improve `CC_AGENTIC_MUTATION_SYSTEM_PROMPT`,
3. improve `build_mutation_prompt()` with better instructions or file pointers,
4. add summarization/pruning for `note.md`,
5. make output parsing stricter or structured,
6. support richer candidate serialization than a single fenced block.

## Risky changes

These require more careful redesign than they may look.

1. Supporting multi-field candidates.
2. Removing the callback-based eval recorder.
3. Switching away from file-based reflection without changing prompts.
4. Changing evaluator output shape without preserving `side_info`.
5. Relying on stdout from `rlm.cli` instead of file outputs.

The prompts explicitly assume the current operational model.

## Mental model to keep while reading the code

This subpackage is best understood as a thin orchestration layer around four moving parts:

- GEPA chooses what candidate/task slice to work on.
- `OptimizeAnythingAdapter` evaluates candidates and returns diagnostics.
- `EvalRecorderCallback` turns those diagnostics into filesystem artifacts.
- Claude Code reads those artifacts, proposes a new candidate, and writes compressed lessons into `note.md`.

So the architecture is not "LLM proposes from a prompt". It is closer to:

- "GEPA runs an external coding agent over a persistent scratchpad of evaluation evidence."

That framing makes most implementation choices here make sense.
