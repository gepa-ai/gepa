# Terrarium

Common evaluation infrastructure for evolving/auto-research systems. Terrarium provides a standard way to define benchmark tasks and run any evolution system against them with controlled, measured evaluation budgets.

## Overview

```
    ┌──────────┐        ┌────────────────────────────────────────┐
    │  Runner  │───────►│           EvalServer                   │
    └──────────┘        │  (single budget choke point)           │
         │              │                                        │
         │              │  server.evaluate()  ◄── in-process     │
         ▼              │  POST /evaluate     ◄── HTTP (ext.)    │
    ┌──────────┐        │                                        │
    │ Adapter  │───────►│  BudgetTracker ── task.eval_fn         │
    └──────────┘        └────────────────────────────────────────┘
```

**Every adapter — in-process or external — goes through the same EvalServer.** The server owns the budget counter and the eval function. Adapters cannot bypass or modify it.

- **In-process adapters** (GEPA, custom) call `server.evaluate(candidate)` directly — no HTTP overhead.
- **External adapters** (Claude Code, any subprocess) POST to `server.url/evaluate` — budget enforced server-side.

## Quick Start

### 1. Run from the command line

```bash
python -m terrarium circle_packing my_adapter.py --max-evals 100
```

### 2. From Python

```python
from terrarium import run

result = run("circle_packing", "my_adapter.py", max_evals=100)
print(f"Best score: {result.best_score}")
print(f"Evals used: {result.total_evals}")
```

### 3. Pass an adapter object directly

```python
from terrarium import run
from terrarium.adapters.gepa import GEPAAdapter

adapter = GEPAAdapter(reflection_lm="openai/gpt-5")
result = run("circle_packing", adapter, max_evals=150)
```

## Available Tasks

| Task | Type | Candidate | Description |
|------|------|-----------|-------------|
| `circle_packing` | single-task | code | Pack 26 circles in a unit square, maximize sum of radii |
| `optuna_blackbox` | single-task | code | Evolve code to minimize a blackbox objective function |
| `aime_math` | dataset | prompt | Optimize a math-solving prompt across AIME problems |
| `arc_agi` | dataset | code | Optimize an ARC-AGI agent for abstract reasoning |

## How to Write an Adapter

An adapter is any class with an `evolve(task, server, max_evals) -> Result` method. The `server` is a terrarium `EvalServer` — use `server.evaluate()` to score candidates.

Create a Python file that defines `create_adapter()`:

```python
# my_adapter.py
from terrarium.adapter import Result
from terrarium.budget import BudgetExhausted

class MyEvolver:
    def evolve(self, task, server, max_evals):
        """
        Args:
            task: Task definition — use task.description, task.initial_candidate,
                  task.train_set, task.test_set as needed.
            server: EvalServer — the single eval + budget choke point.
                - server.evaluate(candidate) -> (score, info)
                - server.evaluate(candidate, example) -> (score, info)
                - Raises BudgetExhausted when budget runs out.
            max_evals: Total eval budget (informational; enforced by server).
        """
        candidate = task.initial_candidate
        best_candidate = candidate
        best_score = float("-inf")

        for i in range(max_evals):
            try:
                score, info = server.evaluate(candidate)
            except BudgetExhausted:
                break

            if score > best_score:
                best_score = score
                best_candidate = candidate

            # Your mutation/evolution logic here
            candidate = self.improve(candidate, score, info)

        return Result(best_candidate=best_candidate, best_score=best_score)

    def improve(self, candidate, score, info):
        # ... your evolution logic ...
        return candidate

def create_adapter():
    return MyEvolver()
```

Then run:

```bash
python -m terrarium circle_packing my_adapter.py --max-evals 100
```

### For dataset tasks

Dataset tasks have examples in `task.train_set`. Pass them to `server.evaluate()`:

```python
class DatasetEvolver:
    def evolve(self, task, server, max_evals):
        candidate = task.initial_candidate
        best_candidate, best_score = candidate, float("-inf")

        for i in range(max_evals // len(task.train_set)):
            scores = []
            for example in task.train_set:
                score, info = server.evaluate(candidate, example)
                scores.append(score)

            avg = sum(scores) / len(scores)
            if avg > best_score:
                best_score = avg
                best_candidate = candidate

            candidate = self.improve(candidate, scores)

        return Result(best_candidate=best_candidate, best_score=best_score)
```

### Built-in adapters

- **`terrarium/adapters/gepa.py`** — Uses GEPA's `optimize_anything`. Calls `server.evaluate()` (in-process, no HTTP).
- **`terrarium/adapters/claude_code.py`** — Launches Claude Code as a subprocess. Gives it a shell script that POSTs to `server.url` (HTTP, budget enforced server-side).

### Black-box / external system adapters

For systems that run as external processes, use the HTTP endpoint. The runner already starts the server — your adapter just needs to tell the subprocess where to POST:

```python
class ExternalSystemAdapter:
    def evolve(self, task, server, max_evals):
        # The server is already running (started by the runner).
        # Tell your external system to POST to it:
        #
        #   POST http://localhost:{port}/evaluate
        #   Body: {"candidate": "...", "example_id": "..."}
        #   Response: {"score": 1.23, "info": {...}, "budget": {...}}
        #   HTTP 429 when budget exhausted

        run_my_system(eval_url=server.url, task_desc=task.description)

        return Result(
            best_candidate=server.best_candidate,
            best_score=server.best_score,
            total_evals=server.budget.used,
        )
```

## How to Add a Benchmark Task

Create a new file in `terrarium/tasks/` and register it:

```python
# terrarium/tasks/my_task.py
from terrarium.registry import register_task
from terrarium.task import Example, Task

DESCRIPTION = "What this task optimizes and how it's scored."

INITIAL_CANDIDATE = "The starting text (code, prompt, etc.) to evolve from."

def evaluate(candidate: str) -> tuple[float, dict]:
    """Score a candidate. Return (score, info_dict).

    For dataset tasks, the signature is:
        def evaluate(candidate: str, example: Example) -> tuple[float, dict]
    """
    score = run_my_eval(candidate)
    return score, {"score": score}

# For dataset tasks, also define train/test sets:
# train_set = [Example(id="1", inputs={"problem": "..."}, expected="answer"), ...]
# test_set = [...]

TASK = register_task(Task(
    name="my_task",
    description=DESCRIPTION,
    initial_candidate=INITIAL_CANDIDATE,
    eval_fn=evaluate,
    # train_set=train_set,   # for dataset tasks
    # test_set=test_set,     # for dataset tasks
    metadata={
        "type": "single_task",  # or "generalization"
        "candidate_type": "code",  # or "prompt"
        "objective": "One-line objective for the evolution system.",
    },
))
```

Then add the import to `terrarium/tasks/__init__.py`:

```python
from terrarium.tasks import my_task  # registers on import
```

### Task types

- **Single-task** (`train_set=None`): One hard problem. `evaluate(candidate)` takes no example. E.g., circle packing, blackbox optimization.
- **Dataset / generalization** (`train_set=[...]`): Many related problems. `evaluate(candidate, example)` is called per-example. E.g., AIME math, ARC-AGI.

### Eval function guidelines

- Return `(score: float, info: dict)`. Higher score = better.
- The `info` dict should include diagnostic feedback that helps the evolution system improve (error messages, partial results, execution traces).
- Keep eval functions deterministic when possible.
- Import heavy dependencies inside the function body to keep task registration fast.

## Architecture

### Unified eval path

```
In-process adapter (GEPA)          External adapter (Claude Code)
        │                                     │
        │ server.evaluate(candidate)          │ POST /evaluate {"candidate": "..."}
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
              ┌─────────────────┐
              │   EvalServer    │   ◄── single choke point
              │                 │
              │  budget.check() │   ◄── pre-check
              │  task.eval_fn() │   ◄── actual eval
              │  budget.record()│   ◄── count it
              │  track best     │   ◄── update leaderboard
              └─────────────────┘
```

The `BudgetTracker` and `EvalServer` live in the terrarium process. The evolution system — whether it's a Python library running in-process or a subprocess communicating over HTTP — can only evaluate through this server. It cannot read, write, or reset the budget counter.

### HTTP endpoints

```
POST /evaluate  — submit a candidate, get score (HTTP 429 when budget exhausted)
GET  /status    — check budget and best score
GET  /task      — get task description and initial candidate
```
