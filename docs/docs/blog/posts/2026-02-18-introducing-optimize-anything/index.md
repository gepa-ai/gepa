---
date:
  created: 2026-02-18
authors:
  - lakshya
  - donghyun
  - wenjie
  - karim
  - shangyin
  - sewon
  - ion
  - joey
  - omar
  - alex
  - matei
equal_contribution:
  - "Lakshya A Agrawal"
  - "Donghyun Lee"
slug: introducing-optimize-anything
title: "optimize_anything: A Universal API for Text Optimization"
description: "A new API setting state-of-the-art results on optimizing code, prompts, agent architectures, and more — if you can measure it, you can optimize it."
---

# `optimize_anything`: A Universal API for Text Optimization

Many real-world problems — designing accurate-but-cheap AI agents, writing fast CUDA kernels, finding cost-minimizing cloud-scheduling policies, crafting effective prompts — share a common structure: maximize an objective **f(x) → score**, where **x** is a text artifact like code, a prompt, a policy, or even numeric hyperparameters serialized as a string. Today, we are introducing **`optimize_anything`**, a simple, declarative API that optimizes any artifact representable as text. You bring a **starting artifact** and an **evaluator**; `optimize_anything` uses LLMs as intelligent proposers to iteratively refine it.

<!-- more -->

<figure markdown="span">
  ![optimize_anything diagram showing the core loop: a string x is passed to an evaluator f(x) which returns a score plus Actionable Side Information, which is then consumed by an LLM intelligent proposer to produce an improved string. Example instantiations shown below: Code Optimization, Numeric Optimization, Prompt/Agent Harness, Policy Optimization.](header_image.png)
  <figcaption>The optimize_anything loop: evaluate a text artifact, capture diagnostic feedback (ASI), and use an LLM to propose targeted improvements.</figcaption>
</figure>

While recent systems like AlphaEvolve, OpenEvolve, and ShinkaEvolve have demonstrated the power of LLM-guided search for algorithm discovery, they operate exclusively in single-task search mode — optimizing one problem at a time, with no built-in support for batch optimization across related tasks or generalization to unseen inputs. `optimize_anything` provides a unified declarative API targeting **three optimization modes** — single-task search, multi-task search, and generalization — that makes LLM-based optimization as simple as defining an evaluator. It is the first API in this paradigm to unify all three modes under one interface.

We demonstrate `optimize_anything` across six diverse domains: it **matches Optuna** — a mature numerical optimizer — on blackbox mathematical optimization, **outperforms AlphaEvolve, ShinkaEvolve, and OpenEvolve** on circle packing, **generates fast CUDA kernels** on KernelBench, discovers **state-of-the-art cloud scheduling algorithms saving 37.3% on egress costs**, achieves **state-of-the-art on prompt optimization**, and rewrites entire **agent architectures** to achieve a +57% (from 32.50% to 89.50%) on ARC-AGI using the same model.

## If It's Text, You Can Optimize It

The key insight behind `optimize_anything` is that a surprisingly wide range of problems can be formulated as optimizing a text artifact. Consider:

- **Code**: CUDA kernels, scheduling algorithms, solver implementations — all are text that can be evaluated by execution.
- **Prompts**: System prompts, few-shot templates, chain-of-thought instructions — evaluated by running them against a dataset.
- **Agent harnesses**: The entire agent — its prompts, sub-agents, control flow, and orchestration logic — is code, and code is text.
- **Configurations**: Hyperparameters, policies, and decision rules can be serialized as JSON strings and evaluated by simulation.

If an artifact can be serialized to text and evaluated programmatically, an LLM can reason about it and propose improvements. `optimize_anything` provides the interface to make this loop trivial to set up.


## From Evolution to Intelligent Design

Traditional optimization methods — gradient descent, evolutionary strategies, Bayesian optimization — rely on purely numerical feedback. They know *that* a candidate failed, but not *why*. You can't pass a numerical optimizer a stack trace. You can't show it the compiler error that explains exactly which line crashed. All that diagnostic context — compiler errors, profiling traces, agent reasoning logs, constraint violation details — gets discarded.

Optimizing without this context is like fumbling in a dark room. `optimize_anything` turns on the light. It captures diagnostic data as **Actionable Side Information (ASI)** — and because the proposer is an LLM, it can actually *read* this feedback and act on it. ASI enables the proposer to move beyond random mutation and take reasoned, corrective action — like reading a stack trace and rewriting the exact API call that caused a crash, or seeing that "Circle 3 and Circle 7 overlap by 0.02 units" and adjusting those specific coordinates.

<figure markdown="span">
  ![Comparison of numerical optimization (left) using gradients on a loss surface, versus text-based optimization (right) using Actionable Side Information. In numerical optimization, the gradient provides direction for improvement. In text-based optimization, ASI — execution traces, error messages, scores — provides reasoned direction for improvement via an LLM.](side_info_diagram.png)
  <figcaption>ASI is the text-optimization analogue of the gradient. Where gradients tell a numerical optimizer which direction to move, ASI tells an LLM proposer why a candidate failed and how to fix it.</figcaption>
</figure>

This marks a fundamental shift. Traditional evolutionary algorithms act like nature's "blind watchmaker" — applying random mutations and keeping whatever scores higher. An LLM with ASI acts as a **designer**: it reads the feedback, diagnoses the problem, and proposes a targeted fix. We are moving from *evolution* to **Intelligent Design**.

**Evolution (blind):** "Change a parameter and hope the score goes up."

**Intelligent Design (ASI):** "The agent failed because it hallucinated an API call. I will rewrite the system prompt to restrict tools to the provided schema."


## The `optimize_anything` API

### The Simplest Form

At its core, the API requires just two things: a starting artifact and an evaluator.

```python
import gepa.optimize_anything as oa

def evaluate(candidate: str) -> float:
    score, diagnostic = run_my_system(candidate)
    print(f"Error: {diagnostic}")  # captured automatically as ASI
    return score

result = oa.optimize_anything(
    seed_candidate="<your initial artifact>",
    evaluator=evaluate,
)
print(result.best_candidate)
```

That's it. The evaluator takes a candidate string and returns a score (higher is better). Any `print()` statements inside the evaluator are automatically captured as ASI and shown to the LLM proposer during reflection. You can also use `oa.log()` for explicit logging, or return structured diagnostic information as a dictionary:

```python
def evaluate(candidate: str) -> tuple[float, dict]:
    result = execute_code(candidate)
    return result.score, {
        "Error": result.stderr,
        "Output": result.stdout,
        "Runtime": f"{result.time_ms:.1f}ms",
    }
```

The side information can be anything: text, structured data, or even **images** (via `gepa.image.Image`) for vision-capable LLMs. The principle is simple: include anything that would help a human expert understand why the candidate succeeded or failed. The LLM will use it to make targeted improvements.

### One Interface, Three Optimization Modes

`optimize_anything` unifies three distinct optimization paradigms under one API, determined by whether you provide a `dataset` and `valset`:

<figure markdown="span">
  ![Decision tree showing three optimization modes. "What is the Goal?" branches into "I want answers" (Search Mode) and "I want a system" (Generalization Mode). Search Mode further splits into "One Problem (Single-Instance)" for tasks like blackbox optimization, and "Many Related Problems (Batch/Transfer)" for tasks like writing CUDA kernels for multiple algorithms.](optimization_problem_types.png)
  <figcaption>The three optimization modes: single-task search, multi-task search, and generalization.</figcaption>
</figure>

**1. Single-Task Search** — "Solve one hard problem." No dataset needed — the candidate *is* the solution, and the evaluator scores it directly (no `example` argument). Examples: finding an optimal circle packing arrangement, discovering a blackbox optimization algorithm. This is the mode that AlphaEvolve, ShinkaEvolve, and OpenEvolve operate in.

```python
oa.optimize_anything(seed_candidate=..., evaluator=...)
```

**2. Multi-Task Search** — "Solve a batch of related problems with cross-transfer." You provide a `dataset` of related tasks. Insights from solving one task help solve the others. Example: optimizing CUDA kernels for multiple algorithms on the same hardware.

```python
oa.optimize_anything(seed_candidate=..., evaluator=..., dataset=tasks)
```

**3. Generalization** — "Build a skill that transfers to unseen problems." You provide both a training `dataset` and a held-out `valset`. The optimizer must learn a general artifact — a prompt, an agent, a policy — that generalizes. This abstracts over traditional machine learning and program synthesis. Examples: prompt optimization, agent architecture discovery, cloud scheduling policy learning.

```python
oa.optimize_anything(seed_candidate=..., evaluator=..., dataset=train, valset=val)
```

The full API signature:

```python
def optimize_anything(
    seed_candidate: str | dict[str, str],  # Starting artifact
    evaluator: Callable,                    # Score + ASI
    dataset: list | None = None,            # Training examples (modes 2 & 3)
    valset: list | None = None,             # Validation set (mode 3)
    objective: str | None = None,           # What to optimize for (natural language)
    background: str | None = None,          # Domain knowledge and constraints
    config: GEPAConfig | None = None,       # Engine, reflection, tracking settings
) -> GEPAResult
```

### The Pareto Insight

In multi-task and generalization modes, `optimize_anything` leverages GEPA's core algorithmic insight: **Pareto-efficient search**. Rather than asking the LLM to improve *all* metrics simultaneously, each reflection step focuses on improving a specific subset of metrics or examples. Over multiple iterations, different subsets are selected — a minibatch of 2-3 examples or objectives at a time — and the proposer can focus its effort. This cycling through subsets, combined with Pareto-efficient selection that preserves candidates that excel on different axes, consistently outperforms naive all-at-once optimization.

## Let's Take It for a Spin

Here is a complete, working example that optimizes SVG code to depict "a pelican riding a bicycle." The evaluator renders the SVG to PNG, asks a VLM to score it on multiple visual aspects, and passes the rendered image back as ASI so the LLM proposer can *see* what it's improving:

```python
import base64, re
from functools import partial
import cairosvg, litellm
from gepa.image import Image
from gepa.optimize_anything import (
    optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig,
)

GOAL = "a pelican riding a bicycle"

def evaluate(candidate, example, *, model):
    """Render SVG → PNG, score with a VLM, return (score, side_info)."""
    svg = re.search(r"(<svg[\s\S]*?</svg>)", candidate["svg_code"]).group(1)
    png_b64 = base64.b64encode(
        cairosvg.svg2png(bytestring=svg.encode(), output_width=1280, output_height=1280)
    ).decode()

    resp = litellm.completion(model=model, messages=[{"role": "user", "content": [
        {"type": "text", "text": example["criteria"]},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
    ]}])
    text = resp.choices[0].message.content
    m = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text)

    return float(m.group(1)) / 10 if m else 0.0, {
        "RenderedSVG": Image(base64_data=png_b64, media_type="image/png"),
        "Feedback": text,
    }

result = optimize_anything(
    seed_candidate={"svg_code": open("seed.svg").read()},
    evaluator=partial(evaluate, model="vertex_ai/gemini-3-pro-preview"),
    dataset=[  # 6 visual aspects → Pareto-efficient selection
        {"id": "overall",     "criteria": f"Rate overall quality of this SVG ({GOAL}). SCORE: X/10"},
        {"id": "anatomy",     "criteria": "Rate pelican accuracy: beak, pouch, plumage. SCORE: X/10"},
        {"id": "bicycle",     "criteria": "Rate bicycle: wheels, frame, handlebars, pedals. SCORE: X/10"},
        {"id": "composition", "criteria": "Rate how convincingly the pelican rides the bicycle. SCORE: X/10"},
        {"id": "visual",      "criteria": "Rate visual appeal and color usage. SCORE: X/10"},
        {"id": "craft",       "criteria": "Rate SVG technical quality: shapes, layering. SCORE: X/10"},
    ],
    objective=f"Optimize SVG code to illustrate '{GOAL}'. Output ONLY valid SVG.",
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=300),
        reflection=ReflectionConfig(
            reflection_lm="vertex_ai/gemini-3-pro-preview",
            reflection_minibatch_size=2,  # focus on 2 aspects per iteration
        ),
    ),
)
print(f"Best score: {result.val_aggregate_scores[result.best_idx]:.3f}")
```

A few things to note:

- The **`dataset`** contains 6 evaluation aspects. GEPA calls the evaluator once per aspect per candidate, tracking scores individually. This enables Pareto-efficient selection: a candidate that excels at bicycle structure but struggles with pelican anatomy is preserved on the frontier, not discarded behind a mediocre average.
- **`reflection_minibatch_size=2`** means each reflection step shows the LLM feedback from just 2 aspects. Over multiple iterations, all 6 aspects get attention — but each reflection is focused and targeted.
- The **rendered image** is passed in `side_info` via `Image(base64_data=...)`, so the VLM proposer can literally *see* what it's improving.
- Starting from a simple seed (score 0.017), the optimizer produces a detailed illustration scoring **0.817** in 20 candidates:

<div style="display: flex; align-items: center; justify-content: center; gap: 1rem;" markdown>
<div style="flex: 1; text-align: center; min-width: 0;" markdown>

![Initial SVG: a basic pelican sketch on a red-framed bicycle against a white background.](initial-pelican.png){ style="width: 100%;" }

<div style="margin: 0.5rem 0 0; max-width: none; width: 100%;"><em>Seed (score: 0.017)</em></div>

</div>
<div style="flex: 1; text-align: center; min-width: 0;" markdown>

![Optimized SVG: a polished pelican riding a bicycle with sky, clouds, sun, road, and grass.](best-pelican.png){ style="width: 100%;" }

<div style="margin: 0.5rem 0 0; max-width: none; width: 100%;"><em>Best candidate (score: 0.817)</em></div>

</div>
</div>

*20 candidates, 300 metric calls. The optimizer added background elements, improved anatomy, and refined the composition — all through LLM reflection on rendered image feedback.*


## Results

We apply `optimize_anything` to six diverse domains spanning search, batch optimization, and generalization. Each result below links to the corresponding [appendix section](#appendix-case-study-code) with the full code.

### 1. Blackbox Mathematical Optimization: Matching Optuna

**Mode: Single-Task Search.** We ask `optimize_anything` to write Python code that implements a blackbox optimization algorithm, and evaluate it on the [EvalSet](https://github.com/sigopt/evalset) benchmark (56 problems). GEPA doesn't solve the optimization problems directly — it evolves the *solver code* itself.

<figure markdown="span">
  ![Bar chart showing GEPA vs Optuna on 56 EvalSet problems with 1% tolerance and 8000 trials. GEPA wins on 7 problems, Optuna wins on 9, and they tie on 40.](optuna_comparison.png)
  <figcaption>GEPA's optimize_anything matches Optuna, the industry-standard blackbox optimizer, on the EvalSet benchmark. On 56 problems, GEPA ties on 40, winning 7 and losing 9.</figcaption>
</figure>

**Key result:** `optimize_anything` matches the performance of Optuna, a mature numerical optimizer, by evolving solver code from a random-search seed. [Full code →](#appendix-a-blackbox-mathematical-optimization)

### 2. Circle Packing: Outperforming AlphaEvolve

**Mode: Single-Task Search.** Pack n=26 circles to maximize the sum of their radii within a unit square. GEPA evolves the packing algorithm code, using execution results and geometric diagnostics as ASI.

<figure markdown="span">
  ![Left: line chart comparing GEPA, ShinkaEvolve, OpenEvolve, and AlphaEvolve on circle packing (n=26). GEPA reaches the highest score (~2.636) with fewer metric calls. Right: magnified view showing GEPA slightly above AlphaEvolve's best.](circle_packing_comparison.png)
  <figcaption>GEPA outperforms AlphaEvolve, ShinkaEvolve, and OpenEvolve on circle packing (n=26), reaching a higher score with fewer evaluations.</figcaption>
</figure>

<figure markdown="span">
  ![Three snapshots of circle packing optimization: Metric Call 0 (score=0.98, sparse circles), Metric Call 50 (score=2.61, tightly packed), Metric Call 89 (score=2.64, near-optimal packing).](circle_packing_trajectory.png)
  <figcaption>Visual progression of the circle packing optimization: from an initial naive arrangement to a near-optimal packing.</figcaption>
</figure>

**Key result:** GEPA outperforms all three open-source LLM-evolution frameworks, reaching a score of 2.636. [Full code →](#appendix-b-circle-packing)

### 3. CUDA Kernel Generation (KernelBench)

**Mode: Multi-Task Search.** Optimize a prompt that instructs an LLM to generate fast CUDA kernels for multiple algorithms from [KernelBench](https://github.com/ScalingIntelligence/KernelBench). Insights from optimizing one kernel transfer to others via shared prompt improvements.

<figure markdown="span">
  ![Line chart showing KernelBench performance vs budget. Fast_p(0) — any correct kernel — reaches 100%. Fast_p(1.0) — matching baseline speed — reaches 87%. Fast_p(1.1) — 10% faster — reaches 48%. Fast_p(1.2) — 20% faster — reaches 25%.](kernelbench_results.png)
  <figcaption>KernelBench results with GEPA (gpt-5 as proposer). 87% of generated kernels match or beat baseline performance; 25% are 20%+ faster.</figcaption>
</figure>

**Key result:** 87% of GEPA-generated kernels match or beat the baseline, with 25% achieving 20%+ speedups. [Full code →](#appendix-c-cuda-kernel-generation)

### 4. AI-Driven Systems Research: CloudCast & Can't Be Late

**Mode: Generalization.** We optimize cloud infrastructure algorithms: **CloudCast** discovers broadcast routing strategies for multi-cloud data transfer (minimizing egress cost), and **Can't Be Late** learns scheduling policies that decide when to use cheap-but-preemptible SPOT instances versus reliable ON_DEMAND instances to complete tasks before deadlines.

<figure markdown="span">
  ![Optimization trajectory for Can't Be Late showing cost savings (%) vs metric calls. Starting from 0% savings (baseline cost=$96.5), GEPA discovers a strategy achieving 7.8% cost savings (optimized cost=$89.0).](cant_be_late_trajectory.png)
  <figcaption>GEPA optimization progress on Can't Be Late: from a simple deadline-check heuristic to a sophisticated scheduling strategy with 7.8% cost savings.</figcaption>
</figure>

**Key result:** `optimize_anything` discovers state-of-the-art algorithms for both problems — **37.3% cost savings** on CloudCast and **7.8% cost savings** on Can't Be Late — outperforming hand-designed heuristics. [Full code →](#appendix-d-cloudcast--cant-be-late)

### 5. Prompt Optimization: AIME Mathematics

**Mode: Generalization.** Optimize a system prompt for solving [AIME](https://en.wikipedia.org/wiki/American_Invitational_Mathematics_Examination) 2025 math competition problems. The evaluator runs gpt-4.1-mini with the candidate prompt and scores whether the answer is correct. GEPA is the [state-of-the-art prompt optimization algorithm](https://gepa-ai.github.io/gepa/).

<figure markdown="span">
  ![Optimization trajectory for AIME 2025 with gpt-4.1-mini. Validation score improves from 46.67% to 57.78% over 400 metric calls. Best test score reaches 60.00%, up from a 46.67% baseline.](aime_results.png)
  <figcaption>AIME 2025 prompt optimization: gpt-4.1-mini accuracy improves from 46.67% to 60.00% through prompt refinement alone.</figcaption>
</figure>

**Key result:** Pure prompt optimization improves gpt-4.1-mini from 46.67% to **60.00%** on AIME 2025 — a 13.3 percentage point gain from changing only the system prompt. [Full code →](#appendix-e-aime-prompt-optimization)

### 6. Agent Architecture Discovery: ARC-AGI

**Mode: Generalization.** This is the most ambitious application. Rather than optimizing a prompt, we optimize the **entire agent**: its code, sub-agent architecture, control flow, helper functions, and prompts — all treated as a single text artifact. The seed is a 10-line naive agent; GEPA evolves it into a 300+ line system with rule induction, code verification, iterative refinement, and structured fallbacks.

<figure markdown="span">
  ![Optimization trajectory for ARC-AGI with Gemini 3 Flash. Validation accuracy improves from 56.5% to 93.5%. Base test score is 32.5%, best test score reaches 89.5%.](arc_agi_trajectory.png)
  <figcaption>ARC-AGI agent evolution: from a naive 10-line agent (32.5% test) to a sophisticated 300+ line system (89.5% test) with Gemini 3 Flash.</figcaption>
</figure>

**Key result:** `optimize_anything` improves ARC-AGI test accuracy from 32.5% to **89.5%** by evolving the entire agent architecture. Many research teams manually iterate on agent designs over weeks — `optimize_anything` automates this process. [Full code →](#appendix-f-arc-agi-agent-architecture-discovery)


## Conclusion & Getting Started

`optimize_anything` makes a simple bet: if your artifact is text and your evaluator is programmatic, you can optimize it. The API is minimal — a seed, an evaluator, and optionally a dataset. The results span blackbox optimization, algorithmic discovery, kernel generation, systems research, prompt tuning, and agent architecture search.

The key ideas: (1) **ASI** turns the optimizer from a blind mutator into an intelligent designer; (2) **Pareto-efficient search** across metrics and examples outperforms naive all-at-once optimization; (3) **three unified modes** cover the full spectrum from solving one hard problem to learning generalizable skills.

Get started:

```bash
pip install gepa
```

```python
import gepa.optimize_anything as oa

result = oa.optimize_anything(
    seed_candidate="<your artifact>",
    evaluator=your_evaluator,
)
```

- [Documentation](https://gepa-ai.github.io/gepa/)
- [GitHub](https://github.com/gepa-ai/gepa)
- [Discord](https://discord.gg/A7dABbtmFw)

---

## Appendix: Case Study Code

### Appendix A: Blackbox Mathematical Optimization

`optimize_anything` evolves Python code that implements a blackbox optimizer. The evaluator sandboxes the proposed code, runs it on benchmark functions, and returns execution diagnostics as ASI.

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

SEED_CODE = """
import numpy as np
def solve(objective_fn, bounds, budget):
    best_y = float('inf')
    for _ in range(budget):
        x = np.random.uniform(bounds[:, 0], bounds[:, 1])
        y = objective_fn(x)
        if y < best_y: best_y = y
    return best_y
"""

def evaluate(candidate, opt_state):
    code = candidate["code"]
    best_xs = extract_best_xs(opt_state)  # warm-start from prior evaluations
    result = execute_code(code, problem_index=0, timeout=300, budget=100, best_xs=best_xs)

    return result["score"], {
        "top_50_attempts": result["top_50_attempts"],
        "bottom_50_attempts": result["bottom_50_attempts"],
        "Stdout": result.get("stdout", ""),
        "Error": result.get("error", ""),
        "Traceback": result.get("traceback", ""),
    }

optimize_anything(
    seed_candidate={"code": SEED_CODE},
    evaluator=evaluate,
    config=GEPAConfig(
        engine=EngineConfig(max_candidate_proposals=20, cache_evaluation=True),
        reflection=ReflectionConfig(reflection_lm="openai/gpt-5"),
    ),
    objective="Evolve the code to implement a sophisticated black-box optimizer "
              "that efficiently minimizes the objective function.",
    background="The target functions are non-convex and high-dimensional. "
               "Gradient information is not available.",
)
```

### Appendix B: Circle Packing

Optimize code that packs n=26 circles within a unit square, maximizing the sum of radii. The evaluator executes the packing code, validates circle positions, and returns geometric diagnostics.

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

def evaluate(candidate, opt_state):
    code = candidate["code"]
    warm_start = extract_best_circles(opt_state)
    result = execute_code(code, timeout=600, warm_start=warm_start)

    if result["success"]:
        score = result["result"]["validation_details"]["sum_radii"]
        metrics = compute_multiple_metrics(result["result"]["all_scores"])
    else:
        score, metrics = 0.0, {"sum_radii": 0.0}

    return score, {
        "scores": {"sum_radii": score},
        "metrics": metrics,
        "circles": result.get("result", {}).get("circles"),
        "stdout": result.get("stdout", ""),
        "error": result.get("error"),
        "traceback": result.get("traceback"),
    }

optimize_anything(
    seed_candidate={"code": SEED_CODE},
    evaluator=evaluate,
    config=GEPAConfig(
        engine=EngineConfig(run_dir="outputs/circle_packing", max_metric_calls=150, cache_evaluation=True),
        reflection=ReflectionConfig(reflection_lm="openai/gpt-5"),
    ),
    objective="Optimize circle packing code to maximize sum of circle radii "
              "within a unit square for N=26 circles.",
)
```

### Appendix C: CUDA Kernel Generation

Multi-task search over KernelBench problems. A shared prompt is optimized to instruct an LLM to generate fast, correct CUDA kernels across multiple algorithms. The `RefinerConfig` enables automatic per-evaluation refinement — after each evaluation, an LLM proposes a refined candidate based on the feedback.

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, RefinerConfig

def evaluate(candidate, example):
    baseline = baselines[example.problem_id]
    code, cuda_docs, eval_result = run_kernel(
        candidate["kernel_gen_prompt"], example.ref_arch, lm, predictor
    )
    score = compute_score(eval_result, baseline)

    return score, {
        "Score": score,
        "problem_id": example.problem_id,
        "Code": code,
        "runtime_ms": eval_result.get("PerformanceStatsMean"),
        "speedup": baseline / eval_result.get("PerformanceStatsMean") if eval_result.get("PerformanceStatsMean") else None,
        "is_correct": eval_result.get("CorrectnessSucceeded", False),
        "error_feedback": format_error(eval_result) if eval_result.get("ErrorType") else None,
    }

optimize_anything(
    seed_candidate={"kernel_gen_prompt": KERNEL_GEN_PROMPT},
    evaluator=evaluate,
    dataset=dataset,  # multiple KernelBench problems
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=2000, cache_evaluation=True),
        refiner=RefinerConfig(),  # auto-refine after each evaluation
    ),
    objective=OBJECTIVE,
    background=BACKGROUND,
)
```

### Appendix D: CloudCast & Can't Be Late

Generalization mode for cloud infrastructure optimization. Both examples optimize Python code implementing scheduling/routing strategies, evaluated by simulation.

**CloudCast** — broadcast routing for multi-cloud data transfer:

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

INITIAL_PROGRAM = """
def search_algorithm(src, dsts, G, num_partitions):
    # Baseline: Dijkstra shortest path by cost
    ...
"""

optimize_anything(
    seed_candidate={"program": INITIAL_PROGRAM},
    evaluator=create_evaluator(timeout=300),
    dataset=train_set,
    valset=val_set,
    objective="Optimize a broadcast routing algorithm for multi-cloud data transfer. "
              "Minimize total cost while maintaining good transfer times.",
    background="Nodes are cloud regions. Edges have cost ($/GB) and throughput (Gbps). "
               "Data is partitioned into chunks that can be routed independently.",
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=100),
        reflection=ReflectionConfig(reflection_lm=llm_model),
    ),
)
```

**Can't Be Late** — cloud scheduling with SPOT vs ON_DEMAND instances:

```python
INITIAL_STRATEGY = """
class EvolveSingleRegionStrategy(Strategy):
    def _step(self, last_cluster_type, has_spot):
        remaining_time = self.deadline - self.env.elapsed_seconds
        remaining_task = self.task_duration - sum(self.task_done_time)
        if remaining_task + self.restart_overhead >= remaining_time:
            return ClusterType.ON_DEMAND  # deadline pressure
        return ClusterType.SPOT if has_spot else ClusterType.NONE
"""

optimize_anything(
    seed_candidate={"program": INITIAL_STRATEGY},
    evaluator=create_evaluator(timeout=300),
    dataset=train_set,
    valset=val_set,
    objective="Optimize a cloud scheduling strategy. Minimize cost while "
              "ensuring task completion before deadline.",
    background="SPOT: ~$0.3/hr, preemptible. ON_DEMAND: ~$1/hr, reliable. "
               "The strategy must guarantee deadline completion.",
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=100, parallel=True, max_workers=128),
        reflection=ReflectionConfig(reflection_lm=llm_model),
    ),
)
```

### Appendix E: AIME Prompt Optimization

Generalization mode for prompt optimization. The evaluator runs a math-solving LLM with the candidate prompt and checks whether the answer is correct.

```python
import dspy
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

class MathSolverSignature(dspy.Signature):
    input = dspy.InputField(desc="The math problem to solve.")
    answer = dspy.OutputField(desc="The final numerical answer.")

predictor = dspy.ChainOfThought(MathSolverSignature)

def evaluate(candidate, example):
    predictor.predict.signature.instructions = candidate["prompt"]
    prediction = predictor(input=example.input)
    score = float(int(example.answer) == int(prediction.answer))

    return score, {
        "Input": example.input,
        "Output": prediction.answer,
        "Expected": example.answer,
        "Reasoning": getattr(prediction, "reasoning", ""),
        "Feedback": f"{'Correct' if score else 'Incorrect'}. Answer: {example.answer}.",
    }

trainset, valset, testset = load_math_dataset()

result = optimize_anything(
    seed_candidate={"prompt": "Solve the math problem carefully. "
                              "Break down the steps and provide the final answer."},
    evaluator=evaluate,
    dataset=trainset,
    valset=valset,
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=600, parallel=True, max_workers=32, cache_evaluation=True),
        reflection=ReflectionConfig(reflection_lm=dspy.LM("openai/gpt-5.1")),
    ),
)
```

### Appendix F: ARC-AGI Agent Architecture Discovery

Generalization mode where the **entire agent code** is the artifact being optimized. The seed is a 10-line naive agent; GEPA evolves it into a multi-stage system with rule induction, code verification, iterative refinement, and structured fallbacks.

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig

SEED_AGENT = '''
import json, re
def solve(train_inputs, train_outputs, test_inputs, llm):
    examples = "\\n".join(f"Input: {i}\\nOutput: {o}"
                          for i, o in zip(train_inputs, train_outputs))
    response = llm(f"Solve ARC puzzle. Training:\\n{examples}\\n"
                   f"Predict outputs as JSON [[...]]:")
    grids = [json.loads(g) for g in re.findall(r"\\[\\[.*?\\]\\]",
             response.replace("\\n", ""))]
    return {"train": grids[:len(train_inputs)],
            "test": [[g] for g in grids[len(train_inputs):]]}
'''

def evaluate(candidate, **kwargs):
    ex = kwargs["example"]
    result = run_agent(
        agent_code=candidate["agent_code"],
        train_in=ex.train_in, train_out=ex.train_out,
        test_in=ex.test_in, test_out=ex.test_out,
        model_id="openrouter/google/gemini-3-flash-preview",
        max_llm_calls=10,
    )
    score = result["test_score"]

    return score, {
        "problem_id": ex.problem_id,
        "training_score": result["training_score"],
        "test_score": result["test_score"],
        "error": result["error"],
        "train_examples": result["train_examples"],
        "test_examples": result["test_examples"],
    }

train_set, val_set, test_set = load_arc_dataset()

result = optimize_anything(
    seed_candidate={"agent_code": SEED_AGENT},
    evaluator=evaluate,
    dataset=train_set,
    valset=val_set,
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=4000, parallel=True,
                            max_workers=64, cache_evaluation=True),
        reflection=ReflectionConfig(
            reflection_lm="openrouter/google/gemini-3-flash-preview",
        ),
    ),
    objective="Evolve agent code to solve ARC-AGI puzzles. The agent receives "
             "training input/output pairs and must predict test outputs.",
    background="ARC puzzles require discovering a transformation rule from "
               "examples and applying it to unseen inputs. The agent has access "
               "to an LLM via the llm() callable.",
)
```

The optimized agent grew from 10 lines to 300+ lines, developing its own helper library for grid analysis, a multi-stage rule induction pipeline, iterative code refinement with verification, and a structured fallback to direct LLM prediction — all discovered automatically by `optimize_anything`.
