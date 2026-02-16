---
date:
  created: 2026-02-18
authors:
  - lakshya
  - donghyun
  - wenjie
  - karim
  - shangyin
  - sanjit
  - koushik
  - sewon
  - dan
  - ion
  - joey
  - omar
  - alex
  - matei
equal_contribution:
  - "Lakshya A Agrawal"
  - "Donghyun Lee"
slug: introducing-optimize-anything
readtime: 12
title: "optimize_anything: A Universal API for Text Optimization"
description: "A new API setting state-of-the-art results on optimizing code, prompts, agent architectures, and more — if you can measure it, you can optimize it."
---

# `optimize_anything`: A Universal API for Text Optimization

Many real-world problems — designing accurate-but-cheap AI agents, writing fast CUDA kernels, finding cost-minimizing cloud-scheduling policies, crafting effective prompts, learning agent skills for coding agents (claude-code) — share a common structure: maximize an objective **f(x) → score**, where **x** is a text artifact like code, a prompt, a policy, or even numeric hyperparameters serialized as a string. Today, we are introducing **`optimize_anything`**, a simple, declarative API that optimizes any artifact representable as text. You bring a **starting artifact** and an **evaluator**; `optimize_anything` uses LLMs as intelligent proposers to iteratively refine it. If you can measure it, you can optimize it.

<!-- more -->

<figure markdown="span">
  ![optimize_anything diagram showing the core loop: a string x is passed to an evaluator f(x) which returns a score plus Actionable Side Information, which is then consumed by an LLM intelligent proposer to produce an improved string. Example instantiations shown below: Code Optimization, Numeric Optimization, Prompt/Agent Harness, Policy Optimization.](header_image.png)
  <figcaption>The optimize_anything loop: evaluate a text artifact, capture diagnostic feedback (ASI), and use an LLM to propose targeted improvements.</figcaption>
</figure>

While recent systems like AlphaEvolve, OpenEvolve, and ShinkaEvolve have demonstrated the power of LLM-guided search for algorithm discovery, they operate exclusively in single-task mode — optimizing one problem at a time with no built-in support for batch optimization or generalization. `optimize_anything` unifies **three optimization modes** — single-task search, multi-task search, and generalization — under one declarative API.

We demonstrate `optimize_anything` across seven diverse domains: it **matches Optuna** — a mature, purpose-built numerical optimizer — on blackbox mathematical optimization, **outperforms AlphaEvolve, ShinkaEvolve, and OpenEvolve** on circle packing, **generates fast CUDA kernels** on KernelBench, discovers **state-of-the-art cloud scheduling algorithms saving 40.2% on egress costs**, achieves **state-of-the-art prompt optimization**, evolves an entire **agent architecture** to boost ARC-AGI accuracy from 32.5% to **89.5%**, and **learns agent skills** that boost Claude Code to near-perfect resolve rates while cutting resolution time by 40%.

## If It's Text, You Can Optimize It

The key insight behind `optimize_anything` is that a surprisingly wide range of problems can be formulated as optimizing a text artifact. Consider:

- **Code**: CUDA kernels, scheduling algorithms, solver implementations — all are text that can be evaluated by execution.
- **Prompts**: System prompts, few-shot templates, chain-of-thought instructions — evaluated by running them against a dataset.
- **Agent architectures**: The entire agent system — its prompts, sub-agents, control flow, and orchestration logic.
- **Configurations**: Hyperparameters, policies, and decision rules can be serialized as JSON strings and evaluated by simulation.
- **Agent Skills**: Repository-specific instructions and best practices for coding agents — evaluated by running the agent on real tasks from the codebase.

If an artifact can be serialized to text and evaluated programmatically, an LLM can reason about it and propose improvements. `optimize_anything` provides the interface to make this loop trivial to set up.


## From Evolution to Intelligent Design

Traditional optimization methods — gradient descent, evolutionary strategies, Bayesian optimization — operate on purely numerical feedback. They know *that* a candidate failed, but not *why*. You can't pass Optuna a stack trace. You can't show a Bayesian optimizer the compiler error that explains exactly which line crashed. All that diagnostic context — compiler errors, profiling traces, agent reasoning logs, constraint violation details — is reduced to a single scalar and the rest is discarded.

`optimize_anything` captures this diagnostic context as **Actionable Side Information (ASI)** — a first-class part of the evaluator contract that *you* design. ASI can be anything that would help an expert diagnose the problem: error messages, profiling data, rendered images, constraint violations, tool outputs. Because the proposer is an LLM, it can actually *read* this feedback. During a **dedicated reflection step**, it reasons about *why* specific candidates succeeded or failed — then proposes targeted improvements.

<figure>
  <div style="width:100%; max-width:800px; margin:0 auto; position:relative; padding-bottom:61.25%; height:0; overflow:hidden;">
    <iframe src="/static/diagrams/side_info_diagram.html" scrolling="no" style="position:absolute; top:0; left:0; width:100%; height:100%; border:none;"></iframe>
  </div>
  <figcaption>ASI is the text-optimization analogue of the gradient. Where gradients tell a numerical optimizer which direction to move, ASI tells an LLM proposer why a candidate failed and how to fix it.</figcaption>
</figure>

Because ASI is user-defined, it can be tailored to the domain: a dictionary of constraint violations for circle packing, a profiler trace for CUDA kernels, or a rendered image of a malformed SVG (via `gepa.image.Image`) so a vision-capable LLM (VLM) can literally *see* what it's improving. This marks a fundamental shift. Traditional evolutionary algorithms act like nature's "blind watchmaker" — applying random mutations and keeping whatever scores higher. An LLM with ASI acts as a **designer**: it reads the feedback, diagnoses the problem, and proposes a targeted fix. We are moving from *evolution* to **Intelligent Design**.

**Evolution (blind):** "Change a parameter and hope the score goes up."

**Intelligent Design (ASI):** "The agent failed because it called `search_api()` which doesn't exist. I will rewrite the system prompt to restrict tools to the provided schema."


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
  ![Decision tree showing three optimization modes. "What is the Goal?" branches into "I want answers" (Search Mode) and "I want a system" (Generalization Mode). Search Mode further splits into "One Problem (Single-Task)" for tasks like blackbox optimization, and "Many Related Problems (Multi-Task)" for tasks like writing CUDA kernels for multiple algorithms.](optimization_problem_types.png)
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
    seed_candidate: str | dict[str, str],   # Starting artifact
    evaluator: Callable,                    # Score + ASI
    dataset: list | None = None,            # Training examples (modes 2 & 3)
    valset: list | None = None,             # Validation set (mode 3)
    objective: str | None = None,           # What to optimize for (natural language)
    background: str | None = None,          # Domain knowledge and constraints
    config: GEPAConfig | None = None,       # Engine, reflection, tracking settings
) -> GEPAResult
```

Notice what's *absent*: no mutation prompts, no task-specific instruction templates, no island configurations, no EVOLVE-BLOCK markers (all common in prior LLM-evolution frameworks). You declare the **what** — your artifact, your evaluator, and any domain knowledge as `background` — and `optimize_anything` handles the **how**: prompt construction, reflection, candidate selection, and search strategy. This declarative design, inspired by [DSPy](https://github.com/stanfordnlp/dspy)'s principle of *programming not prompting*, means the same API call works whether you're optimizing a CUDA kernel, a cloud scheduling policy, or an agent architecture.

### The Pareto Insight

In multi-task and generalization modes, `optimize_anything` leverages GEPA's core algorithmic insight: **Pareto-efficient search**. Rather than asking the LLM to improve *all* metrics simultaneously, each reflection step focuses on improving a specific subset of metrics or examples. Over multiple iterations, different subsets are selected — a minibatch of 2-3 examples or objectives at a time — and the proposer can focus its effort. This cycling through subsets, combined with Pareto-efficient selection that preserves candidates that excel on different axes, consistently outperforms naive all-at-once optimization.

## Let's Take It for a Spin

Here is a complete, working example that optimizes SVG code to depict "a pelican riding a bicycle." Noteably, we use `optimize_anything` to directly optimize the SVG code itself, rather than, optimizing prompts for an LLM to produce the SVG code. Our evaluator renders the SVG code as a PNG image, asks a VLM to score it on a set of visual aspects, and passes the rendered image back as ASI so the LLM proposer can *see* what it's improving:

```python
import base64, re
from functools import partial
import cairosvg, litellm
from gepa.image import Image
from gepa.optimize_anything import (
    optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig,
)
from demo_utils import render_image, get_vlm_score_feedback

GOAL = "a pelican riding a bicycle"

def evaluate(candidate, example, *, model):
    """Render SVG → image, score with a VLM, return (score, side_info)."""
    image = render_image(candidate["svg_code"])
    score, feedback = get_vlm_score_feedback(model, image, example["criteria"])

    return score, {
        "RenderedSVG": Image(base64_data=image, media_type="image/png"),
        "Feedback": feedback,
    }

model = "vertex_ai/gemini-3-flash-preview"
result = optimize_anything(
    seed_candidate={"svg_code": open("seed.svg").read()},
    evaluator=partial(evaluate, model=model),
    dataset=[
        # 6 visual aspects → Pareto-efficient selection
        {"id": "overall",     "criteria": f"Rate overall quality of this SVG ({GOAL}). SCORE: X/10"},
        {"id": "anatomy",     "criteria": "Rate pelican accuracy: beak, pouch, plumage. SCORE: X/10"},
        {"id": "bicycle",     "criteria": "Rate bicycle: wheels, frame, handlebars, pedals. SCORE: X/10"},
        {"id": "composition", "criteria": "Rate how convincingly the pelican rides the bicycle. SCORE: X/10"},
        {"id": "visual",      "criteria": "Rate visual appeal, scenery, and color usage. SCORE: X/10"},
        {"id": "craft",       "criteria": "Rate SVG technical quality: shapes, layering. SCORE: X/10"},
    ],
    objective=f"Optimize SVG code to illustrate '{GOAL}'. Output ONLY valid SVG.",
    config=GEPAConfig(
        engine=EngineConfig(max_metric_calls=150),
        reflection=ReflectionConfig(
            reflection_lm=model,
            reflection_minibatch_size=2,  # focus on 2 aspects per iteration
        ),
    ),
)
print(result.best_candidate)
```



A few things to note:

- The **`dataset`** contains 6 evaluation aspects. GEPA calls the evaluator once per aspect per candidate, tracking scores individually. This enables Pareto-efficient selection: a candidate that excels at bicycle structure but struggles with pelican anatomy is preserved on the frontier, not discarded behind a mediocre average.
- Our desired visual aspects are defined in concise natural language. We avoid the need for detailed rubrics and simply rely on the VLM's judgment for scoring.
- **`reflection_minibatch_size=2`** means each reflection step shows the LLM feedback from just 2 aspects. Over multiple iterations, all 6 aspects get attention — but each reflection is focused and targeted.
- The **rendered image** is passed in `side_info` via `Image(base64_data=...)`, so the VLM proposer can literally *see* what needs improvement. The VLM evaluator **never** sees the SVG code — only the rendered image.
- The reflection model see the feedback (both positive and negative) along with the optimization artifact (source code), and directly optimizes it.
- Starting from a plain white canvas, the optimizer produces a detailed illustration scoring an average of **0.825** across all six aspects in 12 candidates:

<div style="display: flex; align-items: center; justify-content: center; gap: 1rem;" markdown>
<div style="flex: 1; text-align: center; min-width: 0;" markdown>

![Initial SVG: a basic pelican sketch on a red-framed bicycle against a white background.](gemini_3_flash_zero_shot.svg){ style="width: 100%;" }

<div style="margin: 0.5rem 0 0; max-width: none; width: 100%;"><em>Zero-shot attempt from Gemini 3 Flash</em></div>

</div>
<div style="flex: 1; text-align: center; min-width: 0;" markdown>

![Optimized SVG: a polished pelican riding a bicycle with sky, clouds, sun, road, and grass.](optimized_pelican_gemini_3_flash.svg){ style="width: 100%;" }

<div style="margin: 0.5rem 0 0; max-width: none; width: 100%;"><em>Best candidate out of 12 (score: 0.825)</em></div>

</div>
</div>

*With only 12 candidates, the optimizer added background elements, improved anatomy, increased the sophistication of all visual elements, and refined the composition — all through LLM reflection on rendered image feedback. Similar results were also observed with Claude Opus 4.6 who produced 20 candidates.*

<div style="display: flex; align-items: center; justify-content: center; gap: 1rem;" markdown>
<div style="flex: 1; text-align: center; min-width: 0;" markdown>

![Initial SVG: a basic pelican sketch on a red-framed bicycle against a white background.](claude-opus-initial-pelican.png){ style="width: 100%;" }

<div style="margin: 0.5rem 0 0; max-width: none; width: 100%;"><em>Zero-shot attempt from Claude Opus 4.6</em></div>

</div>
<div style="flex: 1; text-align: center; min-width: 0;" markdown>

![Optimized SVG: a polished pelican riding a bicycle with sky, clouds, sun, road, and grass.](claude-opus-best-pelican.png){ style="width: 100%;" }

<div style="margin: 0.5rem 0 0; max-width: none; width: 100%;"><em>Best candidate out of 20 (score: 0.817)</em></div>

</div>
</div>

??? example "Final optimized candidate by Gemini 3 Flash"
    ```xml
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 1280">
        <!-- Background: Sky and distant scenery -->
        <rect width="1280" height="1280" fill="#bbdefb"/>
        
        <!-- Stylized Sun -->
        <circle cx="1100" cy="180" r="80" fill="#fff9c4" opacity="0.8"/>
        
        <!-- Stylized Mountains/Hills for depth -->
        <path d="M0 900 L300 700 L600 850 L900 650 L1280 900 Z" fill="#90caf9" opacity="0.5"/>
        
        <!-- Road and Ground -->
        <rect x="0" y="900" width="1280" height="380" fill="#455a64"/>
        <rect x="0" y="900" width="1280" height="30" fill="#37474f"/>
        <line x1="0" y1="1120" x2="1280" y2="1120" stroke="white" stroke-width="12" stroke-dasharray="100,50" opacity="0.6"/>

        <!-- Bicycle Components -->
        <!-- Rear Wheel -->
        <g transform="translate(400, 950)">
            <circle r="170" fill="#212121"/>
            <circle r="145" fill="#f5f5f5"/>
            <g stroke="#bdbdbd" stroke-width="3">
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(0)"/>
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(30)"/>
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(60)"/>
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(90)"/>
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(120)"/>
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(150)"/>
            </g>
            <circle r="30" fill="#757575"/>
        </g>

        <!-- Front Wheel -->
        <g transform="translate(880, 950)">
            <circle r="170" fill="#212121"/>
            <circle r="145" fill="#f5f5f5"/>
            <g stroke="#bdbdbd" stroke-width="3">
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(15)"/>
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(45)"/>
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(75)"/>
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(105)"/>
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(135)"/>
            <line x1="-145" y1="0" x2="145" y2="0" transform="rotate(165)"/>
            </g>
            <circle r="30" fill="#757575"/>
        </g>

        <!-- Chain and Gear -->
        <circle cx="620" cy="950" r="50" fill="#263238"/>
        <path d="M400 930 L620 910 L620 990 L400 970 Z" fill="#37474f" opacity="0.9"/>

        <!-- Bicycle Frame (Red) -->
        <g stroke="#c62828" stroke-width="28" fill="none" stroke-linejoin="round" stroke-linecap="round">
            <line x1="400" y1="950" x2="620" y2="950"/> <!-- Chainstay -->
            <line x1="620" y1="950" x2="550" y2="620"/> <!-- Seat Tube -->
            <line x1="620" y1="950" x2="820" y2="660"/> <!-- Down Tube -->
            <line x1="560" y1="660" x2="820" y2="660"/> <!-- Top Tube -->
            <line x1="400" y1="950" x2="560" y2="660"/> <!-- Seat Stay -->
            <line x1="880" y1="950" x2="820" y2="660"/> <!-- Fork -->
        </g>

        <!-- Handlebars and Seat -->
        <path d="M820 660 L840 570 M780 580 L890 580" fill="none" stroke="#212121" stroke-width="22" stroke-linecap="round"/>
        <path d="M510 620 L600 620" stroke="#212121" stroke-width="35" stroke-linecap="round"/>

        <!-- Pelican (The Rider) -->
        <!-- Far Leg (hidden/subtle) -->
        <path d="M560 600 L660 880 L690 950" fill="none" stroke="#ef6c00" stroke-width="12" stroke-linecap="round" opacity="0.4"/>
        
        <!-- Pelican Body -->
        <ellipse cx="560" cy="480" rx="220" ry="140" fill="white" stroke="#eceff1" stroke-width="2"/>
        
        <!-- Neck and Head -->
        <path d="M730 460 Q850 460 830 250" fill="none" stroke="white" stroke-width="75" stroke-linecap="round"/>
        <circle cx="830" cy="250" r="60" fill="white"/>
        <circle cx="855" cy="230" r="8" fill="#212121"/>
        
        <!-- Beak (Characteristic pouch) -->
        <path d="M865 220 L1150 270 Q1100 500 865 380 Z" fill="#ffa726"/>
        <path d="M865 220 L1130 275" fill="none" stroke="#f57c00" stroke-width="5"/>
        
        <!-- Wing (Arm reaching for handlebars) -->
        <path d="M620 480 Q760 480 830 580" fill="none" stroke="white" stroke-width="50" stroke-linecap="round" stroke-linejoin="round"/>

        <!-- Near Leg (Pedaling) -->
        <g>
            <path d="M560 600 L620 950" fill="none" stroke="#fb8c00" stroke-width="24" stroke-linecap="round"/>
            <rect x="570" y="935" width="110" height="30" rx="10" fill="#212121"/>
        </g>

        <!-- Motion and Polish -->
        <g stroke="white" stroke-width="6" stroke-linecap="round" opacity="0.4">
            <line x1="200" y1="350" x2="50" y2="350"/>
            <line x1="250" y1="420" x2="80" y2="420"/>
            <line x1="180" y1="500" x2="40" y2="500"/>
        </g>
        <g stroke="#90a4ae" stroke-width="5" stroke-linecap="round" opacity="0.5">
            <line x1="1050" y1="980" x2="1180" y2="980"/>
            <line x1="1080" y1="1040" x2="1220" y2="1040"/>
        </g>
        </svg>
    ```


## Results

We apply `optimize_anything` to seven diverse domains spanning search, batch optimization, and generalization. Each result below links to the corresponding [appendix section](#appendix-case-study-code) with the full code.

### 1. Blackbox Mathematical Optimization: Matching Optuna

**Mode: Single-Task Search.** Given a blackbox objective function, `optimize_anything` discovers an optimization algorithm tailored to it — and matches [Optuna](https://optuna.org/) across the 56-problem [EvalSet](https://github.com/sigopt/evalset) benchmark.

<figure markdown="span" style="margin: 0 auto;">
  ![Bar chart showing GEPA vs Optuna on 56 EvalSet problems with 1% tolerance and 8000 trials. GEPA wins on 7 problems, Optuna wins on 9, and they tie on 40.](optuna_combined2.png)
  <figcaption>GEPA's optimize_anything matches Optuna, the industry-standard blackbox optimizer, on the EvalSet benchmark. (a) Over all 56 EvalSet problems (1 seed, 8 000 trials), GEPA ties Optuna on 40, wins 7, and loses 9. (b) On 10 selected problems where Optuna struggles (2 000 trials, 10 seeds), GEPA finds better solutions on 7 out of 10.</figcaption>
</figure>

On the 56-problem evalset benchmark with large budgets, GEPA and Optuna tie on most problems. But on the hardest problems with lower budgets where Optuna struggles, a striking pattern emerges: Optuna's fixed TPE→CMA-ES pipeline fails in predictable, structural ways. On McCourt13, all 10 Optuna seeds converge to the same local minimum because TPE's independent per-dimension sampling always falls into the dominant trap basin. On Tripod, CMA-ES assumes a smooth, unimodal landscape, but the objective is piecewise-linear with hard discontinuities — so it converges to the wrong basin and cannot escape.

GEPA tailors the solver to each problem by learning from accumulated evaluation history. For boundary optima, it discovers L-BFGS-B, a box-constrained optimizer that naturally sticks to boundaries. For deceptive traps, it evolves multi-start search from diverse starting points, escaping basins that trap single-trajectory methods. While Optuna tunes parameters within a fixed algorithm, GEPA learns to optimize the algorithm itself on the fly.

**Key result:** `optimize_anything` matches the performance of Optuna, a mature numerical optimizer, by evolving solver code from a random-search seed. [Full code →](#appendix-a-blackbox-mathematical-optimization)


### 2. Circle Packing: Outperforming AlphaEvolve

**Mode: Single-Task Search.** Pack n=26 circles to maximize the sum of their radii within a unit square. GEPA evolves the packing algorithm code, using execution results and geometric diagnostics as ASI.

<figure markdown="span">
  ![Left: line chart comparing GEPA, ShinkaEvolve, OpenEvolve, and AlphaEvolve on circle packing (n=26). GEPA reaches the highest score (~2.63598+) with fewer metric calls. Right: magnified view showing GEPA slightly above AlphaEvolve's best.](circle_packing_comparison.png)
  <figcaption>GEPA outperforms AlphaEvolve, ShinkaEvolve, and OpenEvolve on circle packing (n=26), reaching a higher score with fewer evaluations.</figcaption>
</figure>

<figure markdown="span">
  ![Three snapshots of circle packing optimization: Metric Call 0 (score=0.98, sparse circles), Metric Call 50 (score=2.61, tightly packed), Metric Call 89 (score=2.64, near-optimal packing).](circle_packing_trajectory.png)
  <figcaption>Visual progression of the circle packing optimization: from an initial naive arrangement to a near-optimal packing.</figcaption>
</figure>

**Key result:** GEPA outperforms all three open-source LLM-evolution frameworks, reaching a score of 2.63598+. [Full code →](#appendix-b-circle-packing)

### 3. CUDA Kernel Generation (KernelBench)

**Mode: Multi-Task Search.** Optimize a prompt that instructs an LLM to generate fast CUDA kernels for multiple algorithms from [KernelBench](https://github.com/ScalingIntelligence/KernelBench). Insights from optimizing one kernel transfer to others via shared prompt improvements.

<figure markdown="span">
  ![Line chart showing KernelBench performance vs budget. Fast_p(0) — any correct kernel — reaches 100%. Fast_p(1.0) — matching baseline speed — reaches 87%. Fast_p(1.1) — 10% faster — reaches 48%. Fast_p(1.2) — 20% faster — reaches 25%.](kernelbench_results.png)
  <figcaption>KernelBench results with GEPA (gpt-5 as proposer). 87% of generated kernels match or beat baseline performance; 25% are 20%+ faster.</figcaption>
</figure>

**Key result:** 87% of GEPA-generated kernels match or beat the baseline, with 25% achieving 20%+ speedups. [Full code →](#appendix-c-cuda-kernel-generation)

### 4. AI-Driven Systems Research: CloudCast & Can't Be Late

**Mode: Generalization.** We optimize cloud infrastructure algorithms: **CloudCast** discovers broadcast routing strategies for multi-cloud data transfer (minimizing egress cost), and **Can't Be Late** learns scheduling policies that decide when to use cheap-but-preemptible SPOT instances versus reliable ON_DEMAND instances to complete tasks before deadlines.

<div style="display: flex; align-items: flex-start; justify-content: center; gap: 1rem;" markdown>
<div style="flex: 1; text-align: center; min-width: 0;" markdown>

![Optimization trajectory for CloudCast showing cost savings (%) vs metric calls, achieving 40.2% test savings.](cloudcast_trajectory.png){ style="width: 100%;" }

<div style="margin: 0.5rem 0 0;"><em><strong>CloudCast (40.2% cost savings):</strong> Optimizes from baseline Dijkstra routing to a provider-aware Steiner tree algorithm with Pareto-frontier candidate selection.</em></div>

</div>
<div style="flex: 1; text-align: center; min-width: 0;" markdown>

![Optimization trajectory for Can't Be Late showing cost savings (%) vs metric calls, achieving 7.8% test savings.](cant_be_late_trajectory.png){ style="width: 100%;" }

<div style="margin: 0.5rem 0 0;"><em><strong>Can't Be Late (7.8% cost savings):</strong> Optimizes from a simple deadline-check heuristic to an adaptive scheduling strategy that tracks spot availability patterns and computes break-even switching costs.</em></div>

</div>
</div>

**Key result:** `optimize_anything` discovers state-of-the-art algorithms for both problems — **40.2% cost savings** on CloudCast and **7.8% cost savings** on Can't Be Late — outperforming hand-designed heuristics. [Full code →](#appendix-d-cloudcast--cant-be-late)

### 5. Prompt Optimization: AIME Mathematics

**Mode: Generalization.** Optimize a system prompt for solving [AIME](https://en.wikipedia.org/wiki/American_Invitational_Mathematics_Examination) 2025 math competition problems. The evaluator runs gpt-4.1-mini with the candidate prompt and scores whether the answer is correct. GEPA is the [state-of-the-art prompt optimization algorithm](https://gepa-ai.github.io/gepa/).

<figure markdown="span">
  ![Optimization trajectory for AIME 2025 with gpt-4.1-mini. Validation score improves from 46.67% to 57.78% over 350 metric calls. Best test score reaches 60.00%, up from a 46.67% baseline.](aime_results.png)
  <figcaption>AIME 2025 prompt optimization: gpt-4.1-mini accuracy improves from 46.67% to 60.00% through prompt refinement alone.</figcaption>
</figure>

**Key result:** Pure prompt optimization improves gpt-4.1-mini from 46.67% to **60.00%** on AIME 2025 — a 13.3 percentage point gain from changing only the system prompt. [Full code →](#appendix-e-aime-prompt-optimization)

### 6. Agent Architecture Discovery: ARC-AGI

**Mode: Generalization.** This is the most ambitious application. Rather than optimizing a prompt, we optimize the **entire agent**: its code, sub-agent architecture, control flow, helper functions, and prompts — all treated as a single text artifact. The seed is a 10-line naive agent; GEPA evolves it into a 300+ line system with rule induction, code verification, iterative refinement, and structured fallbacks.

<figure markdown="span">
  ![Optimization trajectory for ARC-AGI with Gemini 3 Flash. Validation accuracy improves from 56.5% to 93.5%. Base test score is 32.5%, best test score reaches 89.5%.](arc_agi_trajectory.png)
  <figcaption>ARC-AGI agent evolution: from a naive 10-line agent (32.5% test) to a sophisticated 300+ line system (89.5% test) with Gemini 3 Flash.</figcaption>
</figure>

**Key result:** Using the same underlying model (Gemini 3 Flash), `optimize_anything` improves ARC-AGI test accuracy from 32.5% to **89.5%** by evolving the entire agent architecture — gains that typically require weeks of manual iteration. [Full code →](#appendix-f-arc-agi-agent-architecture-discovery)

### 7. Coding Agent Skills: Learning Skills for Any Repository

**Mode: Generalization.** Skills — natural-language instructions and best practices for working with a specific codebase — are text artifacts too. `optimize_anything` can optimize them: the evaluator runs a coding agent on real tasks from the repository and scores whether it resolves them; the optimized skills must generalize to unseen tasks.

The results are striking: evolved skills boost resolve rates from 24% to **93%** on one repository and from 55% to **82%** on another — and transfer directly to Claude Code, pushing it to near-perfect pass rates while also reducing task duration.

**Key result:** `optimize_anything` learns repository-specific skills that dramatically improve coding agent performance and transfer across models. [Read the full post →](../learning-skills-for-any-repository/)


## Conclusion & Getting Started

`optimize_anything` makes a simple bet: if your artifact is text and your evaluator is programmatic, you can optimize it. The API is minimal — a seed, an evaluator, and optionally a dataset. The results span blackbox optimization, algorithmic discovery, kernel generation, systems research, prompt tuning, agent architecture search, and coding agent skill learning.

The key ideas: (1) **three unified modes** — single-task search, multi-task search, and generalization — under one declarative API; (2) **Actionable Side Information (ASI)** as a first-class API concept that turns the optimizer from a blind mutator into an intelligent designer; (3) **Pareto-efficient search** across metrics and examples that outperforms naive all-at-once optimization.

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
<!-- (@luke: the discord link leads to a dspy channle. is this correct? ) -->

---

## Appendix: Detailed Code Walkthroughs for each Case Study

<div class="appendix-tiles" markdown>

--8<-- "docs/blog/posts/2026-02-18-introducing-optimize-anything/appendix/appendix-a-blackbox-mathematical-optimization.snippet"

--8<-- "docs/blog/posts/2026-02-18-introducing-optimize-anything/appendix/appendix-b-circle-packing.snippet"

--8<-- "docs/blog/posts/2026-02-18-introducing-optimize-anything/appendix/appendix-c-cuda-kernel-generation.snippet"

--8<-- "docs/blog/posts/2026-02-18-introducing-optimize-anything/appendix/appendix-d-cloudcast-cant-be-late.snippet"

--8<-- "docs/blog/posts/2026-02-18-introducing-optimize-anything/appendix/appendix-e-aime-prompt-optimization.snippet"

--8<-- "docs/blog/posts/2026-02-18-introducing-optimize-anything/appendix/appendix-f-arc-agi-agent-architecture-discovery.snippet"

</div>
