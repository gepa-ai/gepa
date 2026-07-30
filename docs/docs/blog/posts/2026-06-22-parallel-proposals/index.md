---
date:
  created: 2026-06-30
authors:
 - jialin
 - lakshya
 - shangyin
 - donghyun
 - dan
 - koushik
 - alex
 - matei
equal_contribution:
  - "Jialin Zhang"
  - "Lakshya A Agrawal"
  - "Shangyin Tan"
  - "Donghyun Lee"
slug: parallel-proposals
readtime: 8
title: "Batching the Optimization Loop: Parallel Proposals in GEPA"
description: "GEPA now supports proposing and evaluating a batch of candidates on each optimization step instead of one candidate at a time. In our sweep on two tasks, most batched runs finished in half the wall-clock time or less, and the fastest in about a quarter to a third. Batch configurations also achieved higher held-out test scores: from 68.9% to 72.1% on LiveBench-Math (with 2×2) and from 49.0% to 60.0% on HoVer (with 8×1)."
social_image: blog/2026-06-22-parallel-proposals/images/throughput.png
citation_keywords: "text optimization, prompt optimization, program optimization, parallel proposals, batched inference, Pareto optimization, GEPA, LiveBench, HoVer, multi-hop retrieval"
---

# Batching the Optimization Loop: Parallel Proposals in GEPA

<figure markdown="span">
  ![Two scatter plots of held-out test performance against optimization wall-clock time. In each, a purple dot labeled Parallel Proposals (P×N) sits above and left of an orange diamond labeled Sequential, inside a shaded region of results that are both faster and better; a dashed line marks the unoptimized baseline, and two arrows from the diamond point up (Better than single mutation) and left (Faster than single mutation). LiveBench-Math: 71.6% in 2.5 hours against 68.9% in 7.7 hours. HoVer: 55.0% in 15 minutes against 49.0% in 47 minutes.](images/throughput.png){ style="width: 100%;" }
  <figcaption>Figure 1. At the same metric-call budget, parallel proposals are faster and better than single mutation.</figcaption>
</figure>

Running GEPA on a task can take hours because each optimization step waits for a proposal and its evaluation before the next step begins. The loop samples a parent, proposes a mutation, evaluates it on a mini-batch, and, if it improves on its parent, evaluates it on the full validation set.

This release adds batched parallel proposals. Instead of advancing one proposal at a time, a step can propose several candidates and dispatch their evaluations concurrently. In our experiments, this reduced wall-clock time substantially. We found that most batch configurations also achieved higher held-out test scores.

## How parallel proposals work

On each step, GEPA now samples several parents from its Pareto frontier, draws several reflective mutations of each, and scores all of the proposals concurrently.

<figure markdown="span">
  ![Diagram: sample P (=3) parent nodes at once; each parent spawns N (=4) child mutations from its own mini-batch sample, giving P×N children; all P·N children are sent as one batch to a "Parallel evaluator" that scores them together.](images/pxn_diagram.svg){ style="width: 100%;" }
  <figcaption>Figure 2. One batched iteration. GEPA samples P parents from the frontier, draws N reflective mutations for each parent, and scores all P·N children in one parallel evaluation. This lets GEPA propose more candidates in each iteration while paying the iteration latency once.</figcaption>
</figure>

??? note "One P×N step in detail"

    In the new P×N sampling strategy, one GEPA step:

    1. samples P parents from the current Pareto frontier;
    2. draws N mini-batches per parent and evaluates each parent on them through `batch_evaluate()`, producing P·N reflection requests;
    3. dispatches the reflection requests concurrently;
    4. screens the proposals by batch-evaluating them again on their mini-batches;
    5. evaluates accepted candidates on the full validation set in parallel, then updates the frontier.

The mechanism is closely analogous to batch Bayesian optimization[^batchopt], which proposes and evaluates a batch of candidates per round rather than adapting after every single one. Similar to how GEPA uses a Pareto frontier instead of a single best numerical score to select candidates, parallel proposals push the idea further by drawing several extensions of the frontier within each step, thus reducing how often the search adapts to the validation set. [Prior work](https://www.science.org/doi/10.1126/science.aaa9375) also shows that repeatedly steering decisions with one fixed holdout can inflate its apparent performance, so committing to more proposals simultaneously should transfer better beyond the validation set.

## Results

We evaluated parallel proposals on [LiveBench-Math](https://livebench.ai/) and [HoVer](https://hover-nlp.github.io/), where the optimized prompt or program is selected using a validation set and then measured on a held-out test set. For both tasks, we used `gpt-5-mini` as the proposer. As in standard GEPA runs, we measured the optimization budget by the number of metric calls. One metric call corresponds to evaluating one candidate on one example. Every setting on a task received the same total metric-call budget.

??? note "Task setup details"

    - **[LiveBench-Math](https://livebench.ai/)** asks a model to solve competition math problems (AMC and AIME questions, symbolic algebra, and olympiad problems), graded by LiveBench's own scorers, with the [Terrarium](https://github.com/gepa-ai/terrarium) split of 100 training, 100 validation, and 168 test problems. Budget: 5,000 metric calls, each one `gpt-4.1-mini` solution attempt.
    - **[HoVer](https://hover-nlp.github.io/)** asks a system to gather the Wikipedia pages needed to verify a multi-hop claim. We optimize the two prompts (a query writer and a note taker) of a four-hop `gpt-4.1-mini` retrieval program over a BM25 index of 5.2 million 2017 Wikipedia abstracts, on three-hop claims split into 200 training, 150 validation, and 200 test claims; one rollout makes about eight calls. During optimization, GEPA scores each rollout by the fraction of the claim's three gold pages that appear in the retrieved pages; the reported headline metric is the strict version, the share of test claims with all three pages retrieved. Budget: 3,000 metric calls, each one full program rollout.

### Runtime

A run's wall-clock time is the sum of the time spent on each iteration. With $k = P \cdot N$ proposals per step, every iteration incurs a step latency $L_{\text{step}}$ for the $k$ reflection calls and $k$ mini-batch evaluations, which run concurrently. Whenever one or more proposals beat their parent on the mini-batch, the iteration additionally pays a full-validation latency $L_{\text{val}}$ to evaluate all accepted proposals on the validation in parallel. 

The metric-call budget determines the total number of proposals a run can afford, whether parallel or sequential (if we assume the candidate acceptance rate stays the same). The number of iterations is the number of proposals divided by $k$, so a width-$k$ run needs about $1/k$ as many iterations as single mutation. The main bottleneck is full validation. If each proposal is accepted with probability $a$, and proposal outcomes are independent, then an iteration triggers full validation with probability

$$q_k = 1 - (1-a)^k,$$

which grows with $k$. The resulting speedup of a width-$k$ run over single mutation is approximately

$$\frac{T(1)}{T(k)} \approx \frac{k\,(L_{\text{step}} + a\,L_{\text{val}})}{L_{\text{step}} + q_k\,L_{\text{val}}}.$$

In practice, two effects slow wide steps down. First, the reflection stage takes as long as the slowest of its $k$ concurrent calls, so $L_{\text{step}}$ grows with width. Second, evaluation is limited by the worker pool. A validation stage can carry up to $k$ accepted candidates, each evaluated on all $V$ validation examples, so with $W$ concurrent workers and a per-rollout latency of $T_e$,

$$L_{\text{val}} \approx \begin{cases} T_e & \text{if } kV \le W, \\ (kV/W)\,T_e & \text{if } kV > W. \end{cases}$$

According to strong scaling[^scaling], a run with a budget of $B$ metric calls needs at least $B \cdot T_e / W$ of wall-clock for evaluation alone. This is a hard limit for runtime, so we recommend not scaling $P \cdot N$ further once the runtime approaches it.

The measured runs align with this model. For example, on LiveBench-Math, moving from single mutation to 2×2 cut the number of iterations by 4.9× (219 to 45), but the fraction of iterations that triggered full validation rose from 17% to 53%, so the run gained a 1.9× speedup (7.7 to 4.1 hours) rather than the full 4.9×. Across the whole sweep, Figure 3 shows the measured runtimes tracking the model's predicted curve.
<figure markdown="span">
  ![Two dual-axis line charts across the nine settings from single to 8×2: an orange line with held-out test performance on the left axis, a purple line with optimization time on the right axis, a dashed lighter-purple curve with the optimization time predicted by the finite-worker model, and a dotted baseline. Measured time falls from 7.7 hours to about 2 on LiveBench-Math and from 47 to 14 minutes on HoVer, and the predicted curve tracks it, flattening near 2.2 hours and 15 minutes. Test performance ranges from 66.7 to 72.1 on LiveBench-Math and from 49.0 to 60.0 on HoVer against single mutation's 68.9 and 49.0.](images/scaling_lines.png){ style="width: 100%;" }
  <figcaption>Figure 3. As the per-step width P·N scales up, runtime falls with diminishing returns, following the optimization time predicted by our runtime model. Most settings perform as well as or better than single mutation on test, and the best setting scores much higher.</figcaption>
</figure>

### Performance

In principle, larger P extends more members of the frontier at once, which should help when no single generally good candidate exists and the frontier holds genuinely different specialists worth advancing in parallel, and larger N draws more mutations with different mini-batches, which should help when the dataset is rich enough to expose many distinct directions to improve one candidate. LiveBench-Math is the second case, with problems spanning multiple areas, so giving each parent several mutations (larger N) transferred better to test than spreading single mutations across more parents (larger P). HoVer additionally keeps a more complementary candidate pool, where different candidates succeed on different claims, and its test scores tend to grow with both P and N. By holding P=2 and comparing different N, test scores rise from 68.6 at 2×1 to 71.6-72.1 at 2×2 through 2×8 on LiveBench-Math, and from 52.5 to 55.0 on HoVer. Based on the results, we suggest scaling N first when in doubt.

### Budget Efficiency

We also probed how efficiently each setting spends its budget during the run. Here we focus on one N-scaled and one P-scaled setting at the same width, 2×4 and 8×1, against single mutation on both tasks.

#### Generalization Gap

On the held-out test sets, 2×4 won 3.0pp over single mutation on LiveBench-Math, and 8×1 won 11.0pp on HoVer. On LiveBench-Math, single mutation actually scored higher on the validation set, but the batched settings transferred better to the test set.

The validation-to-test drop on LiveBench-Math is an overfitting signal. Single mutation adapts after every proposal, steering 219 rounds of feedback against the same 100 validation problems, so a long run can fit their quirks, ultimately resulting in worse transfer from validation set to test set (dropped nine points from validation to test). The 2×4 run spent the same budget in 28 rounds and dropped only two points. 8×1 on LiveBench-Math shows the opposite pattern, a high validation curve with weak transfer, consistent with the P-heavy behavior described above.

Batched settings dominate small budgets: on LiveBench-Math both width-8 settings reach validation scores that single mutation needs about 2,000 calls to match, and on HoVer they lead at every budget. Single mutation's late overtake on LiveBench-Math is the overfitting pattern above, validation gains that do not transfer.

<figure markdown="span">
  ![Two step charts of best validation score against metric calls consumed, for single mutation, 2×4, and 8×1, with stars marking held-out test scores. On LiveBench-Math, 8×1 reaches 0.738 within about 450 calls and 2×4 reaches 0.740 by about 1,400, while single mutation overtakes on validation at about 2,000 calls and ends at 0.783; test stars are 71.9% for 2×4, 69.6% for 8×1, and 68.9% for single mutation. On HoVer, both batched settings lead single mutation at every budget, ending at 0.727 (8×1) and 0.716 (2×4) against 0.709; test recall stars are 0.815, 0.767, and 0.760.](images/budget_pareto.png){ style="width: 100%;" }
  <figcaption>Figure 4. Best validation quality against metric calls consumed, for single mutation and the width-8 pair 2×4 and 8×1. The best setting changes with the budget, and batched settings dominate small budgets.</figcaption>
</figure>

#### Dollar Cost

By measuring the total LLM spend (sum of solver calls and reflection calls), we found that batched settings are more cost-efficient, reaching strong validation scores within the first few dollars. On LiveBench-Math, 8×1 passes a 0.73 validation score within about $2.5 of spend, and on HoVer it reaches its selected program for $2.65, a fraction of single mutation's $14.2 run. The total spend varies with how long each run's candidates make the solver's outputs (per-call output tokens order the totals on both tasks), but stays comparable across settings, in the $13 to $18 range.

<figure markdown="span">
  ![Two Pareto curves of best validation quality against cumulative LLM dollars for single mutation, 2×4, and 8×1. Left, LiveBench-Math: single mutation reaches 0.783 by about $5.5 and ends at $13.2; 2×4 ends at 0.740 for $12.7 and 8×1 at 0.760 for $18.1. Right, HoVer: 8×1 jumps to 0.727 by $2.65 and ends at $15.3; 2×4 ends at 0.716 for $18.1 and single mutation at 0.709 for $13.4. Test stars match Figure 4.](images/pareto_cost.png){ style="width: 100%;" }
  <figcaption>Figure 5. Best validation quality against total LLM spend, solver and reflection calls combined. Batched settings reach strong validation quality within the first few dollars.</figcaption>
</figure>

## Getting started

Parallel proposals are available in [gepa](https://github.com/gepa-ai/gepa). Opt in with a simple setting change below. The sampling strategy says how many candidates to propose per step, and the selection strategy says which of the improved candidates to keep. For example, two parents with two mutations each gives four candidates per step.

```python
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig, ReflectionConfig
from gepa.strategies.proposal_sampling import PxNSampling
from gepa.strategies.proposal_selection import AllImprovements

config = GEPAConfig(
    engine=EngineConfig(
        sampling_strategy=PxNSampling(p=2, n=2),   # 2 parents, 2 mutations each = 4 per step
        selection_strategy=AllImprovements(),
    ),
    reflection=ReflectionConfig(reflection_lm="gpt-5-mini"),
)

result = optimize_anything(
    seed_candidate=seed, evaluator=evaluate,
    dataset=trainset, valset=valset, objective=objective, config=config,
)
```

GEPA by default calls your `evaluate` function in parallel, so all you need is to set the maximum number of workers. Optionally, you may provide a custom `batch_evaluate` function to the `GEPAAdapter` (or pass it as the `batch_evaluator` argument to the `optimize_anything` API). You may choose or define other sampling and selection strategies; see the [API reference](https://gepa-ai.github.io/gepa/api/) for the full list.

## Notes

The default is still single mutation, matching GEPA's earlier behavior, so existing runs do not change.

[^batchopt]: David Ginsbourger, Rodolphe Le Riche, and Laurent Carraro, "[Kriging is well-suited to parallelize optimization](https://link.springer.com/chapter/10.1007/978-3-642-10701-6_6)," 2010.
[^scaling]: Gene M. Amdahl, "[Validity of the single processor approach to achieving large scale computing capabilities](https://dl.acm.org/doi/10.1145/1465482.1465560)," AFIPS 1967.