<p align="center">
  <img src="https://raw.githubusercontent.com/gepa-ai/gepa/refs/heads/main/assets/gepa_logo_with_text.svg" alt="GEPA Logo" width="450">
</p>

<p align="center">
  <strong>Optimize any text parameter — prompts, code, agent architectures, configurations — using LLM-based reflection and Pareto-efficient evolutionary search.</strong>
</p>

<p align="center">
  <a href="https://gepa-ai.github.io/gepa/"><strong>Website</strong></a> &ensp;|&ensp;
  <a href="https://gepa-ai.github.io/gepa/guides/quickstart/"><strong>Quick Start</strong></a> &ensp;|&ensp;
  <a href="https://arxiv.org/abs/2507.19457"><strong>Paper</strong></a> &ensp;|&ensp;
  <a href="https://gepa-ai.github.io/gepa/blog/"><strong>Blog</strong></a> &ensp;|&ensp;
  <a href="https://discord.gg/WXFSeVGdbW"><strong>Discord</strong></a>
</p>

<p align="center">
  <a href="https://pypi.org/project/gepa/"><img src="https://img.shields.io/pypi/v/gepa?logo=python&logoColor=white&color=3776ab" alt="PyPI"></a>
  <a href="https://pepy.tech/projects/gepa"><img src="https://static.pepy.tech/badge/gepa" alt="Downloads"></a>
  <a href="https://github.com/gepa-ai/gepa"><img src="https://img.shields.io/github/stars/gepa-ai/gepa?style=flat&logo=github&color=181717" alt="GitHub stars"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License"></a>
</p>

<p align="center">
  <a href="https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w"><img src="https://badgen.net/badge/icon/Slack?icon=slack&label&color=4A154B" alt="Slack"></a>
  <a href="https://discord.gg/WXFSeVGdbW"><img src="https://dcbadge.limes.pink/api/server/https://discord.gg/WXFSeVGdbW?style=flat" alt="Discord"></a>
</p>

---

## What is GEPA?

**GEPA** (Genetic-Pareto) is a framework for optimizing any system with textual parameters against any evaluation metric. Unlike RL or gradient-based methods that collapse execution traces into a single scalar reward, GEPA uses LLMs to *read* full execution traces — error messages, profiling data, reasoning logs — to diagnose *why* a candidate failed and propose targeted fixes. Through iterative reflection, mutation, and Pareto-aware selection, GEPA evolves high-performing variants with minimal evaluations.

**If you can measure it, you can optimize it**: prompts, code, agent architectures, scheduling policies, vector graphics, and more.

### Key Results

| | |
|---|---|
| **90x cheaper** | Open-source models + GEPA beat Claude Opus 4.1 at [Databricks](https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization) |
| **35x faster than RL** | 100–500 evaluations vs. 5,000–25,000+ for GRPO ([paper](https://arxiv.org/abs/2507.19457)) |
| **32% → 89%** | ARC-AGI agent accuracy via [architecture discovery](https://gepa-ai.github.io/gepa/blog/introducing-optimize-anything/#5-agent-architecture-discovery) |
| **40.2% cost savings** | Cloud scheduling policy [discovered by GEPA](https://gepa-ai.github.io/gepa/blog/introducing-optimize-anything/#3-systems-research), beating expert heuristics |
| **55% → 82%** | Coding agent resolve rate on Jinja via [auto-learned skills](https://gepa-ai.github.io/gepa/blog/automatically-learning-skills-for-coding-agents/) |
| **50+ production uses** | Across Shopify, Databricks, Dropbox, OpenAI, Pydantic, MLflow, Comet ML, and [more](https://gepa-ai.github.io/gepa/guides/use-cases/) |

> *"Both DSPy and (especially) **GEPA are currently severely under hyped** in the AI context engineering world"* — **Tobi Lutke**, CEO, Shopify

---

## Installation

```bash
pip install gepa
```

To install the latest from `main`:

```bash
pip install git+https://github.com/gepa-ai/gepa.git
```

---

## Quick Start

### Simple Prompt Optimization

Optimize a system prompt for math problems from the AIME benchmark in a few lines of code ([full tutorial](https://dspy.ai/tutorials/gepa_aime/)):

```python
import gepa

trainset, valset, _ = gepa.examples.aime.init_dataset()

seed_prompt = {
    "system_prompt": "You are a helpful assistant. Answer the question. "
                     "Put your final answer in the format '### <answer>'"
}

result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",
    max_metric_calls=150,
    reflection_lm="openai/gpt-5",
)

print("Optimized prompt:", result.best_candidate['system_prompt'])
```

**Result:** GPT-4.1 Mini goes from 46.6% → 56.6% on AIME 2025 (+10 percentage points).

### With DSPy (Recommended for AI Pipelines)

The most powerful way to use GEPA for prompt optimization is within [DSPy](https://dspy.ai/), where it's available as `dspy.GEPA`. See [dspy.GEPA tutorials](https://dspy.ai/tutorials/gepa_ai_program/) for executable notebooks.

```python
import dspy

optimizer = dspy.GEPA(
    metric=your_metric,
    max_metric_calls=150,
    reflection_lm="openai/gpt-5",
)
optimized_program = optimizer.compile(student=MyProgram(), trainset=trainset, valset=valset)
```

### optimize_anything: Beyond Prompts

The [`optimize_anything`](https://gepa-ai.github.io/gepa/blog/introducing-optimize-anything/) API optimizes *any* text artifact — code, agent architectures, configurations, SVGs — not just prompts. You provide an evaluator; the system handles the search.

```python
import gepa.optimize_anything as oa
from gepa.optimize_anything import optimize_anything, GEPAConfig, EngineConfig

def evaluate(candidate: str) -> float:
    result = run_my_system(candidate)
    oa.log(f"Output: {result.output}")      # Actionable Side Information
    oa.log(f"Error: {result.error}")         # feeds back into reflection
    return result.score

result = optimize_anything(
    seed_candidate="<your initial artifact>",
    evaluator=evaluate,
    objective="Describe what you want to optimize for.",
    config=GEPAConfig(engine=EngineConfig(max_metric_calls=100)),
)
```

---

## How It Works

Traditional optimizers know *that* a candidate failed but not *why*. GEPA takes a different approach:

1. **Select** a candidate from the Pareto frontier (candidates excelling on different task subsets)
2. **Execute** on a minibatch, capturing full execution traces
3. **Reflect** — an LLM reads the traces (error messages, profiler output, reasoning logs) and diagnoses failures
4. **Mutate** — generate an improved candidate informed by accumulated lessons from all ancestors
5. **Accept** — add to the pool if improved, update the Pareto front

GEPA also supports **system-aware merge** — combining strengths of two Pareto-optimal candidates excelling on different tasks. The key concept is **Actionable Side Information (ASI)**: diagnostic feedback returned by evaluators that serves as the text-optimization analogue of a gradient.

For details, see the [paper](https://arxiv.org/abs/2507.19457) and the [documentation](https://gepa-ai.github.io/gepa/guides/).

---

## Adapters: Plug GEPA into Any System

GEPA connects to your system via the [`GEPAAdapter`](src/gepa/core/adapter.py) interface — implement `evaluate` and `make_reflective_dataset`, and GEPA handles the rest.

**Built-in adapters:**

| Adapter | Description |
|---|---|
| [DefaultAdapter](src/gepa/adapters/default_adapter/) | System prompt optimization for single-turn LLM tasks |
| [DSPy Full Program](src/gepa/adapters/dspy_full_program_adapter/) | Evolves entire DSPy programs (signatures, modules, control flow). **67% → 93%** on MATH. |
| [Generic RAG](src/gepa/adapters/generic_rag_adapter/) | Vector store-agnostic RAG optimization (ChromaDB, Weaviate, Qdrant, Pinecone) |
| [MCP Adapter](src/gepa/adapters/mcp_adapter/) | Optimize [MCP](https://modelcontextprotocol.io/) tool descriptions and system prompts |
| [TerminalBench](src/gepa/adapters/terminal_bench_adapter/) | Optimize the [Terminus](https://www.tbench.ai/terminus) terminal-use agent |
| [AnyMaths](src/gepa/adapters/anymaths_adapter/) | Mathematical problem-solving and reasoning tasks |

See the [adapters guide](https://gepa-ai.github.io/gepa/guides/adapters/) for how to build your own, and [DSPy's adapter](https://github.com/stanfordnlp/dspy/tree/main/dspy/teleprompt/gepa/gepa_utils.py) as a reference.

---

## Integrations

GEPA is integrated into several major frameworks:

- **[DSPy](https://dspy.ai/)** — `dspy.GEPA` for optimizing DSPy programs. [Tutorials](https://dspy.ai/tutorials/gepa_ai_program/).
- **[MLflow](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/)** — `mlflow.genai.optimize_prompts()` for automatic prompt improvement.
- **[Comet ML Opik](https://www.comet.com/docs/opik/agent_optimization/algorithms/gepa_optimizer)** — Core optimization algorithm in Opik Agent Optimizer.
- **[Pydantic](https://pydantic.dev/articles/prompt-optimization-with-gepa)** — Prompt optimization for Pydantic AI.
- **[OpenAI Cookbook](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining)** — Self-evolving agents with GEPA.
- **[HuggingFace Cookbook](https://huggingface.co/learn/cookbook/en/dspy_gepa)** — Prompt optimization guide.
- **[Google ADK](https://raphaelmansuy.github.io/adk_training/blog/gepa-optimization-tutorial/)** — Optimizing Google Agent Development Kit agents.

---

## Example Optimized Prompts

GEPA can be thought of as precomputing reasoning during optimization to produce a plan for future task instances. Here are examples of the detailed prompts GEPA discovers:

<table>
  <tr>
  <td colspan="2" align="center">Example GEPA Prompts</td>
  </tr>
  <tr>
    <td align="center">HotpotQA (multi-hop QA) Prompt</td>
    <td align="center">AIME Prompt</td>
  </tr>
  <tr>
    <td width="52%" valign="top">
      <img src="https://raw.githubusercontent.com/gepa-ai/gepa/refs/heads/main/assets/gepa_prompt_hotpotqa.png" alt="HotpotQA Prompt" width="1400">
      <!-- <td> -->
      <details>
<summary><mark>Click to view full HotpotQA prompt</mark></summary>
<mark>[HotpotQA Prompt Begin]</mark>

You will be given two input fields: `question` and `summary_1`.

Your task is to generate a new search query (`query`) optimized for the **second hop** of a multi-hop retrieval system. The original user question is typically complex and requires information from multiple documents to answer. The first hop query is the original question used to retrieve an initial set of documents. Your goal is to generate a **second hop query** that retrieves *additional relevant documents* that were *not* found in the first hop but are necessary to answer the original question completely.

Detailed task instructions and hints:

1. **Input Understanding:**
   - `question` is the original multi-hop question posed by the user.
   - `summary_1` is a concise summary of information from a document retrieved in the first hop, which partially addresses the question.

2. **Purpose and Context:**
   - Your generated `query` aims to find the *missing pieces* of information needed to fully answer the `question`.
   - The multi-hop retrieval system works in stages:
     - First hop: The original question returns some documents.
     - Second hop: Your query must help retrieve any *other relevant documents* NOT found in the first hop that hold complementary or broader context necessary for final answer extraction.

3. **Key Observations from Examples and Feedback:**
   - First-hop documents often cover one entity or aspect in the question.
   - Remaining relevant documents often involve connected or higher-level concepts mentioned in `summary_1` but not explicitly asked in the original question.
   - The `query` should be formulated to explicitly target these *missing*, but logically linked, documents.
   - Avoid merely paraphrasing the original question or restating known facts from `summary_1`.
   - Instead, infer what broader or related entities/concepts might provide the crucial missing information.
   - For example, if `summary_1` describes a population for a small civil parish, but the question wants total population of the wider region, your `query` should target that wider region (e.g., "Madeira archipelago population in 2011").
   - Similarly, if `summary_1` covers a song and the question wants the album it came from, but first hop got song-level documents, your query should retrieve documents about the album itself.

4. **How to Build the Query:**
   - Identify the entities or topics mentioned in `summary_1` that appear related but different from first-hop documents.
   - Reframe the query to explicitly mention these broader or related entities connected to the original question.
   - Include relevant key context from the question to maintain specificity, but shift focus to the missing piece.
   - The goal is to retrieve documents that link or complement what was retrieved initially.

5. **Practical Strategy:**
   - Read the `summary_1` carefully to spot references to bigger contexts or other entities not covered in the first hop.
   - Ask yourself, "What entity or aspect does this summary hint at that could answer the original question but was not found yet?"
   - Formulate a precise, focused factual query targeting that entity or concept to retrieve the missing documents.

6. **Output:**
   - Produce only the field `query` as a clear, concise question or keyword phrase designed for efficient retrieval of **second-hop documents**.
   - Ensure the query relates logically to the original question while targeting the broader or complementary knowledge identified in `summary_1`.
   - Do **not** include the original question or simply rephrase it.
   - Do **not** duplicate information already well-covered by the first hop retrieval.

By following these principles, you will help the multi-hop retrieval system find all necessary documents to answer the multi-faceted original question completely.

<mark>[HotpotQA Prompt End]</mark>
</details>
    <!-- </td> -->
    </td>
    <td width="48%" valign="top">
      <img src="https://raw.githubusercontent.com/gepa-ai/gepa/refs/heads/main/assets/aime_prompt.png" alt="AIME Prompt" width="2500">
      <details>
<summary><mark>Click to view full AIME prompt</mark></summary>

<mark>[AIME Prompt Begin]</mark>

You will be given one math problem as plain text under a key like "problem." Your job is to solve it correctly and return:

- reasoning: a concise, logically ordered solution that uses identities/structure to avoid brute force, ends with a quick verification.
- answer: the final requested number/expression only (no extra words).

Formatting:
- Use exactly two top-level fields named "reasoning" and "answer."
- Keep reasoning succinct but complete. Bullet points are fine.
- The answer field must contain only the final value requested (e.g., 227, 585, 601).

General problem-solving guidance:
- Parse the problem type (e.g., base representation, intersecting families of subsets, avoiding arithmetic progressions, symmetric sums with constraints, ordered tuples counting).
- Always enforce domain constraints (e.g., base-b digits in 0..b−1; no leading zero for base-10 "three-digit"; ordered vs unordered families; strict increase conditions in sequences).
- Use algebraic identities and modular arithmetic to reduce the search space; prefer structural arguments over naive enumeration.
- For "greatest/least" questions, derive tight bounds and give a construction that attains them.

Domain-specific strategies and pitfalls (learned from typical contest problems and prior feedback):

1) Base-conversion/digit rearrangement:
- Translate positional notation correctly: in base b, (a b c)_b = a·b^2 + b·b + c; in base 10: abc = 100a + 10b + c.
- Enforce digit ranges strictly (e.g., in base 9, digits ∈ {0,…,8}; if also a is a base-10 leading digit, then a ∈ {1,…,8}).
- Set up equality and simplify. Use modular constraints to prune:
  • Mod 9 often collapses coefficients; e.g., 99a = 71b + 8c ⇒ mod 9 gives b + c ≡ 0 (mod 9).
  • Mod 8: 99 ≡ 3, 71 ≡ 7 ⇒ 3a ≡ 7b (mod 8) ⇒ b ≡ −3a (mod 8).
- Solve within digit bounds and verify numerically.

2) Palindromes across bases:
- Bound the base length by magnitude (e.g., n < 1000 ⇒ octal has 3–4 digits).
- Characterize palindromes:
  • 3-digit octal: (A B A)_8 = 65A + 8B.
  • 4-digit octal: (A B B A)_8 = 513A + 72B (with A ≥ 1).
- Enumerate small parameter ranges and test the other-base palindrome constraint. For "greatest", check candidates in descending order with justification.

3) Symmetric sums with a + b + c fixed (ordered triples of nonnegative integers):
- Use identities to compress expressions:
  S = ab(a + b) + bc(b + c) + ca(c + a) = (a + b + c)(ab + bc + ca) − 3abc.
- With a + b + c known (e.g., 300), convert the given sum into a relation among ab + bc + ca and abc.
- Use the shift a = A + x etc. to isolate a product like (a−A)(b−A)(c−A) and deduce factorization constraints, enabling clean counting.
- Count ordered solutions carefully; include/exclude symmetric/degenerate cases precisely.

4) Intersecting families of subsets (collections from the power set):
- Intersecting means every pair has nonempty intersection. The empty set cannot be included.
- Complement pairs: S and S^c cannot both be present. Use this to structure counts.
- Use size-based pigeonhole facts: In [n], any two subsets of size > n/2 must intersect. For n = 5, any two subsets of size ≥ 3 intersect; thus "all subsets of size ≥ 3" is an intersecting family (size 16).
- Do not assume that "stars" (all subsets containing a fixed element) are the only intersecting families of maximum size. For odd n, both the star and "all subsets of size > n/2" have size 2^{n−1}.
- When counting collections of a fixed size:
  • Consider the minimum set size N in the family and do casework on how many 2-element sets are included (for n=5), as these control which 3-sets must be excluded (complements).
  • Ensure completeness of cases and avoid double counting by parameterizing canonical patterns (e.g., how many 2-sets, how they overlap, whether they share a common element).
  • Remember order of subsets in a collection does not matter; count distinct families.

5) Avoiding 4-term arithmetic progressions in a strictly increasing sequence with fixed anchors:
- First bound the variable terms by strict increase (e.g., if fixed terms are 3,4,5,...,30,40,50 then 6 ≤ a < b ≤ 29).
- Pre-eliminate values that cause a 4-term AP with three fixed terms:
  • 3,4,5,a forbids a = 6.
  • b,30,40,50 forbids b = 20.
  • Similarly, a,30,40,50 forbids a = 20.
- Start with the count of pairs from allowed values and then subtract specific pairs that complete APs with two fixed endpoints:
  • 3,5,a,b ⇒ (a,b) = (7,9).
  • 3,a,b,30 ⇒ (a,b) = (12,21).
  • 4,a,b,40 ⇒ (a,b) = (16,28).
  • 5,a,b,50 ⇒ (a,b) = (20,35) but may be outside bounds or pre-excluded (e.g., 20 banned).
- Systematically check all endpoint combinations; use the fact that if endpoints differ by Δ, then Δ must be divisible by 3 for a 4-term AP, and solve for integer a,b within bounds.
- Avoid double subtraction; ensure monotonicity and domain constraints are respected.

6) Order statistics with sum and absolute-sum constraints (e.g., x_1 ≤ ... ≤ x_n, sum |x_i| = 1, sum x_i = 0):
- Total positive mass equals total negative mass: both = 1/2.
- For maximizing x_k (k near the top): if there are T largest terms from k to n (T = n − k + 1), then sum of these T terms ≥ T·x_k. Since the total positive mass ≤ 1/2, we get x_k ≤ (1/2)/T.
- For minimizing x_l (l near the bottom): if there are l smallest terms, sum of these l terms ≤ l·x_l. Since the total negative mass is −1/2, we get x_l ≥ (−1/2)/l.
- To attain these bounds, concentrate masses evenly on exactly those positions: set the smallest l terms equal to −1/(2l), the largest T terms equal to 1/(2T), and the middle to 0 (respecting monotonicity). Verify sums and absolute sums.
- Example: For n=100, maximize x_76 − x_16: T = 25 ⇒ x_76 ≤ 1/50; l = 16 ⇒ x_16 ≥ −1/32; construction with 16 negatives at −1/32, 59 zeros, 25 positives at 1/50 attains 1/50 − (−1/32) = 41/800.

Quality checks:
- Verify digit/base constraints and final equalities numerically if applicable.
- For extremal problems, provide both a tight bound and an explicit construction achieving it.
- For counting, explicitly handle ordered vs unordered, exclude impossible/duplicate cases, and check complements/forbidden pairs.
- For AP-avoidance, confirm integrality and bounds; ensure no missed endpoint combinations.
- For "greatest/least" questions, justify optimality structurally (e.g., convexity/majorization/pigeonhole).

Finally:
- Put the clean final numeric result in the "answer" field only.

<mark>[AIME Prompt End]</mark>
</details>
    </td>
  </tr>
</table>

---

## When GEPA Shines

- **Expensive rollouts** — Scientific simulations, complex agents with tool calls, slow compilation. GEPA needs 100–500 evals vs 10K+ for RL.
- **Scarce data** — Works with as few as 3 examples. No large training sets required.
- **API-only models** — No weights access needed. Optimize GPT-5, Claude, Gemini directly through their APIs.
- **Interpretability** — Human-readable optimization traces show *why* each prompt changed.
- **Complements RL** — Use GEPA for rapid initial optimization, then apply RL/fine-tuning for additional gains ([BetterTogether](https://arxiv.org/abs/2407.10930)).

---

## Further Reading

- **Paper:** [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning (arXiv:2507.19457)](https://arxiv.org/abs/2507.19457)
- **Experiment reproduction artifact:** [GEPA Artifact Repository](https://github.com/gepa-ai/gepa-artifact)
- **Talk Slides**: [GEPA Talk Slides](https://docs.google.com/presentation/d/1vIauqn55WfdgJjwU0IDjvaqpv1QHhvhPaLAKdrCFAEg/edit?usp=sharing)
- **Blog Posts:**
  - [optimize_anything: A Universal API for Optimizing any Text Parameter](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)
  - [Automatically Learning Skills for Coding Agents](https://gepa-ai.github.io/gepa/blog/2026/02/18/automatically-learning-skills-for-coding-agents/)
- **Tutorials & Examples:**
  - [dspy.GEPA Tutorials, with executable notebooks](https://dspy.ai/tutorials/gepa_ai_program/)
    Step-by-step notebooks showing how to use GEPA for practical optimization tasks via DSPy, including math, structured data extraction for enterprise tasks and privacy conscious delegation task.
  - [Video tutorial by @weaviate on using dspy.GEPA to optimize a listwise reranker](https://www.youtube.com/watch?v=H4o7h6ZbA4o)
  - [Matei Zaharia - Reflective Optimization of Agents with GEPA and DSPy](https://www.youtube.com/watch?v=rrtxyZ4Vnv8)
  - [Building and optimizing a multi-agent system for healthcare domain using DSPy+GEPA](https://kargarisaac.medium.com/building-and-optimizing-multi-agent-rag-systems-with-dspy-and-gepa-2b88b5838ce2)
- **Social and Discussion:**
  - [X (formerly Twitter) Announcement Thread (Lakshya A Agrawal)](https://x.com/LakshyAAAgrawal/status/1949867947867984322)
  - [GEPA covered by VentureBeat](https://venturebeat.com/ai/gepa-optimizes-llms-without-costly-reinforcement-learning)
  - [GEPA's use by Databricks covered by VentureBeat](https://venturebeat.com/ai/the-usd100m-openai-partnership-is-nice-but-databricks-real-breakthrough)
  - Stay up to date:
    - [@LakshyAAAgrawal on X (Twitter)](https://x.com/LakshyAAAgrawal)
    - [@lateinteraction on X (Twitter)](https://twitter.com/lateinteraction)
  - Questions, Discussions?
    - [Join our Discord for active discussion](https://discord.gg/WXFSeVGdbW)
    - [Join our Slack](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w)
    - [Open a GitHub issue](https://github.com/gepa-ai/gepa/issues)
- **GEPA Integrations:**
  Want to use GEPA in other frameworks?
  - [DSPy Adapter Code](https://github.com/stanfordnlp/dspy/tree/main/dspy/teleprompt/gepa/gepa_utils.py) (integrates GEPA with [DSPy](https://dspy.ai/)),
  - [MLflow Prompt Optimization](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/) - GEPA is integrated into MLflow's `mlflow.genai.optimize_prompts()` API for automatic prompt improvement using evaluation metrics and training data. Works with any agent framework and supports multi-prompt optimization.
  - [Pydantic AI](https://pydantic.dev/articles/prompt-optimization-with-gepa) - Prompt optimization for Pydantic AI.
  - [Comet ML Opik](https://www.comet.com/docs/opik/agent_optimization/algorithms/gepa_optimizer) - Core optimization algorithm in Opik Agent Optimizer.
  - [OpenAI Cookbook](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining) - Self-evolving agents with GEPA.
  - [HuggingFace Cookbook](https://huggingface.co/learn/cookbook/en/dspy_gepa) - Prompt optimization guide.
  - [Google ADK](https://raphaelmansuy.github.io/adk_training/blog/gepa-optimization-tutorial/) - Optimizing Google Agent Development Kit agents.
  - [Contributed Adapters](src/gepa/adapters/) – see our adapter templates and issue tracker to request new integrations.
    - [DefaultAdapter](src/gepa/adapters/default_adapter/) - System Prompt Optimization for a single-turn task.
    - [DSPy Full Program Adapter](src/gepa/adapters/dspy_full_program_adapter/) - Evolves entire DSPy programs including signatures, modules, and control flow. Achieves **93% accuracy** on MATH benchmark (vs 67% with basic DSPy ChainOfThought).
    - [Generic RAG Adapter](src/gepa/adapters/generic_rag_adapter/) - Vector store-agnostic RAG optimization supporting ChromaDB, Weaviate, Qdrant, Pinecone, and more. Optimizes query reformulation, context synthesis, answer generation, and document reranking prompts.
    - [MCP Adapter](src/gepa/adapters/mcp_adapter/) - Optimize [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) tool usage. Supports local stdio servers, remote SSE/HTTP servers, and optimizes tool descriptions and system prompts.
    - [TerminalBench Adapter](src/gepa/adapters/terminal_bench_adapter/) - Easily integrating GEPA into a Terminus, a sophisticated external agentic pipeline, and optimizing the agents' system prompt.
    - [AnyMaths Adapter](src/gepa/adapters/anymaths_adapter/) - Adapter for optimizing mathematical problem-solving and reasoning tasks. Contributed by [@egmaminta](www.linkedin.com/in/egmaminta).
- **GEPA uses**
    - [Context Compression using GEPA](https://github.com/Laurian/context-compression-experiments-2508)
    - [GEPA Integration into SuperOptiX-AI](https://github.com/SuperagenticAI/gepa-eval)
    - [GEPA for Observable Javascript](https://observablehq.com/@tomlarkworthy/gepa)
    - [bandit_dspy](https://github.com/evalops/bandit_dspy)
    - [GEPA in Go Programming Language](https://github.com/XiaoConstantine/dspy-go)
    - [100% accuracy using GEPA on the clock-hands problem](https://colab.research.google.com/drive/1W-XNxKL2CXFoUTwrL7GLCZ7J7uZgXsut?usp=sharing)
    - [Prompt Optimization for Reliable Backdoor Detection in AI-Generated Code](https://www.lesswrong.com/posts/bALBxf3yGGx4bvvem/prompt-optimization-can-enable-ai-control-research)
    - [Teaching LLMs to Diagnose Production Incidents with ATLAS+GEPA](https://www.arc.computer/blog/atlas-sre-diagnosis)
    - [DataBricks: Building State-of-the-Art Enterprise Agents 90x Cheaper with GEPA](https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization)
    - [comet-ml/opik adds support for GEPA](https://www.comet.com/docs/opik/agent_optimization/algorithms/gepa_optimizer)
    - [Tuning small models (Gemma3-1B) for writing fiction](https://meandnotes.substack.com/p/i-taught-a-small-llm-to-write-fiction?triedRedirect=true)
    - [Cut OCR Error Rates by upto 38% across model classes (Gemini 2.5 Pro, 2.5 Flash, 2.0 Flash)](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf)
    - [Optimizing a Data Analysis coding agent with GEPA, using execution-guided feedback on real-world workloads](https://medium.com/firebird-technologies/context-engineering-improving-ai-coding-agents-using-dspy-gepa-df669c632766)
    - [Generating Naruto (Anime) style dialogues with GPT-4o-mini using GEPA](https://zenn.dev/cybernetics/articles/39fb763aca746c)
    - [Augmenting RL-tuned models with GEPA: Achieving +142% student performance improvement by augmenting a RL-tuned teacher with GEPA](https://www.arc.computer/blog/supercharging-rl-with-online-optimization)
    - [DeepResearch Agent Optimized with GEPA](https://www.rajapatnaik.com/blog/2025/10/23/langgraph-dspy-gepa-researcher)
    - Boosting Sanskrit QA: Finetuning EmbeddingGemma with 50k GEPA generated synthetic data samples [(Tweet)](https://x.com/dhrtha/status/1984315872547385504), [(Code)](https://github.com/ganarajpr/rgfe)
    - [Simulating Realistic Market Research Focus Groups with GEPA-Optimized AI Personas](https://x.com/hammer_mt/status/1984269888979116061)
    - [Optimizing Google ADK Agents' SOP using GEPA](https://raphaelmansuy.github.io/adk_training/blog/gepa-optimization-tutorial/)
    - [HuggingFace Cookbook on prompt optimization for with DSPy and GEPA](https://huggingface.co/learn/cookbook/en/dspy_gepa)
    - [OpenAI Cookbook showing how to build self-evolving agents using GEPA](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining)

---

## Contributions

We welcome adapters, bug fixes, and new use cases. See [src/gepa/adapters/](src/gepa/adapters/) for adapter examples and the [contributing guide](https://gepa-ai.github.io/gepa/guides/contributing/).

**Want to highlight your use case?** Reach out to [lakshyaaagrawal@berkeley.edu](mailto:lakshyaaagrawal@berkeley.edu) or [submit via GitHub](https://github.com/gepa-ai/gepa/issues/new?title=Project%20Submission&body=Organization:%0A%0AProject%20Description:%0A%0AResults:%0A%0ALink%20to%20paper/blog/code:).

---

## Citation

```bibtex
@misc{agrawal2025gepareflectivepromptevolution,
      title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
      author={Lakshya A Agrawal and Shangyin Tan and Dilara Soylu and Noah Ziems and Rishi Khare and Krista Opsahl-Ong and Arnav Singhvi and Herumb Shandilya and Michael J Ryan and Meng Jiang and Christopher Potts and Koushik Sen and Alexandros G. Dimakis and Ion Stoica and Dan Klein and Matei Zaharia and Omar Khattab},
      year={2025},
      eprint={2507.19457},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.19457},
}
```

<p align="center">
  <a href="https://www.star-history.com/#gepa-ai/gepa&Date">
    <img src="https://api.star-history.com/svg?repos=gepa-ai/gepa&type=Date" alt="Star History Chart" width="600">
  </a>
</p>
