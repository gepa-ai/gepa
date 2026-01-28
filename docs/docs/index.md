# GEPA: System Optimization through Reflective Text Evolution

<p align="center">
  <img src="https://raw.githubusercontent.com/gepa-ai/gepa/refs/heads/main/assets/gepa_logo_with_text.svg" alt="GEPA Logo" width="400">
</p>

**GEPA** (Genetic-Pareto) is a framework for **optimizing arbitrary systems composed of text components**—like AI prompts, code snippets, or textual specs—against any evaluation metric.

[![PyPI - Version](https://img.shields.io/pypi/v/gepa)](https://pypi.org/project/gepa/)
[![PyPI Downloads](https://static.pepy.tech/badge/gepa)](https://pepy.tech/projects/gepa)

## Overview

GEPA employs LLMs to reflect on system behavior, using feedback from execution and evaluation traces to drive targeted improvements. Through iterative mutation, reflection, and Pareto-aware candidate selection, GEPA evolves robust, high-performing variants with minimal evaluations, co-evolving multiple components in modular systems for domain-specific gains.

This repository provides the official implementation of the GEPA algorithm as proposed in the paper titled "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning" ([arXiv:2507.19457](https://arxiv.org/abs/2507.19457)).

## Installation

```bash
pip install gepa
```

To install the very latest from `main`:

```bash
pip install git+https://github.com/gepa-ai/gepa.git
```

## Quick Start

### The Easiest Path: DSPy Integration

!!! tip "Recommended"
    The easiest and most powerful way to use GEPA for prompt optimization is within [DSPy](https://dspy.ai/), where the GEPA algorithm is directly available through the `dspy.GEPA` API. See [dspy.GEPA Tutorials](https://dspy.ai/tutorials/gepa_ai_program/).

### Simple Prompt Optimization Example

GEPA can be run in just a few lines of code. Here's an example optimizing a system prompt for math problems:

```python
import gepa

# Load AIME dataset
trainset, valset, _ = gepa.examples.aime.init_dataset()

seed_prompt = {
    "system_prompt": "You are a helpful assistant. You are given a question and you need to answer it. The answer should be given at the end of your response in exactly the format '### <final answer>'"
}

# Run GEPA optimization
gepa_result = gepa.optimize(
    seed_candidate=seed_prompt,
    trainset=trainset,
    valset=valset,
    task_lm="openai/gpt-4.1-mini",  # Model being optimized
    max_metric_calls=150,            # Budget
    reflection_lm="openai/gpt-5",    # Strong model for reflection
)

print("Optimized Prompt:", gepa_result.best_candidate['system_prompt'])
```

This achieves **+10% improvement** (46.6% → 56.6%) on AIME 2025 with GPT-4.1 Mini!

## Key Concepts

### Adapter Architecture

GEPA is built around a flexible [`GEPAAdapter`](api/core/GEPAAdapter.md) abstraction that lets it plug into any system:

- **System**: A harness that uses text components to perform a task
- **Candidate**: A mapping from component names to component text
- **Trajectory**: Execution traces captured during evaluation

### Built-in Adapters

| Adapter | Description |
|---------|-------------|
| [`DefaultAdapter`](api/adapters/DefaultAdapter.md) | Single-turn LLM with system prompt optimization |
| [`DSPyAdapter`](api/adapters/DSPyAdapter.md) | DSPy program instruction and prompt optimization |
| [`DSPyFullProgramAdapter`](api/adapters/DSPyFullProgramAdapter.md) | Full DSPy program evolution including structure |
| [`RAGAdapter`](api/adapters/RAGAdapter.md) | RAG systems with any vector store |
| [`MCPAdapter`](api/adapters/MCPAdapter.md) | Model Context Protocol tool optimization |
| [`TerminalBenchAdapter`](api/adapters/TerminalBenchAdapter.md) | Terminal/shell agent optimization |

## How GEPA Works

GEPA optimizes text components using an evolutionary search algorithm with LLM-based reflection for mutations. Key features:

1. **Reflective Mutation**: Uses task-specific textual feedback (compiler errors, profiler reports, etc.) to guide improvements
2. **Pareto Frontier Tracking**: Maintains candidates that excel on different subsets of the validation data
3. **Multi-Component Optimization**: Co-evolves multiple text components simultaneously

For details, see the paper: [GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457)

## Example GEPA Prompts

<table>
  <tr>
  <td colspan="2" align="center"><strong>Example GEPA Prompts</strong></td>
  </tr>
  <tr>
    <td align="center">HotpotQA (multi-hop QA) Prompt</td>
    <td align="center">AIME Prompt</td>
  </tr>
  <tr>
    <td width="52%" valign="top">
      <img src="https://raw.githubusercontent.com/gepa-ai/gepa/refs/heads/main/assets/gepa_prompt_hotpotqa.png" alt="HotpotQA Prompt" width="1400">
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

## GEPA in Action

Discover how organizations and researchers are using GEPA to optimize AI systems across diverse domains.

<div class="grid cards" markdown>

-   :material-office-building: **Enterprise & Production**

    ---

    - [**DataBricks**](https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization): 90x cost reduction with open-source models outperforming Claude Opus 4.1, Sonnet 4, and GPT-5
    - [**OpenAI Cookbook**](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining): Self-evolving agents and autonomous retraining loops
    - [**HuggingFace Cookbook**](https://huggingface.co/learn/cookbook/en/dspy_gepa): Prompt optimization with DSPy and GEPA
    - [**Comet-ml Opik**](https://www.comet.com/docs/opik/agent_optimization/algorithms/gepa_optimizer): GEPA as core optimization algorithm

    [:material-arrow-right: View all enterprise use cases](guides/use-cases.md#enterprise-production)

-   :material-code-braces: **AI Coding Agents**

    ---

    - [**Arc.computer ATLAS**](https://www.arc.computer/blog/atlas-sre-diagnosis): Production incident diagnosis and root cause analysis
    - [**FireBird Auto-Analyst**](https://medium.com/firebird-technologies/context-engineering-improving-ai-coding-agents-using-dspy-gepa-df669c632766): Data analysis coding agents with 4 specialized agents
    - [**AI Code Safety**](https://www.lesswrong.com/posts/bALBxf3yGGx4bvvem/prompt-optimization-can-enable-ai-control-research): Backdoor detection and safety monitoring
    - [**DeepResearch Agent**](https://www.rajapatnaik.com/blog/2025/10/23/langgraph-dspy-gepa-researcher): LangGraph + DSPy + GEPA research system

    [:material-arrow-right: View all coding agent use cases](guides/use-cases.md#ai-coding-agents-research-tools)

-   :material-hospital-building: **Domain-Specific Applications**

    ---

    - [**Healthcare**](https://kargarisaac.medium.com/building-and-optimizing-multi-agent-rag-systems-with-dspy-and-gepa-2b88b5838ce2): Multi-agent RAG for diabetes and COPD
    - [**OCR**](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf): Up to 38% error reduction across Gemini models
    - [**Market Research**](https://x.com/hammer_mt/status/1984269888979116061): AI personas for realistic focus groups
    - [**Creative Writing**](https://meandnotes.substack.com/p/i-taught-a-small-llm-to-write-fiction): Fiction generation with small models

    [:material-arrow-right: View all domain applications](guides/use-cases.md#domain-specific-applications)

-   :material-lightbulb: **Advanced Capabilities**

    ---

    - [**Multimodal/VLM**](https://tinyurl.com/gepa-ocr-example): OCR performance optimization
    - **Agent Architecture Discovery**: Automated design search
    - **Adversarial Prompt Search**: Edge case discovery
    - [**Unverifiable Tasks**](https://x.com/AsfiShaheen/status/1967866903331999807): Evaluator-optimizer patterns

    [:material-arrow-right: View all advanced capabilities](guides/use-cases.md#advanced-capabilities)

</div>

[:material-arrow-right: **Explore all use cases and community integrations**](guides/use-cases.md)

## Further Resources

- **Paper**: [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)
- **Tutorials**: [dspy.GEPA Tutorials](https://dspy.ai/tutorials/gepa_ai_program/)
- **Discord**: [Join our community](https://discord.gg/A7dABbtmFw)
- **Slack**: [GEPA Slack](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w)

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
