---
hide:
  - navigation
  - toc
---

<style>
.hero-section {
  text-align: center;
  padding: 1.5rem 0 1rem 0;
}
.hero-title {
  font-size: 3.5rem;
  font-weight: 800;
  margin-bottom: 1rem;
  background: linear-gradient(120deg, #2196F3, #00BCD4);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-subtitle {
  font-size: 1.3rem;
  margin-bottom: 1rem;
  color: var(--md-default-fg-color--light);
}
.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 2rem;
  margin: 3rem 0;
  text-align: center;
}
.stat-card {
  padding: 1.5rem;
  border-radius: 8px;
  background: var(--md-code-bg-color);
  border: 1px solid var(--md-default-fg-color--lightest);
}
.stat-number {
  font-size: 2.5rem;
  font-weight: 700;
  color: #2196F3;
  display: block;
  margin-bottom: 0.5rem;
}
.stat-label {
  font-size: 0.9rem;
  color: var(--md-default-fg-color--light);
}
.cta-buttons {
  display: flex;
  gap: 1rem;
  justify-content: center;
  margin: 1rem 0;
  flex-wrap: wrap;
}
.cta-primary, .cta-secondary {
  padding: 0.75rem 2rem;
  border-radius: 6px;
  font-weight: 600;
  text-decoration: none;
  display: inline-block;
  transition: all 0.2s;
}
.cta-primary {
  background: #2196F3;
  color: white;
}
.cta-primary:hover {
  background: #1976D2;
  transform: translateY(-2px);
}
.cta-secondary {
  border: 2px solid #2196F3;
  color: #2196F3;
}
.cta-secondary:hover {
  background: rgba(33, 150, 243, 0.1);
}
.feature-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin: 2rem 0;
}
.feature-card {
  padding: 1.5rem;
  border-radius: 8px;
  border-left: 4px solid #2196F3;
  background: var(--md-code-bg-color);
}
.feature-icon {
  font-size: 2rem;
  margin-bottom: 1rem;
}
.feature-title {
  font-size: 1.2rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}
.companies {
  text-align: center;
  margin: 1rem 0;
  padding: 1.5rem;
  background: var(--md-code-bg-color);
  border-radius: 8px;
}
.company-logos {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 2rem;
  align-items: center;
  margin-top: 1.5rem;
}
</style>

<div class="hero-section">
  <div style="text-align: center; margin-bottom: 1rem;">
    <img src="https://raw.githubusercontent.com/gepa-ai/gepa/refs/heads/main/assets/gepa_logo_with_text.svg" alt="GEPA Logo" style="max-width: 400px; width: 100%;">
  </div>
  <!-- <h1 class="hero-title">GEPA</h1> -->
  <p class="hero-subtitle" style="margin-bottom: 1rem;">
    Optimize AI Systems Through Reflective Evolution
  </p>
  <p style="font-size: 1rem; max-width: 800px; margin: 0 auto 1rem; line-height: 1.5;">
    Transform your AI agents and prompts using LLM-powered reflection. Achieve production-grade performance improvements without expensive reinforcement learning or fine-tuning.
  </p>

  <div class="cta-buttons">
    <a href="#quick-start" class="cta-primary">
      🚀 Get Started
    </a>
    <a href="guides/quickstart/" class="cta-secondary">
      📖 Read the Docs
    </a>
    <a href="https://github.com/gepa-ai/gepa" class="cta-secondary">
      ⭐ View on GitHub
    </a>
  </div>

  <p>
    <a href="https://pypi.org/project/gepa/" target="_blank"><img src="https://img.shields.io/pypi/v/gepa?style=for-the-badge&logo=python&logoColor=white" alt="PyPI"></a>
    <a href="https://pypistats.org/packages/gepa" target="_blank"><img src="https://img.shields.io/pypi/dm/gepa?style=for-the-badge&logo=python&logoColor=white" alt="Downloads"></a>
    <a href="https://github.com/gepa-ai/gepa" target="_blank"><img src="https://img.shields.io/github/stars/gepa-ai/gepa?style=for-the-badge&logo=github" alt="GitHub stars"></a>
    <a href="https://arxiv.org/abs/2507.19457" target="_blank"><img src="https://img.shields.io/badge/arXiv-2507.19457-b31b1b.svg?style=for-the-badge" alt="Paper"></a>
  </p>
</div>

---

## :material-office-building: Trusted by Industry Leaders

<div class="companies">
  <div class="company-logos">
    <a href="https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">Databricks</a>
    <a href="https://x.com/tobi/status/1963434604741701909" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">Shopify</a>
    <a href="https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">OpenAI</a>
    <a href="https://huggingface.co/learn/cookbook/en/dspy_gepa" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">HuggingFace</a>
    <a href="https://www.comet.com/site/blog/opik-product-releases-october2025/" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">Comet ML</a>
    <a href="https://github.com/weaviate/recipes/blob/main/integrations/llm-agent-frameworks/dspy/GEPA-Hands-On-Reranker.ipynb" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">Weaviate</a>
    <a href="https://dropbox.tech/machine-learning/vp-josh-clemm-knowledge-graphs-mcp-and-dspy-dash" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">Dropbox</a>
    <a href="https://mlflow.org/blog/mlflow-prompt-optimization" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">MLFlow</a>
    <a href="https://x.com/risk_seeking/status/2015853790512222602?s=20" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">Standard Metrics</a>
    <a href="https://x.com/swyx/status/1991598247782281371?s=20" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">The Browser Company</a>
    <a href="https://www.linkedin.com/posts/dria-ai_today-were-releasing-something-weve-used-activity-7396920472237477888-WyXN" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">Dria</a>
    <a href="https://www.linkedin.com/posts/robin-salimans_agentic-ai-systems-are-only-as-good-as-activity-7399199482644692992-5n-L" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">Zapier</a>
    <a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/verifiers/gepa" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">Prime Intellect</a>
    <a href="https://www.pharmabiz.com/NewsDetails.aspx?aid=181272&sid=2" target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">ClirNet</a>
    <a href="https://docs.usesynth.ai/sdk/jobs/prompt-optimization/gepa"  target="_blank" style="font-weight: 600; font-size: 1.1rem; text-decoration: none; color: inherit;">SynthAI</a>
    <span style="font-weight: 600; font-size: 1.1rem;">Uber</span>
    <span style="font-weight: 600; font-size: 1.1rem;">NuBank</span>
    <span style="font-weight: 600; font-size: 1.1rem;">Veris.AI</span>
    <span style="font-weight: 600; font-size: 1.1rem;">AWS</span>
    <span style="font-weight: 600; font-size: 1.1rem;">Infosys</span>
    <span style="font-weight: 600; font-size: 1.1rem;">Invitae</span>
    <span style="font-weight: 600; font-size: 1.1rem;">Meta</span>
    <span style="font-weight: 600; font-size: 1.1rem;">Cerebras</span>
    <span style="font-weight: 600; font-size: 1.1rem;">Bespoke AI Labs</span>
  </div>
</div>

!!! quote ":material-shopping: **Tobi Lutke**, CEO Shopify"
    Both DSPy and (especially) **GEPA are currently severely under hyped** in the AI context engineering world

    [:material-twitter: View tweet](https://x.com/tobi/status/1963434604741701909)

!!! quote ":material-dropbox: **Drew Houston**, CEO Dropbox"
    Have heard great things about DSPy plus GEPA, which is an **even stronger prompt optimizer than miprov2** — repo and (fascinating) examples of generated prompts at https://github.com/gepa-ai/gepa and paper at https://arxiv.org/abs/2507.19457

    [:material-twitter: View tweet](https://x.com/drewhouston/status/1974750621690728623)

!!! quote ":material-brain: **OpenAI**"
    Self-evolving agents that autonomously retrain themselves using GEPA to improve performance over time.

    [:material-link: Read cookbook](https://developers.openai.com/cookbook/examples/partners/self_evolving_agents/autonomous_agent_retraining)

!!! quote ":material-code-braces: **Chad Boyda**, CTO AppSumo"
    DSPy's GEPA is prompt engineering! **The only kind we should all collectively be doing.** What a work of art

    [:material-twitter: View tweet](https://x.com/chadboyda/status/1955298177197764963)

!!! quote ":material-shopping: **Tobi Lutke**, CEO Shopify"
    QMD update shipped: New fine-tuned query expansion model, **GEPA-optimized synthetic training data**, Semantic chunking that actually understands document structure

    [:material-twitter: View QMD announcement](https://x.com/tobi/status/2017750533361070425?s=20)

!!! quote ":material-database: **Ivan Zhou**, Databricks"
    Automated prompt optimization (GEPA) can **push open-source models beyond frontier performance on enterprise tasks — at a fraction of the cost!** gpt-oss-120b + GEPA beats Claude Opus 4.1 on Information Extraction (+2.2 points) — while being **90× cheaper to serve**. GEPA + SFT together gives the **highest gains**.

    [:material-twitter: View tweet](https://x.com/ivanhzyy/status/1971066193747689521)

<div style="text-align: center; margin: 2rem 0;">
  <a href="guides/use-cases/" style="text-decoration: none; color: #2196F3; font-weight: 600;">
    → View all 50+ use cases and success stories
  </a>
</div>

---

## :material-chart-line: Performance at a Glance

<div class="stats-grid">
  <div class="stat-card">
    <a href="https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization" target="_blank" style="text-decoration: none; color: inherit;">
      <span class="stat-number">90×</span>
      <span class="stat-label">Cost Reduction</span>
      <p style="margin-top: 0.5rem; font-size: 0.85rem;">Open-source models beat Claude Opus 4.1</p>
    </a>
  </div>
  <div class="stat-card">
    <a href="https://arxiv.org/abs/2507.19457" target="_blank" style="text-decoration: none; color: inherit;">
      <span class="stat-number">35×</span>
      <span class="stat-label">Efficiency Gain</span>
      <p style="margin-top: 0.5rem; font-size: 0.85rem;">vs. Reinforcement Learning methods</p>
    </a>
  </div>
  <div class="stat-card">
    <span class="stat-number">150</span>
    <span class="stat-label">Sample Calls</span>
    <p style="margin-top: 0.5rem; font-size: 0.85rem;">To achieve state-of-the-art results</p>
  </div>
  <div class="stat-card">
    <a href="guides/use-cases.md" style="text-decoration: none; color: inherit;">
      <span class="stat-number">50+</span>
      <span class="stat-label">Production Use Cases</span>
      <p style="margin-top: 0.5rem; font-size: 0.85rem;">Across diverse industries</p>
    </a>
  </div>
</div>

---

## :material-rocket-launch-outline: Quick Start {#quick-start}

!!! example "Install in seconds"

    ```bash
    pip install gepa
    ```

=== "Basic Example"

    ```python title="optimize.py" linenums="1"
    import gepa

    # Load your dataset
    trainset, valset, _ = gepa.examples.aime.init_dataset()

    # Define your initial prompt
    seed_prompt = {
        "system_prompt": "You are a helpful assistant..."
    }

    # Run optimization
    result = gepa.optimize(
        seed_candidate=seed_prompt,
        trainset=trainset,
        valset=valset,
        task_lm="openai/gpt-4.1-mini",      # Model to optimize
        max_metric_calls=150,                # Optimization budget
        reflection_lm="openai/gpt-5",       # Model for reflection
    )

    # Use your optimized prompt
    print(result.best_candidate['system_prompt'])
    ```

    !!! success "Result"
        **+10% improvement** (46.6% → 56.6%) on AIME 2025 with GPT-4.1 Mini!

=== "With DSPy"

    ```python title="dspy_example.py" linenums="1"
    import dspy
    from dspy.teleprompt import GEPA

    # Define your DSPy program
    class RAG(dspy.Module):
        def __init__(self):
            self.retrieve = dspy.Retrieve(k=3)
            self.generate = dspy.ChainOfThought("context, question -> answer")

        def forward(self, question):
            context = self.retrieve(question).passages
            return self.generate(context=context, question=question)

    # Optimize with GEPA
    gepa = GEPA(metric=your_metric, max_metric_calls=150)
    optimized_rag = gepa.compile(RAG(), trainset=trainset)
    ```

    !!! tip "DSPy Integration"
        GEPA is built into DSPy! See [DSPy tutorials](https://dspy.ai/tutorials/gepa_ai_program/) for more.

=== "Custom System"

    ```python title="custom_system.py" linenums="1"
    from gepa import optimize
    from gepa.core.adapter import EvaluationBatch

    # Create a custom adapter for your system
    class MySystemAdapter:
        def evaluate(self, batch, candidate, capture_traces=False):
            outputs, scores, trajectories = [], [], []

            for example in batch:
                # Run your system with the candidate
                prompt = candidate['my_prompt']
                result = my_system.run(prompt, example)

                # Compute score (higher is better)
                score = compute_score(result, example)
                outputs.append(result)
                scores.append(score)

                if capture_traces:
                    # Capture execution trace for reflection
                    trajectories.append({
                        'input': example,
                        'output': result.output,
                        'steps': result.intermediate_steps,
                        'errors': result.errors
                    })

            return EvaluationBatch(
                outputs=outputs,
                scores=scores,
                trajectories=trajectories if capture_traces else None
            )

        def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
            # Build feedback dataset for each component
            reflective_data = {}

            for component in components_to_update:
                reflective_data[component] = []

                for traj, score in zip(eval_batch.trajectories, eval_batch.scores):
                    reflective_data[component].append({
                        'Inputs': traj['input'],
                        'Generated Outputs': traj['output'],
                        'Feedback': f"Score: {score}. Errors: {traj['errors']}"
                    })

            return reflective_data

    # Optimize your custom system
    result = optimize(
        seed_candidate={'my_prompt': 'Initial prompt...'},
        trainset=my_trainset,
        valset=my_valset,
        adapter=MySystemAdapter(),
        task_lm="openai/gpt-4.1-mini",
    )
    ```

<div style="text-align: center; margin: 2rem 0;">
  <a href="guides/quickstart/" style="padding: 0.75rem 2rem; background: #2196F3; color: white; text-decoration: none; border-radius: 6px; font-weight: 600;">
    📖 Full Documentation
  </a>
</div>

---

## :material-newspaper: Featured In

<div class="grid cards" markdown>

-   :material-database: **[Databricks Blog](https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization)**

    ---

    Building State-of-the-Art Enterprise Agents 90x Cheaper with Automated Prompt Optimization

-   :material-brain: **[OpenAI Cookbook](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining)**

    ---

    Self-Evolving Agents: Autonomous Agent Retraining

-   :material-heart: **[HuggingFace Cookbook](https://huggingface.co/learn/cookbook/en/dspy_gepa)**

    ---

    Prompt Optimization with DSPy and GEPA

-   :material-newspaper-variant: **[VentureBeat: GEPA Breakthrough](https://venturebeat.com/infrastructure/the-usd100m-openai-partnership-is-nice-but-databricks-real-breakthrough)**

    ---

    The $100M OpenAI partnership is nice, but Databricks' real breakthrough is GEPA

-   :material-newspaper-variant: **[VentureBeat](https://venturebeat.com/ai/gepa-optimizes-llms-without-costly-reinforcement-learning/)**

    ---

    GEPA Optimizes LLMs Without Costly Reinforcement Learning

-   :material-chart-box: **[Trader's Union](https://tradersunion.com/news/billionaires/show/484192-lutke-dspy-gepa-ai/)**

    ---

    Shopify CEO Tobi Lutke on DSPy and GEPA transforming AI

-   :material-presentation: **[AI Engineer Code Summit 2025](https://x.com/swyx/status/1991598247782281371?s=20)**

    ---

    The Browser Company's Dia Browser uses GEPA (covered by @swyx)

-   :material-school: **[Berkeley AI Summit](https://youtu.be/c39fJ2WAj6A?t=6386)**

    ---

    Matei Zaharia presents on "Reflective Optimization with GEPA and DSPy"

</div>

---

## :material-application: Production Use Cases

<div class="grid cards" markdown>

-   :material-office-building: **Enterprise & Production**

    ---

    - **90x cost reduction** at Databricks
    - Self-evolving agents in OpenAI Cookbook
    - Core algorithm in Comet ML Opik
    - Multi-cloud cost optimization (-31%)

    [:material-arrow-right: Explore](guides/use-cases.md#enterprise-production)

-   :material-code-braces: **AI Coding Agents**

    ---

    - Production incident diagnosis (Arc.computer)
    - Data analysis agents (FireBird)
    - Code safety monitoring
    - Automated research systems

    [:material-arrow-right: Explore](guides/use-cases.md#ai-coding-agents-research-tools)

-   :material-hospital-building: **Domain-Specific**

    ---

    - Healthcare multi-agent RAG systems
    - **38% OCR error reduction** (Intrinsic Labs)
    - Market research AI personas
    - Creative writing with small models

    [:material-arrow-right: Explore](guides/use-cases.md#domain-specific-applications)

-   :material-brain: **Research & Advanced**

    ---

    - Multi-objective optimization
    - Agent architecture discovery
    - Adversarial prompt search
    - Unverifiable task optimization

    [:material-arrow-right: Explore](guides/use-cases.md#advanced-capabilities)

</div>

---

## :material-compare: Why Choose GEPA?

| Feature | GEPA | Reinforcement Learning | Manual Prompting |
|---------|------|----------------------|------------------|
| **Cost** | :material-check-circle:{ style="color: #4CAF50" } Low | :material-close-circle:{ style="color: #f44336" } Very High | :material-check-circle:{ style="color: #4CAF50" } Low |
| **Sample Efficiency** | :material-check-circle:{ style="color: #4CAF50" } High (150 calls) | :material-close-circle:{ style="color: #f44336" } Low (10K+ calls) | :material-minus-circle:{ style="color: #FF9800" } N/A |
| **Performance** | :material-check-circle:{ style="color: #4CAF50" } SOTA | :material-check-circle:{ style="color: #4CAF50" } SOTA | :material-close-circle:{ style="color: #f44336" } Suboptimal |
| **Interpretability** | :material-check-circle:{ style="color: #4CAF50" } Natural Language | :material-close-circle:{ style="color: #f44336" } Black Box | :material-check-circle:{ style="color: #4CAF50" } Clear |
| **Setup Time** | :material-check-circle:{ style="color: #4CAF50" } Minutes | :material-close-circle:{ style="color: #f44336" } Days/Weeks | :material-check-circle:{ style="color: #4CAF50" } Minutes |
| **Framework Support** | :material-check-circle:{ style="color: #4CAF50" } Any System | :material-minus-circle:{ style="color: #FF9800" } Framework Specific | :material-check-circle:{ style="color: #4CAF50" } Any System |
| **Multi-Objective** | :material-check-circle:{ style="color: #4CAF50" } Native | :material-minus-circle:{ style="color: #FF9800" } Complex | :material-close-circle:{ style="color: #f44336" } Manual |

---

## :material-cog: How It Works

!!! abstract "The GEPA Algorithm"

    GEPA uses **reflective prompt evolution** to optimize AI systems:

    1. **:material-numeric-1-circle: Evaluate** your current system on a dataset
    2. **:material-numeric-2-circle: Reflect** on failures using an LLM to generate natural language feedback
    3. **:material-numeric-3-circle: Mutate** your prompts/instructions based on the feedback
    4. **:material-numeric-4-circle: Select** the best candidates using Pareto frontier tracking
    5. **:material-numeric-5-circle: Repeat** until convergence or budget exhausted

    <div style="text-align: center; margin: 2rem 0;">
      <img src="https://raw.githubusercontent.com/gepa-ai/gepa/refs/heads/main/assets/gepa_logo_with_text.svg" alt="GEPA Algorithm" style="max-width: 500px; width: 100%;">
    </div>

    **Key Innovation**: Instead of gradient-based updates, GEPA uses natural language reasoning to understand what's wrong and how to fix it.

[:material-file-document: Read the Paper](https://arxiv.org/abs/2507.19457){ .md-button }
[:material-book-open-variant: Technical Details](guides/quickstart.md){ .md-button }

---

## :material-account-group: Join the Community

<div class="grid cards" markdown>

-   :fontawesome-brands-discord: **Discord**

    ---

    Join 1,000+ developers building with GEPA

    [:material-open-in-new: Join Server](https://discord.gg/A7dABbtmFw)

-   :material-slack: **Slack**

    ---

    Connect with the core team and contributors

    [:material-open-in-new: Join Workspace](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w)

-   :material-github: **GitHub**

    ---

    Contribute, report issues, or star the repo

    [:material-open-in-new: View Repository](https://github.com/gepa-ai/gepa)

-   :material-twitter: **Twitter/X**

    ---

    Follow for updates and community highlights

    [:material-open-in-new: Follow @LakshyAAAgrawal](https://x.com/LakshyAAAgrawal)

</div>

---

## :material-upload: Share Your Success Story

!!! tip "Get Featured"

    Using GEPA in production? We'd love to showcase your work!

    <div style="display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1rem;">
      <a href="mailto:lakshyaaagrawal@berkeley.edu?subject=Feature%20GEPA%20usecase" style="padding: 0.5rem 1.5rem; background: #2196F3; color: white; text-decoration: none; border-radius: 4px; font-weight: 600;">
        Submit via Email
      </a>
      <a href="https://github.com/gepa-ai/gepa/issues/new?title=Use%20Case%20Submission&body=Organization:%0A%0AUse%20Case:%0A%0AResults:%0A%0ALogo%20URL:%0A%0ALink%20to%20blog/tweet:" style="padding: 0.5rem 1.5rem; border: 2px solid #2196F3; color: #2196F3; text-decoration: none; border-radius: 4px; font-weight: 600;">
        Submit via GitHub
      </a>
      <a href="https://discord.gg/A7dABbtmFw" style="padding: 0.5rem 1.5rem; border: 2px solid #2196F3; color: #2196F3; text-decoration: none; border-radius: 4px; font-weight: 600;">
        Share on Discord
      </a>
    </div>

---

## :material-book-open-page-variant: Learn More

<div class="grid cards" markdown>

-   :material-lightning-bolt: **Quickstart**

    ---

    Get up and running in 5 minutes

    [:material-arrow-right: Start Here](guides/quickstart.md)

-   :material-school: **Tutorials**

    ---

    Step-by-step guides and examples

    [:material-arrow-right: View Tutorials](tutorials/)

-   :material-api: **API Reference**

    ---

    Complete documentation of all components

    [:material-arrow-right: API Docs](api/)

-   :material-help-circle: **FAQ**

    ---

    Common questions and troubleshooting

    [:material-arrow-right: Read FAQ](guides/faq.md)

-   :material-file-document: **Research Paper**

    ---

    Academic publication on arXiv

    [:material-arrow-right: Read Paper](https://arxiv.org/abs/2507.19457)

-   :material-presentation: **Use Cases**

    ---

    50+ production examples and integrations

    [:material-arrow-right: Explore](guides/use-cases.md)

</div>

---

## :material-format-quote-close: Citation

If you use GEPA in your research, please cite our paper:

```bibtex
@misc{agrawal2025gepareflectivepromptevolution,
      title={GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning},
      author={Lakshya A Agrawal and Shangyin Tan and Dilara Soylu and Noah Ziems and
              Rishi Khare and Krista Opsahl-Ong and Arnav Singhvi and Herumb Shandilya and
              Michael J Ryan and Meng Jiang and Christopher Potts and Koushik Sen and
              Alexandros G. Dimakis and Ion Stoica and Dan Klein and Matei Zaharia and Omar Khattab},
      year={2025},
      eprint={2507.19457},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.19457}
}
```

---

<div style="text-align: center; padding: 3rem 0; background: var(--md-code-bg-color); border-radius: 8px; margin: 2rem 0;">
  <h2 style="margin-bottom: 1rem;">Ready to Get Started?</h2>
  <p style="font-size: 1.1rem; margin-bottom: 2rem; color: var(--md-default-fg-color--light);">
    Install GEPA and start optimizing in minutes
  </p>
  <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
    <a href="guides/quickstart/" style="padding: 0.75rem 2rem; background: #2196F3; color: white; text-decoration: none; border-radius: 6px; font-weight: 600;">
      🚀 Get Started
    </a>
    <a href="https://github.com/gepa-ai/gepa" style="padding: 0.75rem 2rem; border: 2px solid #2196F3; color: #2196F3; text-decoration: none; border-radius: 6px; font-weight: 600;">
      ⭐ Star on GitHub
    </a>
  </div>
</div>
