# Showcase

Discover how organizations and researchers are using GEPA to optimize AI systems across diverse domains. These examples showcase the versatility and impact of reflective prompt evolution.

!!! tip "Living Document"
    This page is continuously updated with new use cases from the community. Have a GEPA success story? Share it on [Discord](https://discord.gg/WXFSeVGdbW), [Slack](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w), or [Twitter/X](https://x.com/LakshyAAAgrawal)!

**Quick Navigation:**

- [:material-office-building: Enterprise & Production](#enterprise-production)
- [:material-code-braces: AI Coding Agents & Research Tools](#ai-coding-agents-research-tools)
- [:material-hospital-building: Domain-Specific Applications](#domain-specific-applications)
- [:material-lightbulb: Advanced Capabilities](#advanced-capabilities)
- [:material-trophy: Research & Academic](#research-academic)
- [:material-newspaper: Media & Press Coverage](#media-press-coverage)
- [:material-trending-up: Emerging Applications](#emerging-applications)
- [:material-source-branch: Community Integrations](#community-integrations)
- [:material-cloud-outline: Infrastructure & DevOps](#infrastructure-devops)
- [:material-account-voice: Creative & Generative](#creative-generative-applications)
- [:material-database-sync: Data Processing & Synthesis](#data-processing-synthesis)
- [:material-book-open-variant: Community Tutorials](#community-tutorials-guides)
- [:material-earth: International Coverage](#international-coverage)

---

## :material-office-building: Enterprise & Production

<div class="grid cards" markdown>

-   **DataBricks: 90x Cost Reduction**

    ---

    ![DataBricks Enterprise Agents](../static/img/use-cases/databricks_enterprise.png){ .card-image }

    DataBricks achieved **90x cheaper inference** while maintaining or improving performance by optimizing enterprise agents with GEPA.

    **Key Results:**

    - Open-source models optimized with GEPA outperform Claude Opus 4.1, Claude Sonnet 4, and GPT-5
    - Consistent **3-7% performance gains** across all model types
    - At 100,000 requests, serving costs represent 95%+ of AI expenditure—GEPA makes this sustainable

    [:material-arrow-right: Read the full blog](https://www.databricks.com/blog/building-state-art-enterprise-agents-90x-cheaper-automated-prompt-optimization)

-   **OpenAI Cookbook: Self-Evolving Agents**

    ---

    ![OpenAI Cookbook](../static/img/use-cases/openai_cookbook.png){ .card-image }

    The official OpenAI Cookbook (Nov 2025) features GEPA for building **autonomous self-healing workflows**.

    **What You'll Learn:**

    - Diagnose why agents fall short of production readiness
    - Build automated LLMOps retraining loops
    - Combine human review, LLM-as-judge evaluations, and GEPA optimization

    [:material-arrow-right: View cookbook](https://cookbook.openai.com/examples/partners/self_evolving_agents/autonomous_agent_retraining)

-   **HuggingFace Cookbook**

    ---

    ![HuggingFace Cookbook](../static/img/use-cases/huggingface_cookbook.png){ .card-image }

    Comprehensive guide on **prompt optimization with DSPy and GEPA**.

    **What's Inside:**

    - Setting up DSPy with language models
    - Processing mathematical problem datasets
    - Building Chain-of-Thought reasoning programs
    - Error-driven feedback optimization

    [:material-arrow-right: View cookbook](https://huggingface.co/learn/cookbook/en/dspy_gepa)

-   **Google ADK Agents Optimization**

    ---

    ![Google ADK Training](../static/img/use-cases/google_adk.png){ .card-image }

    Tutorial on optimizing **Google Agent Development Kit (ADK)** agents using GEPA for improved performance.

    **Key Topics:**

    - Optimizing agent SOPs (Standard Operating Procedures)
    - Integrating GEPA with ADK workflows
    - Production deployment patterns

    [:material-arrow-right: View tutorial](https://raphaelmansuy.github.io/adk_training/blog/gepa-optimization-tutorial/)

-   **Comet-ml Opik Integration**

    ---

    GEPA is integrated into Comet's **Opik Agent Optimizer** platform as a core optimization algorithm.

    **Capabilities:**

    - Optimize prompts, agents, and multimodal systems
    - Works alongside MetaPrompt, HRPO, Few-Shot Bayesian optimizers
    - Automates prompt editing, testing, and tool refinement

    [:material-arrow-right: View documentation](https://www.comet.com/docs/opik/agent_optimization/algorithms/gepa_optimizer)

</div>

---

## :material-code-braces: AI Coding Agents & Research Tools

<div class="grid cards" markdown>

-   **Production Incident Diagnosis**

    ---

    ![ATLAS Incident Diagnosis](../static/img/use-cases/atlas_incidents.png){ .card-image }

    Arc.computer's **ATLAS** system uses GEPA-optimized agents to teach LLMs to diagnose production incidents.

    **Application:**

    - Automated root cause analysis (RCA)
    - Dynamic collection of logs, metrics, and databases
    - Reduces manual burden on on-call engineers

    [:material-arrow-right: Learn more](https://www.arc.computer/blog/atlas-sre-diagnosis)

-   **ATLAS Augmented: +142% Student Performance**

    ---

    ![ATLAS Augmented](../static/img/use-cases/atlas_augmented.jpg){ .card-image }

    GEPA can **augment even RL-tuned models**. The Intelligence Arc team uses GEPA in their ATLAS framework to improve an already powerful and RL-tuned teacher model.

    **Key Result:**

    - **+142% student performance improvement** when guided by the GEPA-improved teacher
    - Demonstrates that GEPA works alongside RL, not just as an alternative
    - Shows GEPA's value even for already-optimized models

    [:material-arrow-right: Read the technical blog](https://www.arc.computer/blog/supercharging-rl-with-online-optimization)

-   **Data Analysis Coding Agents**

    ---

    ![FireBird Auto-Analyst](../static/img/use-cases/firebird_auto_analyst.png){ .card-image }

    FireBird Technologies optimized their **Auto-Analyst** platform using GEPA for improved code execution.

    **Architecture:**

    - 4 specialized agents: Pre-processing, Statistical Analytics, Machine Learning, Visualization
    - Optimized 4 primary signatures covering 90% of all code runs
    - Tested across multiple model providers to avoid overfitting

    [:material-arrow-right: Read the article](https://medium.com/firebird-technologies/context-engineering-improving-ai-coding-agents-using-dspy-gepa-df669c632766)

-   **Backdoor Detection in AI Code**

    ---

    ![LessWrong Backdoor Detection](../static/img/use-cases/lesswrong_backdoor.png){ .card-image }

    GEPA enables **AI control research** by optimizing classifiers to detect backdoors in AI-generated code.

    **Approach:**

    - Trusted monitoring using weaker models
    - Classification based on suspicion scores
    - Safety measured by true positive rate at given false positive rate

    [:material-arrow-right: Read on LessWrong](https://www.lesswrong.com/posts/bALBxf3yGGx4bvvem/prompt-optimization-can-enable-ai-control-research)

-   **AI Code Safety Monitoring**

    ---

    ![Code Safety Monitoring](../static/img/use-cases/code_safety.png){ .card-image }

    GEPA enables **monitoring safety of AI-generated code** through optimized classifiers.

    **Capabilities:**

    - Detect potentially unsafe code patterns
    - Monitor code generation in real-time
    - Improve detection accuracy with reflective optimization

    [:material-arrow-right: Try the example](https://tinyurl.com/gepa-ai-code-monitor)

-   **DeepResearch Agent**

    ---

    A production-grade **agentic research system** combining LangGraph + DSPy + GEPA.

    **Pipeline:**

    - Query planning with diverse search queries
    - Parallel web search via Exa API
    - Summarization, gap analysis, and iterative research rounds
    - Module-specific GEPA optimization for each agent role

    [:material-arrow-right: View tutorial](https://www.rajapatnaik.com/blog/2025/10/23/langgraph-dspy-gepa-researcher)

</div>

---

## :material-hospital-building: Domain-Specific Applications

<div class="grid cards" markdown>

-   **Healthcare Multi-Agent RAG**

    ---

    Building **multi-agent RAG systems** for diabetes and COPD using DSPy and GEPA.

    **System Design:**

    - Two specialized subagents (disease experts)
    - Vector database search for medical documents
    - ReAct subagents individually optimized with GEPA
    - Lead agent for orchestration

    [:material-arrow-right: Read the guide](https://kargarisaac.medium.com/building-and-optimizing-multi-agent-rag-systems-with-dspy-and-gepa-2b88b5838ce2)

-   **OCR Accuracy: Up to 38% Error Reduction**

    ---

    ![OCR Intrinsic Labs](../static/img/use-cases/ocr_intrinsic.png){ .card-image }

    Intrinsic Labs achieved significant **OCR error rate reductions** across Gemini model classes.

    **Models Improved:**

    - Gemini 2.5 Pro
    - Gemini 2.5 Flash
    - Gemini 2.0 Flash

    A grounded benchmark for document-understanding agents under operational constraints.

    [:material-arrow-right: Read the research](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf)

    [:material-arrow-right: Resources page](https://www.intrinsic-labs.ai/resources/prompt-optimized-ocr-for-production)

-   **Market Research AI Personas**

    ---

    ![Market Research Focus Groups](../static/img/use-cases/market_research.png){ .card-image }

    Simulating **realistic focus groups** with GEPA-optimized AI personas for market research.

    **Benefits:**

    - Eliminates geographic constraints and facility costs
    - No moderator bias
    - Tests across different personality types
    - Research timelines: weeks → hours

    [:material-arrow-right: Learn more](https://x.com/hammer_mt/status/1984269888979116061)

-   **Fiction Writing with Small Models**

    ---

    ![Creative Writing](../static/img/use-cases/creative_writing.png){ .card-image }

    Teaching **Gemma3-1B** to write engaging fiction through GEPA optimization.

    Demonstrates that small models can handle creative tasks with the right prompts.

    [:material-arrow-right: Read on Substack](https://meandnotes.substack.com/p/i-taught-a-small-llm-to-write-fiction?triedRedirect=true)

</div>

---

## :material-lightbulb: Advanced Capabilities

<div class="grid cards" markdown>

-   **Multimodal/VLM Performance (OCR)**

    ---

    ![Multimodal OCR](../static/img/use-cases/multimodal_ocr.png){ .card-image }

    GEPA improves **Multimodal/VLM Performance** for OCR tasks through optimized prompting strategies.

    [:material-arrow-right: Try the example](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf)

-   **Agent Architecture Discovery**

    ---

    ![Architecture Discovery](../static/img/use-cases/arc_agi_image.png){ .card-image }

    GEPA for **automated agent architecture discovery** - finding optimal agent designs through evolutionary search.

    [:material-arrow-right: View ARC-AGI tutorial](../tutorials/arc_agi.ipynb)

-   **Adversarial Prompt Search**

    ---

    ![Adversarial Prompt Search](../static/img/use-cases/adversarial_prompt.png){ .card-image }

    GEPA for **adversarial prompt search** - discovering edge cases and failure modes in AI systems.

    *Advanced application for AI safety research*

-   **Unverifiable Tasks (Evaluator-Optimizer)**

    ---

    ![Unverifiable Evaluator](../static/img/use-cases/unverifiable_evaluator.png){ .card-image }

    GEPA for **unverifiable tasks** using evaluator-optimizer patterns where ground truth is unavailable.

    [:material-arrow-right: View example](https://x.com/AsfiShaheen/status/1967866903331999807)

</div>

---

## :material-trophy: Research & Academic

<div class="grid cards" markdown>

-   **Berkeley AI Summit: GEPA Deep Dive**

    ---

    @matei_zaharia presents GEPA at Berkeley AI Summit, explaining how reflective prompt evolution works even with few rollouts.

    **Key Insight:**

    > "FLOPs are getting cheaper, but rollouts for complex agentic tasks are not. The next frontier of AI will be limited by rollouts budget!"

    [:material-arrow-right: Watch the presentation](https://youtu.be/c39fJ2WAj6A?t=6386)

-   **NeurIPS 2025 Workshop: 12.5% → 62.5% Gains**

    ---

    ![NeurIPS Poster](../static/img/use-cases/neurips_poster.jpg){ .card-image }

    Veris.AI used GEPA in their RAISE framework to achieve **12.5% → 62.5% gains** in task correctness accuracy, demonstrating GEPA's immediate practical impact for training reliable domain-specific AI agents through simulated environments.

    **Key Results:**

    - RAISE: Simulation-first experiential learning framework
    - GEPA prompt optimization for 4 epochs
    - Poster session at NeurIPS 2025 San Diego

    [:material-arrow-right: View the announcement](https://x.com/solidwillity/status/1997747867629633971)

-   **100% on Clock Hands Problem**

    ---

    Achieving **perfect accuracy** on the challenging clock hands mathematical reasoning problem using GEPA optimization.

    **Application:**

    - Complex spatial reasoning
    - Mathematical problem-solving
    - Demonstrates GEPA on hard reasoning tasks

    [:material-arrow-right: Try the notebook](https://colab.research.google.com/drive/1W-XNxKL2CXFoUTwrL7GLCZ7J7uZgXsut?usp=sharing)

</div>

---

## :material-newspaper: Media & Press Coverage

<div class="grid cards" markdown>

-   **VentureBeat: GEPA Optimizes LLMs Without Costly RL**

    ---

    VentureBeat coverage of GEPA's approach to optimizing LLMs without expensive reinforcement learning.

    **Highlights:**

    - Explains reflective prompt evolution to a broader audience
    - Discusses cost and efficiency benefits
    - Industry perspective on GEPA's impact

    [:material-arrow-right: Read the article](https://venturebeat.com/ai/gepa-optimizes-llms-without-costly-reinforcement-learning/)

-   **DAIR.AI: Top AI Papers of the Week**

    ---

    GEPA featured in DAIR.AI's "Top AI Papers of The Week" roundup, alongside other breakthrough research.

    **Recognition:**

    - Listed among Graph-R1, AlphaEarth, Self-Evolving Agents
    - Highlighted natural language reflection approach

    [:material-arrow-right: View DAIR.AI newsletter](https://github.com/dair-ai/ML-Papers-of-the-Week)

-   **DSPy Weekly Newsletter**

    ---

    GEPA regularly featured in the DSPy Weekly newsletter, tracking adoption and new use cases.

    **Coverage:**

    - Issue #4: "GEPA is 🌶️🔥 and on a hype 🚄 as people discover GEPA"
    - Regular updates on community applications

    [:material-arrow-right: Read DSPy Weekly](https://dspyweekly.com/newsletter/4/)

-   **LinkedIn AI Talk: Automatic Prompt Optimization**

    ---

    Vaibhav Gupta (CEO @ Boundary / BAML) provides a detailed GEPA tutorial, first explaining the algorithm and then walking through a code example.

    **What's Covered:**

    - GEPA algorithm explanation
    - Step-by-step code walkthrough
    - Practical implementation guidance

    [:material-arrow-right: Watch the event](https://www.linkedin.com/events/automaticpromptoptimization-ait7404883890873618433/theater/)

</div>

---

## :material-trending-up: Emerging Applications

<div class="grid cards" markdown>

-   **The State of AI Coding 2025**

    ---

    ![Greptile State of AI Coding](../static/img/use-cases/greptile_state_of_ai_coding.jpg){ .card-image }

    GEPA was highlighted in Greptile's comprehensive **State of AI Coding 2025** report as a key advancement in AI coding capabilities.

    **Key Insight:**

    GEPA evolves prompts via trace analysis, matching RL performance with far fewer rollouts—making it ideal for coding agent optimization.

    [:material-arrow-right: Read the report](https://greptile.com/state-of-ai-coding-2025)

-   **Model Migration Workflows**

    ---

    GEPA is proving valuable for **migrating existing LLM-based workflows** to new models across model families.

    **Pattern:**

    - Keep your DSPy program structure
    - Change only the LM initialization  
    - Re-run GEPA optimization for the new model
    - Much faster than manually re-tuning prompts

    This is especially useful as new models are released and organizations need to migrate quickly.

-   **Evaluator-Optimizer Pattern**

    ---

    ![Evaluator Optimizer](../static/img/use-cases/evaluator_optimizer.jpg){ .card-image }

    @hammer_mt shares the powerful **Evaluator-Optimizer pattern** for fuzzy generative tasks where evals are informal and subjective.

    **Use Case:**

    - Creative writing tasks
    - Persona generation
    - Tasks without ground-truth labels

    [:material-arrow-right: Watch the talk](https://www.youtube.com/watch?v=H4o7h6ZbA4o)

-   **Program Synthesis & Kernel Optimization**

    ---

    GEPA shows promise for **program synthesis** tasks:

    **Applications:**

    - CUDA kernel optimization
    - AMD NPU kernel generation
    - Outperforms RAG and iterative refinement (Section 6 of paper)

    Especially valuable for tasks with expensive rollouts (simulation, long runtime).

-   **GPU Parallelization (OpenACC)**

    ---

    ![GPU Optimization](../static/img/use-cases/gpu_optimization.png){ .card-image }

    Jhaveri & @cristalopes applied GEPA to **GPU optimization**, targeting OpenACC parallelization.

    **Results:**

    - Boosted GPT-5 Nano to generate pragmas improving compilation success from 87% → 100%
    - Models saw up to **50% increase** in # functional GPU speedups

    Demonstrates GEPA's applicability to code synthesis beyond prompts.

-   **Material Science Applications**

    ---

    GEPA being explored for **material science** workflows where simulations are costly.

    **Why GEPA:**

    - High sample efficiency
    - Works with expensive evaluation functions
    - Can optimize simulation parameters

    *Exploratory use case from the research community*

-   **Continuous Learning & Self-Improvement**

    ---

    GEPA enables **continual learning** patterns:

    **Emerging Pattern:**

    1. Deploy optimized agent
    2. Collect feedback from production
    3. Batch feedback and re-optimize
    4. Redeploy improved agent

    Works alongside RL (see BetterTogether paper) for even better results.

-   **Letta: Continual Learning in Token Space**

    ---

    Letta's blog post explores **continual learning in token space**, discussing how GEPA and similar approaches enable agents to learn and improve over time.

    **Concepts:**

    - Memory-augmented agents
    - Long-term learning patterns
    - Token-space optimization

    [:material-arrow-right: Read the blog](https://www.letta.com/blog/continual-learning)

</div>

---

## :material-source-branch: Community Integrations

<div class="grid cards" markdown>

-   **Weaviate Podcast #127: Deep Dive on GEPA**

    ---

    Comprehensive podcast episode covering GEPA in depth with Lakshya A. Agrawal.

    **Topics Covered:**

    - Natural Language Rewards
    - Reflective prompt evolution principles
    - Production deployment patterns

    [:material-arrow-right: Listen to podcast](https://www.youtube.com/watch?v=rrtxyZ4Vnv8)

-   **Weaviate GEPA Hands-On Notebook**

    ---

    Interactive notebook demonstrating GEPA for reranking optimization in RAG pipelines.

    **What's Inside:**

    - End-to-end GEPA optimization
    - Integration with Weaviate vector store
    - Practical reranking examples

    [:material-arrow-right: View notebook](https://github.com/weaviate/recipes/blob/main/integrations/llm-agent-frameworks/dspy/GEPA-Hands-On-Reranker.ipynb)

-   **LangStruct GEPA Examples**

    ---

    Strong examples demonstrating GEPA's effectiveness with Gemini Flash and other models.

    [:material-arrow-right: Explore examples](https://langstruct.dev/examples/gepa/)

-   **GEPA in Go**

    ---

    Full Go implementation of DSPy concepts including GEPA optimization.

    **Features:**

    - Native Go implementation
    - MIT licensed
    - Includes CLI tools and examples

    [:material-arrow-right: View on GitHub](https://github.com/XiaoConstantine/dspy-go)

-   **Observable JavaScript**

    ---

    ![Observable JavaScript](../static/img/use-cases/observable_js.jpeg){ .card-image }

    Interactive JavaScript notebooks exploring GEPA for web-based optimization.

    **By Tom Larkworthy** (Tech Lead, formerly Firebase/Google)

    Explore reflective prompt evolution directly in your browser.

    [:material-arrow-right: Try it on Observable](https://observablehq.com/@tomlarkworthy/gepa)

-   **Context Compression**

    ---

    Experiments using GEPA for **context compression** to reduce token usage while maintaining quality.

    Explore novel approaches to efficient prompt engineering.

    [:material-arrow-right: View experiments](https://github.com/Laurian/context-compression-experiments-2508)

-   **bandit_dspy**

    ---

    DSPy library for **security-aware LLM development** using Bandit principles.

    Part of the EvalOps ecosystem for AI evaluation and development tools.

    [:material-arrow-right: Explore on GitHub](https://github.com/evalops/bandit_dspy)

-   **SuperOptiX-AI**

    ---

    SuperOptiX uses GEPA as its **framework-agnostic optimizer** across multiple agent frameworks including DSPy, OpenAI SDK, CrewAI, Google ADK, and more.

    [:material-arrow-right: Explore SuperOptiX](https://superagenticai.github.io/superoptix-ai/guides/gepa-optimization/)

    [:material-arrow-right: Read the blog post](https://super-agentic.ai/resources/super-posts/gepa-dspy-optimizer-superoptix-revolutionizing-ai-agent-optimization)

</div>

---

## :material-cloud-outline: Infrastructure & DevOps

<div class="grid cards" markdown>

-   **Multi-Cloud Data Transfer Cost Optimization**

    ---

    ![Multi-Cloud Data Transfer](../static/img/use-cases/multi_cloud_transfer.jpeg){ .card-image }

    The ADRS team used GEPA to minimize multi-cloud data transfer costs.

    **Results:**

    - GEPA autonomously evolved a naive replication strategy into a sophisticated "shared-tree" topology
    - **31% cost reduction** with just $5 of optimization spend
    - Demonstrates GEPA's ability to optimize complex infrastructure configurations

    [:material-arrow-right: View research](https://x.com/LakshyAAAgrawal/status/2014459447364694154)

-   **Sales Support Multi-Agent Routing**

    ---

    ![Databricks Sales Support](../static/img/use-cases/sales_support_routing.jpeg){ .card-image }

    Databricks used GEPA to optimize a sales-support multi-agent system's routing component.

    **Key Results:**

    - **75% relative gains** in routing accuracy
    - Demonstrates multi-agent orchestration optimization
    - Production-ready deployment patterns

    [:material-arrow-right: Read the blog](https://medium.com/@AI-on-Databricks/multi-ai-powered-sales-support-databricks-with-langchain-gepa-prompt-optimization-8104654bb538)

-   **Self-Improving Agent Systems (GEPA + TRM)**

    ---

    Building self-improving AI agents that combine GEPA with TRM (Test-time Reasoning Modification) for both orchestration optimization and reasoning enhancement.

    **Architecture:**

    - GEPA for orchestration/prompt optimization
    - TRM for reasoning enhancement
    - Continuous monitoring and feedback loops
    - Automated retraining without human intervention

    [:material-arrow-right: Read the guide](https://medium.com/@bindupriya117/building-self-improving-ai-agents-gepa-for-orchestration-trm-for-reasoning-1602e96f3e2b)

</div>

---

## :material-account-voice: Creative & Generative Applications

<div class="grid cards" markdown>

-   **AI Voice/Persona Discovery**

    ---

    GEPA's multi-objective guided optimization can find an authentic "AI voice" using an 8-dimensional score representing different voice characteristics.

    **Dimensions Optimized:**

    - Point of view
    - Authority level
    - Cadence and rhythm
    - And 5 more characteristics

    [:material-arrow-right: Read the guide](https://augchan42.github.io/2025/09/02/dspy-voice-evolution-authenticity/)

-   **Human-Like Response Generation**

    ---

    GEPA+DSPy can optimize AI to generate human-like responses, passing sophisticated detection systems.

    **Application:**

    - More natural conversational AI
    - Better user engagement
    - Authentic persona maintenance

    *Community-reported application*

-   **Non-Obvious GEPA Insights**

    ---

    ![Non-Obvious GEPA Insights](../static/img/use-cases/non_obvious_insights.png){ .card-image }

    Deep dive into non-obvious lessons learned from practical GEPA usage, covering edge cases, unexpected behaviors, and advanced patterns.

    [:material-arrow-right: Read the blog](https://www.elicited.blog/posts/non-obvious-things-about-gepa)

</div>

---

## :material-database-sync: Data Processing & Synthesis

<div class="grid cards" markdown>

-   **Synthetic Data Generation**

    ---

    Use GEPA to optimize query generation pipelines for creating high-quality synthetic datasets.

    **Example: Sanskrit NLP**

    - GEPA+DSPy optimizes a query generation pipeline
    - Differentiates between document pairs
    - Generated 50k samples for Gemma embedding fine-tuning

    [:material-arrow-right: View project](https://github.com/ganarajpr/rgfe)

-   **Text2SQL Optimization**

    ---

    GEPA has been successfully used for Text2SQL tasks with a system prompt/user prompt breakdown.

    **Pattern:**

    - System prompt specifies the task (evolved by GEPA)
    - User prompt contains dynamic content
    - Alternatively: use DSPy signature for text2sql

    [:material-arrow-right: See SQL Generator tutorial](https://www.rajapatnaik.com/blog/2025/10/20/sql-generator)

-   **Enterprise Agents Blog**

    ---

    ![Enterprise Agents](../static/img/use-cases/enterprise_agents_blog.jpg){ .card-image }

    Building enterprise agents for real-world workflows with GEPA: tackling unstructured data, task decomposition, and context blowup.

    **Key Topics:**

    - Modular agent design
    - Low-data optimization strategies
    - Cost-effective deployment

    [:material-arrow-right: Read the blog](https://slavozard.bearblog.dev/experiences-from-building-enterprise-agents-with-dspy-and-gepa/)

</div>

---

## :material-book-open-variant: Community Tutorials & Guides

<div class="grid cards" markdown>

-   **DSPy 3 + GEPA: Advanced RAG Framework**

    ---

    Comprehensive guide on building powerful AI agents with DSPy 3 and GEPA.

    **What's Covered:**

    - Auto reasoning and prompting
    - Step-by-step agent building
    - Professional-level RAG optimization

    [:material-arrow-right: Read the guide](https://gaodalie.substack.com/p/dspy-3-gepa-the-most-advanced-rag)

-   **Teaching AI to Spot Fake XKCD Comics**

    ---

    ![Teaching AI to Spot Fake XKCD](../static/img/use-cases/xkcd_fake_detection.png){ .card-image }

    Fun, accessible explanation of GEPA concepts with XKCD-inspired visualizations.

    [:material-arrow-right: Read the blog](https://danprice.ai/blog/xkcd-dspy-gepa)

-   **20% Improvement in Structured Extraction with DSPy + GEPA**

    ---

    ![DSPy Optimization](../static/img/use-cases/dspy_structured_extraction.png){ .card-image }

    Achieving **20+ percentage-point improvement** in exact match accuracy for structured extraction tasks using DSPy and GEPA.

    **Key Insight:**

    The benefit is not only improved performance, but that optimization allows transferring capability to cheaper models while retaining acceptable accuracy, improving the cost profile of applications.

    [:material-arrow-right: Read the guide](https://kmad.ai/DSPy-Optimization)

-   **GEPA Impact Analysis: 81% → 90% Accuracy**

    ---

    Practical analysis achieving **81% → 90% accuracy** on sales call transcript analysis in just 3 hours and ~$0.50.

    **Key Insight:**

    > "I stopped thinking of prompts as things I _write_ and started thinking of them as things I _evolve_. My job shifted from 'craft the perfect prompt' to 'define what good looks like and let the system find it.'"

    GEPA's genetic mutations work best with precise feedback—targeted feedback like "you're conflating greetings with rapport" produces targeted fixes.

    [:material-arrow-right: Read the analysis](https://risheekkumar.in/posts/gepa-impact/gepa_impact_final.html)

-   **Lakshya's GEPA Blog**

    ---

    Personal blog post explaining GEPA concepts and applications.

    [:material-arrow-right: Read the blog](https://lakshyaag.com/blogs/gepa)

-   **GEPA for De-identification**

    ---

    Tutorial on using GEPA for PII de-identification tasks with DSPy.

    [:material-arrow-right: Read the tutorial](https://www.rajapatnaik.com/blog/2025/10/14/dspy-gepa-deidentification)

-   **SQL Generator with GEPA**

    ---

    Building optimized Text2SQL systems with DSPy and GEPA.

    [:material-arrow-right: Read the tutorial](https://www.rajapatnaik.com/blog/2025/10/20/sql-generator)

</div>

---

## :material-earth: International Coverage

GEPA has gained significant attention in the global AI community, with tutorials, blogs, and discussions in multiple languages.

<div class="grid cards" markdown>

-   **Japanese AI Community**

    ---

    GEPA has seen strong adoption in the Japanese AI community with multiple tutorials and explanations.

    **Resources:**

    - [GEPA Explained (Japanese)](https://youtu.be/P5mW0IbotlY) - Video explaining GEPA's reflective learning approach
    - [MLflow + GEPA on Databricks Free Edition](https://qiita.com/isanakamishiro2/items/f15c4c4c79bd22222ccf) - Qiita tutorial
    - [Naruto-Style Dialogues with GEPA](https://zenn.dev/cybernetics/articles/39fb763aca746c) - Creative application
    - Multiple AI Daily News Japan features

-   **Chinese AI Community**

    ---

    GEPA has been featured in Chinese AI publications and discussions.

    **Resources:**

    - [GEPA Revolutionary Breakthrough](https://jieyibu.net/a/65905) - 35x efficiency improvement explained
    - Technical translations and explanations

</div>

---

## :material-rocket-launch: Get Started

Ready to optimize your own AI systems with GEPA?

<div class="grid cards" markdown>

-   **Quick Start Guide**

    ---

    Get up and running with GEPA in minutes.

    [:material-arrow-right: Start here](quickstart.md)

-   **Create Custom Adapters**

    ---

    Integrate GEPA with your specific system.

    [:material-arrow-right: Learn adapters](adapters.md)

-   **API Reference**

    ---

    Complete documentation of all GEPA components.

    [:material-arrow-right: View API](../api/index.md)

-   **Join the Community**

    ---

    Connect with other GEPA users and contributors.

    [:material-arrow-right: Discord](https://discord.gg/WXFSeVGdbW) [:material-arrow-right: Slack](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w)

</div>
