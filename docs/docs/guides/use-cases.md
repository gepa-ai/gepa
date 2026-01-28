# GEPA in Action

Discover how organizations and researchers are using GEPA to optimize AI systems across diverse domains. These examples showcase the versatility and impact of reflective prompt evolution.

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

-   **Sanskrit QA with EmbeddingGemma**

    ---

    Boosting Sanskrit question-answering by finetuning EmbeddingGemma with **50k GEPA-generated synthetic samples**.

    **Approach:**

    - Synthetic data generation using GEPA
    - Fine-tuned embedding models
    - Improved performance on low-resource language tasks

    [:material-arrow-right: View code](https://github.com/ganarajpr/rgfe)

-   **Fiction Writing with Small Models**

    ---

    ![Creative Writing](../static/img/use-cases/creative_writing.png){ .card-image }

    Teaching **Gemma3-1B** to write engaging fiction through GEPA optimization.

    Demonstrates that small models can handle creative tasks with the right prompts.

    [:material-arrow-right: Read on Substack](https://meandnotes.substack.com/p/i-taught-a-small-llm-to-write-fiction?triedRedirect=true)

-   **Naruto-Style Dialogues with GPT-4o-mini**

    ---

    Generating anime-style character dialogues using GEPA-optimized prompts.

    A creative application demonstrating GEPA's versatility for stylized content generation.

    [:material-arrow-right: Read the article (Japanese)](https://zenn.dev/cybernetics/articles/39fb763aca746c)

</div>

---

## :material-lightbulb: Advanced Capabilities

<div class="grid cards" markdown>

-   **Multimodal/VLM Performance (OCR)**

    ---

    ![Multimodal OCR](../static/img/use-cases/multimodal_ocr.png){ .card-image }

    GEPA improves **Multimodal/VLM Performance** for OCR tasks through optimized prompting strategies.

    [:material-arrow-right: Try the example](https://tinyurl.com/gepa-ocr-example)

-   **Agent Architecture Discovery**

    ---

    ![Architecture Discovery](../static/img/use-cases/architecture_discovery.png){ .card-image }

    GEPA for **automated agent architecture discovery** - finding optimal agent designs through evolutionary search.

-   **Adversarial Prompt Search**

    ---

    ![Adversarial Prompt Search](../static/img/use-cases/adversarial_prompt.png){ .card-image }

    GEPA for **adversarial prompt search** - discovering edge cases and failure modes in AI systems.

-   **Unverifiable Tasks (Evaluator-Optimizer)**

    ---

    ![Unverifiable Evaluator](../static/img/use-cases/unverifiable_evaluator.png){ .card-image }

    GEPA for **unverifiable tasks** using evaluator-optimizer patterns where ground truth is unavailable.

    [:material-arrow-right: View example](https://x.com/AsfiShaheen/status/1967866903331999807)

</div>

---

## :material-school: Learning Resources & Tutorials

<div class="grid cards" markdown>

-   **Official DSPy Tutorials**

    ---

    Step-by-step notebooks for practical optimization tasks.

    **Topics Covered:**

    - **AIME Math**: 10% gains on AIME 2025 with GPT-4.1 Mini
    - **Structured Data Extraction**: Enterprise facility support analysis
    - **Privacy-Conscious Delegation**: Secure task handling

    [:material-arrow-right: Start learning](https://dspy.ai/tutorials/gepa_ai_program)

-   **Video Tutorials**

    ---

    Visual learning resources from the community.

    **Featured Videos:**

    - 🎬 **Weaviate**: DSPy GEPA for Listwise Reranker optimization
    - 🎬 **Matei Zaharia**: Reflective Optimization of Agents with GEPA and DSPy

    [:material-arrow-right: Weaviate Video](https://www.youtube.com/watch?v=H4o7h6ZbA4o)

    [:material-arrow-right: Matei Zaharia Talk](https://www.youtube.com/watch?v=rrtxyZ4Vnv8)

-   **100% on Clock Hands Problem**

    ---

    Achieving perfect accuracy on the challenging clock hands mathematical reasoning problem.

    [:material-arrow-right: Try the notebook](https://colab.research.google.com/drive/1W-XNxKL2CXFoUTwrL7GLCZ7J7uZgXsut?usp=sharing)

</div>

---

## :material-source-branch: Community Integrations

<div class="grid cards" markdown>

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

    [:material-arrow-right: Discord](https://discord.gg/A7dABbtmFw) [:material-arrow-right: Slack](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w)

</div>
