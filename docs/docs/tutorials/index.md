# Tutorials

Welcome to the GEPA tutorials! These hands-on notebooks will help you learn GEPA through practical examples.

## Available Tutorials

### DSPy Full Program Evolution

Learn how to use GEPA to evolve entire DSPy programs, including custom signatures, modules, and control flow logic.

- **[DSPy Full Program Evolution](dspy_full_program_evolution.ipynb)** - Evolve a complete DSPy program from a basic `ChainOfThought` to a sophisticated multi-step reasoning system. This tutorial demonstrates how GEPA can improve a program from 67% to 93% accuracy on the MATH benchmark.

### ARC AGI Example

- **[ARC AGI Example](arc_agi.ipynb)** - Apply GEPA to the ARC (Abstraction and Reasoning Corpus) challenge, demonstrating how to optimize programs for complex reasoning tasks.

### 3D Unicorn Optimization (Seedless)

Use GEPA's `seed_candidate=None` mode to evolve a complete Python program (build123d + pyrender) that generates a 3D unicorn — starting from no code at all. This tutorial demonstrates how to optimize when you know *what good looks like* (evaluation aspects) but don't know *how to get started* (no seed). GEPA bootstraps the first candidate from a natural-language objective, then iteratively refines it using VLM feedback on rendered multi-view images.

- **[3D Unicorn Optimization](3d_unicorn_optimization.ipynb)** - From no code to a 600+ line 3D modeling program through seedless optimization with visual feedback.

## External Tutorials

For more tutorials, especially those focused on the DSPy integration, see:

### Official DSPy Tutorials

- [dspy.GEPA Tutorials](https://dspy.ai/tutorials/gepa_ai_program/) - Official DSPy tutorials with executable notebooks
- [GEPA for AIME (Math)](https://dspy.ai/tutorials/gepa_aime/) - Optimize prompts for competition math (+10% improvement on AIME 2025)
- [GEPA for Structured Information Extraction](https://dspy.ai/tutorials/gepa_facilitysupportanalyzer/) - Enterprise task optimization
- [GEPA for Privacy-Conscious Delegation](https://dspy.ai/tutorials/gepa_papillon/) - Papillon benchmark (+9% with just 3 examples)
- [GEPA for Code Backdoor Classification](https://dspy.ai/tutorials/gepa_trusted_monitor/) - AI control applications
- [GEPA Advanced Guide](https://dspy.ai/api/optimizers/GEPA/GEPA_Advanced/) - Multimodal, custom proposers, and advanced configuration

### Community Tutorials & Blogs

- [Non-Obvious Things About GEPA](https://www.elicited.blog/posts/non-obvious-things-about-gepa) - Deep insights and lessons learned by @realsanketp
- [Enterprise Agents with DSPy and GEPA](https://slavozard.bearblog.dev/experiences-from-building-enterprise-agents-with-dspy-and-gepa/) - Production deployment patterns by @slavozard
- [Multi-Agent RAG for Healthcare](https://kargarisaac.medium.com/building-and-optimizing-multi-agent-rag-systems-with-dspy-and-gepa-2b88b5838ce2) - Diabetes and COPD agents
- [Context Engineering for AI Coding Agents](https://medium.com/firebird-technologies/context-engineering-improving-ai-coding-agents-using-dspy-gepa-df669c632766) - Data analysis agent optimization by @ArslanSAAS
- [Teaching Small LLMs to Write Fiction](https://meandnotes.substack.com/p/i-taught-a-small-llm-to-write-fiction) - Creative writing with Gemma3-1B
- [AI Voice Evolution & Authenticity](https://augchan42.github.io/2025/09/02/dspy-voice-evolution-authenticity/) - Multi-objective voice optimization
- [GEPA in Observable JavaScript](https://observablehq.com/@tomlarkworthy/gepa) - Interactive browser-based GEPA by @tomlarkworthy
- [OCR Optimization Research](https://www.intrinsic-labs.ai/research/ocr-gepa-v1.pdf) - 38% error reduction with Gemini models by Intrinsic Labs
- [GEPA for AI Code Safety](https://tinyurl.com/gepa-ai-code-monitor) - Tutorial notebook by @hi_ZachParent
- [Solving Agent Tool Sprawl with DSPy](https://viksit.substack.com/p/solving-agent-tool-sprawl-with-dspy) - GEPA optimization of tools and routes
- [XKCD Comics with DSPy and GEPA](https://danprice.ai/blog/xkcd-dspy-gepa) - Fun application of GEPA
- [Databricks Sales Support Multi-Agent](https://medium.com/@AI-on-Databricks/multi-ai-powered-sales-support-databricks-with-langchain-gepa-prompt-optimization-8104654bb538) - 75% routing accuracy improvement
- [DeepResearch Agent](https://www.rajapatnaik.com/blog/2025/10/23/langgraph-dspy-gepa-researcher) - LangGraph + DSPy + GEPA research system by @RajaPatnaik
- [Self-Improving AI Agents](https://medium.com/@bindupriya117/building-self-improving-ai-agents-gepa-for-orchestration-trm-for-reasoning-1602e96f3e2b) - GEPA for orchestration, TRM for reasoning
- [Context Compression Experiments](https://github.com/Laurian/context-compression-experiments-2508) - GEPA for optimizing context compression prompts by @gridinoc
- [Google ADK Training with GEPA](https://raphaelmansuy.github.io/adk_training/blog/gepa-optimization-tutorial/) - Optimizing Google Agent Development Kit agents
- [Speeding Up a Sudoku Solver with GEPA optimize_anything](https://blog.mariusvach.com/posts/gepa-sudoku-solver) - Use `optimize_anything` to speed up a Python Sudoku solver `optimize_anything`

### International Tutorials

- [GEPA Explained (Japanese Video)](https://youtu.be/P5mW0IbotlY) - AIが反省し始めた？内省的学習法のGEPAの仕組み
- [MLflow + GEPA on Databricks Free Edition (Japanese)](https://qiita.com/isanakamishiro2/items/f15c4c4c79bd22222ccf) - Qiita tutorial
- [Naruto-Style Dialogues with GEPA (Japanese)](https://zenn.dev/cybernetics/articles/39fb763aca746c) - Creative application
- [GEPA Revolutionary Breakthrough (Chinese)](https://jieyibu.net/a/65905) - 35x efficiency improvement explained
- [DSPy + GEPA Tutorial (HuggingFace Cookbook)](https://huggingface.co/learn/cookbook/en/dspy_gepa) - Featured by @TheDojoMX

### Quick Start Tools

- **DSPy + GEPA Skill** - Quick way to try DSPy + GEPA without setup. Simply install and start experimenting! Created by [@raveeshbhalla](https://x.com/raveeshbhalla)
- **[Arbor: Agent Architecture Discovery](https://github.com/Ziems/arbor)** - GEPA-integrated tool for discovering optimal agent architectures, well-integrated into DSPy by @NoahZiems

### Language-Specific Implementations

- [DSPy-Go](https://github.com/XiaoConstantine/dspy-go) - Full Go implementation including GEPA
- [Ax (DSPy in TypeScript)](https://github.com/ax-ai/ax) - GEPA available in TypeScript
- [DSRs - DSPy in Rust](https://github.com/Herumb/dsrs) - Rust implementation targeting the nerdiest users

## Video Tutorials

- [Weaviate: GEPA for Listwise Reranker & Evaluator-Optimizer Pattern](https://www.youtube.com/watch?v=H4o7h6ZbA4o) - Step-by-step optimization tutorial including fuzzy generative tasks by @hammer_mt
- [Matei Zaharia at Berkeley AI Summit](https://www.youtube.com/live/c39fJ2WAj6A?si=moA1Z4tcsWHzMd2u&t=8041) - GEPA and reflective prompt evolution with few rollouts
- [Weaviate Podcast #127: Deep Dive on GEPA](https://www.youtube.com/watch?v=rrtxyZ4Vnv8) - High-level overview with Lakshya A. Agrawal
- [GEPA Paper Walkthrough (ReallyEasyAI)](https://www.youtube.com/watch?v=3Gc9BY0nuXY) - Detailed paper explanation
- [Karl Weinmeister: GEPA Short](https://www.youtube.com/shorts/QGLbYx1OTu8) - Quick overview of GEPA for agent improvement

## Running Tutorials Locally

To run these tutorials locally:

```bash
# Install GEPA with full dependencies
pip install gepa[full]

# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook
```

Then navigate to the tutorial notebook you want to run.

## Prerequisites

Before starting the tutorials, ensure you have:

1. **API Keys**: Most tutorials require an OpenAI API key (or other LLM provider)
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

2. **Python Environment**: Python 3.10+ with GEPA installed
   ```bash
   pip install gepa[full]
   ```

3. **Optional**: Install DSPy for the DSPy-specific tutorials
   ```bash
   pip install dspy
   ```
