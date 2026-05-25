# GEPA Adapters

> GEPA 🤝 Any Framework

This directory provides the interface to allow GEPA to plug into systems and frameworks of your choice! GEPA can interface with any system consisting of text components, by implementing `GEPAAdapter` in [../core/adapter.py](../core/adapter.py).

Currently, GEPA has the following adapters:
- [DSPy Adapter](./dspy_adapter/): This adapter integrates GEPA into [DSPy](https://dspy.ai/), to allow it to optimize any DSPy module's signature instructions.
- [Default Adapter](./default_adapter/): This adapter integrates GEPA into a single-turn LLM environment, where the task is specified as a user message, and an answer string must be present in the assistant response. GEPA optimizes the system prompt.
- [AnyMaths Adapter](./anymaths_adapter/): This adapter integrates GEPA with litellm and ollama to solve single-turn mathematical problems.
- [LangChain Adapter](./langchain_adapter/): This adapter integrates GEPA with [LangChain](https://python.langchain.com/) — works with any chat model via `init_chat_model`, plus tool-using agents (`create_agent`), LangGraph graphs, and RAG pipelines.

If there are any frameworks you would like GEPA integrated into, please create an issue or PR!
