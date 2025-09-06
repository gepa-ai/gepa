#!/usr/bin/env python3
"""
GEPA ChromaDB RAG Optimization Example

This example demonstrates how to use GEPA to optimize a RAG system using ChromaDB
as the vector store. It supports both local Ollama models and cloud-based LLMs.

Usage:
    # Basic usage with Ollama (recommended for getting started)
    python chromadb_optimization.py

    # With specific model
    python chromadb_optimization.py --model ollama/llama3.1:8b

    # With cloud model (requires API key)
    python chromadb_optimization.py --model gpt-4o-mini

    # Full optimization run
    python chromadb_optimization.py --max-iterations 20

Requirements:
    pip install gepa[rag] chromadb

For Ollama setup:
    ollama pull qwen3:8b
    ollama pull nomic-embed-text:latest
"""

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path

# Suppress all warnings
warnings.filterwarnings("ignore")

os.environ["PYTHONWARNINGS"] = "ignore"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import gepa
from gepa.adapters.generic_rag_adapter import (
    ChromaVectorStore,
    GenericRAGAdapter,
    RAGDataInst,
)


def create_llm_client(model_name: str):
    """Create LLM client supporting both Ollama and cloud models."""
    try:
        import litellm

        litellm.drop_params = True
        litellm.set_verbose = False
    except ImportError:
        raise ImportError("LiteLLM is required. Install with: pip install litellm")

    def llm_client(messages_or_prompt, **kwargs):
        try:
            # Handle both string prompts and message lists
            if isinstance(messages_or_prompt, str):
                # Convert string prompt to messages format
                messages = [{"role": "user", "content": messages_or_prompt}]
            else:
                # Use as-is if it's already in messages format
                messages = messages_or_prompt

            params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 400),
                "temperature": kwargs.get("temperature", 0.1),
            }

            if "ollama/" in model_name:
                params["request_timeout"] = 120

            response = litellm.completion(**params)
            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"Error: Unable to generate response ({e!s})"

    return llm_client


def setup_chromadb_knowledge_base():
    """Create ChromaDB vector store with AI/ML knowledge base."""
    try:
        from chromadb.utils import embedding_functions
    except ImportError:
        raise ImportError("ChromaDB is required. Install with: pip install chromadb")

    # Create temporary directory for this example
    temp_dir = tempfile.mkdtemp()
    print(f"📁 ChromaDB directory: {temp_dir}")

    # Initialize ChromaDB with default embedding function
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    vector_store = ChromaVectorStore.create_local(
        persist_directory=temp_dir, collection_name="ai_ml_knowledge", embedding_function=embedding_function
    )

    # AI/ML knowledge base articles
    documents = [
        {
            "content": "Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
            "metadata": {"doc_id": "ml_basics", "topic": "machine_learning", "difficulty": "beginner"},
        },
        {
            "content": "Deep Learning is a subset of machine learning based on artificial neural networks with representation learning. It can learn from data that is unstructured or unlabeled. Deep learning models are inspired by information processing patterns found in biological neural networks.",
            "metadata": {"doc_id": "dl_basics", "topic": "deep_learning", "difficulty": "intermediate"},
        },
        {
            "content": "Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.",
            "metadata": {"doc_id": "nlp_basics", "topic": "nlp", "difficulty": "intermediate"},
        },
        {
            "content": "Computer Vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.",
            "metadata": {"doc_id": "cv_basics", "topic": "computer_vision", "difficulty": "intermediate"},
        },
        {
            "content": "Reinforcement Learning is an area of machine learning where an agent learns to behave in an environment by performing actions and seeing the results. The agent receives rewards by performing correctly and penalties for performing incorrectly.",
            "metadata": {"doc_id": "rl_basics", "topic": "reinforcement_learning", "difficulty": "advanced"},
        },
        {
            "content": "Large Language Models (LLMs) are a type of artificial intelligence model designed to understand and generate human-like text. They are trained on vast amounts of text data and can perform various natural language tasks such as translation, summarization, and question answering.",
            "metadata": {"doc_id": "llm_basics", "topic": "large_language_models", "difficulty": "advanced"},
        },
    ]

    # Add documents to ChromaDB
    vector_store.collection.add(
        documents=[doc["content"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents],
        ids=[doc["metadata"]["doc_id"] for doc in documents],
    )

    print(f"✅ Created ChromaDB knowledge base with {len(documents)} articles")
    return vector_store


def create_training_data() -> tuple[list[RAGDataInst], list[RAGDataInst]]:
    """Create training and validation datasets for RAG optimization."""

    # Training examples
    train_data = [
        RAGDataInst(
            query="What is machine learning?",
            ground_truth_answer="Machine Learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
            relevant_doc_ids=["ml_basics"],
            metadata={"category": "definition", "difficulty": "beginner"},
        ),
        RAGDataInst(
            query="How does deep learning work?",
            ground_truth_answer="Deep Learning is a subset of machine learning based on artificial neural networks with representation learning. It can learn from data that is unstructured or unlabeled. Deep learning models are inspired by information processing patterns found in biological neural networks.",
            relevant_doc_ids=["dl_basics"],
            metadata={"category": "explanation", "difficulty": "intermediate"},
        ),
        RAGDataInst(
            query="What is natural language processing?",
            ground_truth_answer="Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics.",
            relevant_doc_ids=["nlp_basics"],
            metadata={"category": "definition", "difficulty": "intermediate"},
        ),
    ]

    # Validation examples
    val_data = [
        RAGDataInst(
            query="Explain computer vision in AI",
            ground_truth_answer="Computer Vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.",
            relevant_doc_ids=["cv_basics"],
            metadata={"category": "explanation", "difficulty": "intermediate"},
        ),
        RAGDataInst(
            query="What are large language models?",
            ground_truth_answer="Large Language Models (LLMs) are a type of artificial intelligence model designed to understand and generate human-like text. They are trained on vast amounts of text data and can perform various natural language tasks such as translation, summarization, and question answering.",
            relevant_doc_ids=["llm_basics"],
            metadata={"category": "definition", "difficulty": "advanced"},
        ),
    ]

    return train_data, val_data


def clean_answer(answer: str) -> str:
    """Clean up LLM answer by removing thinking tokens and truncating appropriately."""
    # Remove thinking tokens if present
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    cleaned = cleaned.strip()

    # If still empty or starts with <think> without closing tag, try to find content after
    if not cleaned or cleaned.startswith("<think>"):
        # Look for content after thinking section
        lines = answer.split("\n")
        content_lines = []
        skip_thinking = False

        for line in lines:
            if "<think>" in line:
                skip_thinking = True
                continue
            if "</think>" in line:
                skip_thinking = False
                continue
            if not skip_thinking and line.strip():
                content_lines.append(line.strip())

        cleaned = " ".join(content_lines)

    # Show more of the answer - increase limit significantly
    if len(cleaned) > 500:
        return cleaned[:500] + "..."
    return cleaned or answer[:500] + ("..." if len(answer) > 500 else "")


def create_initial_prompts() -> dict[str, str]:
    """Create initial prompt templates for optimization."""
    return {
        "answer_generation": """You are an AI expert providing accurate technical explanations.

Based on the retrieved context, provide a clear and informative answer to the user's question.

Guidelines:
- Use information from the provided context
- Be accurate and concise
- Include key technical details
- Structure your response clearly

Context: {context}

Question: {query}

Answer:"""
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GEPA ChromaDB RAG Optimization Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chromadb_optimization.py
  python chromadb_optimization.py --model ollama/llama3.1:8b
  python chromadb_optimization.py --model gpt-4o-mini --max-iterations 10
        """,
    )

    parser.add_argument("--model", type=str, default="ollama/qwen3:8b", help="LLM model (default: ollama/qwen3:8b)")
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="ollama/nomic-embed-text:latest",
        help="Embedding model (default: ollama/nomic-embed-text:latest)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="GEPA optimization iterations (default: 5, use 0 to skip optimization)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main():
    """Main function demonstrating ChromaDB RAG optimization."""
    args = parse_arguments()

    print("🚀 GEPA ChromaDB RAG Optimization")
    print("=" * 50)
    print(f"📊 Model: {args.model}")
    print(f"🔗 Embeddings: {args.embedding_model}")
    print(f"🔄 Max Iterations: {args.max_iterations}")

    try:
        # Step 1: Setup ChromaDB knowledge base
        print("\n1️⃣ Setting up ChromaDB knowledge base...")
        vector_store = setup_chromadb_knowledge_base()

        # Step 2: Create datasets
        print("\n2️⃣ Creating training and validation datasets...")
        train_data, val_data = create_training_data()
        print(f"   📚 Training examples: {len(train_data)}")
        print(f"   📝 Validation examples: {len(val_data)}")

        # Step 3: Initialize LLM client
        print(f"\n3️⃣ Initializing LLM client ({args.model})...")
        llm_client = create_llm_client(args.model)

        # Test LLM
        test_response = llm_client([{"role": "user", "content": "Say 'OK' only."}])
        if "Error:" not in test_response:
            print(f"   ✅ LLM connected: {test_response[:30]}...")
        else:
            print(f"   ⚠️ LLM issue: {test_response}")

        # Step 4: Initialize RAG adapter
        print("\n4️⃣ Initializing GenericRAGAdapter...")
        rag_adapter = GenericRAGAdapter(
            vector_store=vector_store,
            llm_model=llm_client,
            embedding_model=args.embedding_model,
            rag_config={
                "retrieval_strategy": "similarity",
                "top_k": 3,
                "retrieval_weight": 0.3,
                "generation_weight": 0.7,
            },
        )

        # Step 5: Create initial prompts
        print("\n5️⃣ Creating initial prompts...")
        initial_prompts = create_initial_prompts()

        # Step 6: Test initial performance
        print("\n6️⃣ Testing initial performance...")
        eval_result = rag_adapter.evaluate(batch=val_data[:1], candidate=initial_prompts, capture_traces=True)

        initial_score = eval_result.scores[0]
        print(f"   📊 Initial score: {initial_score:.3f}")
        print(f"   💬 Sample answer: {clean_answer(eval_result.outputs[0]['final_answer'])}")

        # Step 7: Run GEPA optimization
        if args.max_iterations > 0:
            print(f"\n7️⃣ Running GEPA optimization ({args.max_iterations} iterations)...")

            result = gepa.optimize(
                seed_candidate=initial_prompts,
                trainset=train_data,
                valset=val_data,
                adapter=rag_adapter,
                reflection_lm=llm_client,
                max_metric_calls=args.max_iterations,
            )

            best_score = result.val_aggregate_scores[result.best_idx]
            print("   🎉 Optimization complete!")
            print(f"   🏆 Best score: {best_score:.3f}")
            print(f"   📈 Improvement: {best_score - initial_score:+.3f}")
            print(f"   🔄 Total iterations: {result.total_metric_calls or 0}")

            # Test optimized prompts
            print("\n   Testing optimized prompts...")
            optimized_result = rag_adapter.evaluate(
                batch=val_data[:1], candidate=result.best_candidate, capture_traces=False
            )
            print(f"   💬 Optimized answer: {clean_answer(optimized_result.outputs[0]['final_answer'])}")

        else:
            print("\n7️⃣ Skipping optimization (use --max-iterations > 0 to enable)")

        print("\n✅ ChromaDB RAG optimization completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()

        print("\n🔧 Troubleshooting tips:")
        print("  • Ensure Ollama is running: ollama list")
        print("  • Check models are available: ollama pull qwen3:8b")
        print("  • For cloud models: set API keys (OPENAI_API_KEY, etc.)")
        print("  • Try smaller model: --model ollama/qwen3:8b")


if __name__ == "__main__":
    main()
