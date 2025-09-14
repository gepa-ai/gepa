#!/usr/bin/env python3
"""
GEPA RAG Optimization Example with Multiple Vector Stores

This example demonstrates how to use GEPA to optimize a RAG system using various
vector stores, showcasing their unique capabilities and search methods.

Supported Vector Stores:
- ChromaDB: Local/persistent vector store with simple setup
- LanceDB: Developer-friendly serverless vector database
- Milvus: Cloud-native vector database with Lite mode
- Qdrant: High-performance vector database with advanced filtering
- Weaviate: Vector database with hybrid search capabilities

Usage:
    # ChromaDB (default, no external dependencies)
    python rag_optimization.py --vector-store chromadb

    # LanceDB (local, no Docker required)
    python rag_optimization.py --vector-store lancedb

    # Milvus Lite (local SQLite-based)
    python rag_optimization.py --vector-store milvus

    # Qdrant (in-memory or with Docker)
    python rag_optimization.py --vector-store qdrant

    # Weaviate (requires Docker)
    python rag_optimization.py --vector-store weaviate

    # With specific models
    python rag_optimization.py --vector-store chromadb --model ollama/llama3.1:8b

    # Full optimization run
    python rag_optimization.py --vector-store qdrant --max-iterations 20

Requirements:
    Base: pip install gepa[rag]
    ChromaDB: pip install chromadb
    LanceDB: pip install lancedb pyarrow sentence-transformers
    Milvus: pip install pymilvus sentence-transformers
    Qdrant: pip install qdrant-client
    Weaviate: pip install weaviate-client

Prerequisites:
    - For Ollama: ollama pull qwen3:8b && ollama pull nomic-embed-text:latest
    - For Weaviate: docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.1
    - For Qdrant (optional): docker run -p 6333:6333 qdrant/qdrant
"""

import argparse
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

# Suppress all warnings for clean output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import gepa  # noqa: E402
from gepa.adapters.generic_rag_adapter import GenericRAGAdapter, RAGDataInst  # noqa: E402

# Vector store imports (lazy loaded)
_vector_stores = {}


def lazy_import_vector_store(store_name: str):
    """Lazy import vector store classes to avoid dependency issues."""
    global _vector_stores

    if store_name in _vector_stores:
        return _vector_stores[store_name]

    try:
        if store_name == "chromadb":
            from gepa.adapters.generic_rag_adapter import ChromaVectorStore
            _vector_stores[store_name] = ChromaVectorStore
            return ChromaVectorStore
        elif store_name == "lancedb":
            from gepa.adapters.generic_rag_adapter import LanceDBVectorStore
            _vector_stores[store_name] = LanceDBVectorStore
            return LanceDBVectorStore
        elif store_name == "milvus":
            from gepa.adapters.generic_rag_adapter import MilvusVectorStore
            _vector_stores[store_name] = MilvusVectorStore
            return MilvusVectorStore
        elif store_name == "qdrant":
            from gepa.adapters.generic_rag_adapter import QdrantVectorStore
            _vector_stores[store_name] = QdrantVectorStore
            return QdrantVectorStore
        elif store_name == "weaviate":
            from gepa.adapters.generic_rag_adapter import WeaviateVectorStore
            _vector_stores[store_name] = WeaviateVectorStore
            return WeaviateVectorStore
        else:
            raise ValueError(f"Unknown vector store: {store_name}")
    except ImportError as e:
        raise ImportError(
            f"Failed to import {store_name} dependencies: {e}\n"
            f"Install with: pip install {get_install_command(store_name)}"
        )


def get_install_command(store_name: str) -> str:
    """Get pip install command for vector store dependencies."""
    commands = {
        "chromadb": "chromadb",
        "lancedb": "lancedb pyarrow sentence-transformers",
        "milvus": "pymilvus sentence-transformers",
        "qdrant": "qdrant-client",
        "weaviate": "weaviate-client"
    }
    return commands.get(store_name, "unknown")


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
                messages = [{"role": "user", "content": messages_or_prompt}]
            else:
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
            return f"Error: Unable to generate response ({e})"

    return llm_client


def create_embedding_function():
    """Create embedding function using sentence-transformers as fallback."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return lambda text: model.encode(text)
    except ImportError:
        # Fallback to litellm for embedding
        try:
            import litellm
            def embed_text(text: str):
                try:
                    response = litellm.embedding(model="ollama/nomic-embed-text:latest", input=text)
                    if hasattr(response, "data") and response.data:
                        if hasattr(response.data[0], "embedding"):
                            return response.data[0].embedding
                        elif isinstance(response.data[0], dict) and "embedding" in response.data[0]:
                            return response.data[0]["embedding"]
                    elif isinstance(response, dict):
                        if response.get("data"):
                            return response["data"][0]["embedding"]
                        elif "embedding" in response:
                            return response["embedding"]
                    raise ValueError(f"Unknown response format: {type(response)}")
                except Exception as e:
                    raise RuntimeError(
                        f"Embedding failed: {e}. Please check your embedding model setup (sentence-transformers or litellm) and ensure all dependencies are installed."
                    )
            return embed_text
        except ImportError:
            raise ImportError("Either sentence-transformers or litellm is required for embeddings")


def setup_chromadb_store():
    """Set up ChromaDB vector store with sample data."""
    print("🗄️ Setting up ChromaDB vector store...")

    try:
        from chromadb.utils import embedding_functions
    except ImportError:
        raise ImportError("ChromaDB is required. Install with: pip install chromadb")

    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"   📁 ChromaDB directory: {temp_dir}")

    # Initialize ChromaDB
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    chroma_vector_store = lazy_import_vector_store("chromadb")
    vector_store = chroma_vector_store.create_local(
        persist_directory=temp_dir,
        collection_name="ai_ml_knowledge",
        embedding_function=embedding_function
    )

    documents = get_sample_documents()
    vector_store.collection.add(
        documents=[doc["content"] for doc in documents],
        metadatas=[doc["metadata"] for doc in documents],
        ids=[doc["metadata"]["doc_id"] for doc in documents],
    )

    print(f"   ✅ Created ChromaDB knowledge base with {len(documents)} articles")
    return vector_store


def setup_lancedb_store():
    """Set up LanceDB vector store with sample data."""
    print("🗄️ Setting up LanceDB vector store...")

    try:
        embedding_function = create_embedding_function()
        lancedb_vector_store = lazy_import_vector_store("lancedb")

        vector_store = lancedb_vector_store.create_local(
            table_name="rag_demo",
            embedding_function=embedding_function,
            db_path="./lancedb_demo",
            vector_size=384,
        )

        documents = get_sample_documents_simple()
        embeddings = [embedding_function(doc["content"]) for doc in documents]
        ids = vector_store.add_documents(documents, embeddings)

        print(f"   ✅ Added {len(ids)} documents to LanceDB table")
        return vector_store

    except ImportError as e:
        raise ImportError(f"LanceDB dependencies missing: {e}\nInstall with: pip install lancedb pyarrow sentence-transformers")


def setup_milvus_store():
    """Set up Milvus vector store with sample data."""
    print("🗄️ Setting up Milvus vector store...")

    try:
        embedding_function = create_embedding_function()
        milvus_vector_store = lazy_import_vector_store("milvus")

        vector_store = milvus_vector_store.create_local(
            collection_name="rag_demo",
            embedding_function=embedding_function,
            vector_size=384,
            uri="./milvus_demo.db",
        )

        documents = get_sample_documents_simple()
        embeddings = [embedding_function(doc["content"]) for doc in documents]
        ids = vector_store.add_documents(documents, embeddings)

        print(f"   ✅ Added {len(ids)} documents to Milvus collection")
        return vector_store

    except ImportError as e:
        raise ImportError(f"Milvus dependencies missing: {e}\nInstall with: pip install pymilvus sentence-transformers")


def setup_qdrant_store():
    """Set up Qdrant vector store with sample data."""
    print("🗄️ Setting up Qdrant vector store...")

    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError:
        raise ImportError("Qdrant client required. Install with: pip install qdrant-client")

    # Connect to in-memory Qdrant
    client = QdrantClient(path=":memory:")
    print("   ✅ Connected to in-memory Qdrant")

    collection_name = "AIKnowledge"

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    # Create embedding function and determine vector size
    embedding_fn = create_embedding_function()
    sample_vector = embedding_fn("test")
    vector_size = len(sample_vector)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    # Add documents
    documents = get_sample_documents_for_qdrant()

    points = []
    for i, doc in enumerate(documents):
        doc_vector = embedding_fn(doc["content"])
        payload = dict(doc)
        payload["original_id"] = f"doc_{i + 1}"

        point = models.PointStruct(
            id=i + 1,
            vector=doc_vector,
            payload=payload,
        )
        points.append(point)

    client.upsert(collection_name=collection_name, points=points, wait=True)
    print(f"   ✅ Created Qdrant knowledge base with {len(documents)} articles")

    qdrant_vector_store = lazy_import_vector_store("qdrant")
    vector_store = qdrant_vector_store(client, collection_name, embedding_fn)
    return vector_store


def setup_weaviate_store():
    """Set up Weaviate vector store with sample data."""
    print("🗄️ Setting up Weaviate vector store...")

    try:
        import weaviate
        import weaviate.classes as wvc
    except ImportError:
        raise ImportError("Weaviate client required. Install with: pip install weaviate-client")

    # Connect to local Weaviate
    try:
        client = weaviate.connect_to_local()
        print("   ✅ Connected to local Weaviate")
    except Exception as e:
        print(f"   ❌ Failed to connect to Weaviate: {e}")
        print("   💡 Make sure Weaviate is running:")
        print("      docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.1")
        raise

    collection_name = "AIKnowledge"

    # Delete existing collection if it exists
    try:
        client.collections.delete(collection_name)
        print(f"   🗑️ Removed existing collection: {collection_name}")
    except Exception:
        pass

    # Create collection
    collection = client.collections.create(
        name=collection_name,
        properties=[
            wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT, description="Document content"),
            wvc.config.Property(name="topic", data_type=wvc.config.DataType.TEXT, description="Topic category"),
            wvc.config.Property(name="difficulty", data_type=wvc.config.DataType.TEXT, description="Difficulty level"),
        ],
        vectorizer_config=wvc.config.Configure.Vectorizer.none(),
        inverted_index_config=wvc.config.Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2,
        ),
    )

    # Create embedding function and add documents
    embedding_fn = create_embedding_function()
    documents = get_sample_documents_for_weaviate()

    with collection.batch.dynamic() as batch:
        for doc in documents:
            doc_vector = embedding_fn(doc["content"])
            batch.add_object(properties=doc, vector=doc_vector)

    client.close()
    print(f"   ✅ Created Weaviate knowledge base with {len(documents)} articles")

    # Reconnect and create vector store wrapper
    client_for_store = weaviate.connect_to_local()
    weaviate_vector_store = lazy_import_vector_store("weaviate")
    vector_store = weaviate_vector_store(client_for_store, collection_name, embedding_fn)
    return vector_store


def get_sample_documents() -> list[dict[str, Any]]:
    """Get sample documents for ChromaDB (with nested metadata structure)."""
    return [
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


def get_sample_documents_simple() -> list[dict[str, str]]:
    """Get sample documents for LanceDB/Milvus (flat structure)."""
    return [
        {"content": "Machine learning is a method of data analysis that automates analytical model building."},
        {"content": "It is a branch of artificial intelligence based on the idea that systems can learn from data."},
        {"content": "Machine learning algorithms build a model based on training data to make predictions."},
        {"content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks."},
        {"content": "It uses multiple layers to progressively extract higher-level features from raw input."},
        {"content": "Deep learning models can automatically learn representations of data with multiple levels of abstraction."},
        {"content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence."},
        {"content": "It deals with the interaction between computers and human language."},
        {"content": "NLP techniques enable computers to process and analyze large amounts of natural language data."},
    ]


def get_sample_documents_for_qdrant() -> list[dict[str, Any]]:
    """Get sample documents for Qdrant (with flat metadata)."""
    return [
        {
            "content": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
            "topic": "artificial_intelligence",
            "difficulty": "beginner",
            "category": "definition",
        },
        {
            "content": "Machine Learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "topic": "machine_learning",
            "difficulty": "beginner",
            "category": "definition",
        },
        {
            "content": "Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks have been applied to computer vision, speech recognition, and natural language processing.",
            "topic": "deep_learning",
            "difficulty": "intermediate",
            "category": "technical",
        },
        {
            "content": "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. The goal is to program computers to process and analyze large amounts of natural language data.",
            "topic": "nlp",
            "difficulty": "intermediate",
            "category": "technical",
        },
        {
            "content": "Computer Vision is a field of artificial intelligence (AI) that enables computers and systems to derive meaningful information from digital images, videos and other visual inputs. It uses machine learning models to analyze and interpret visual data.",
            "topic": "computer_vision",
            "difficulty": "intermediate",
            "category": "application",
        },
        {
            "content": "Transformers are a deep learning architecture that has revolutionized natural language processing. They rely entirely on self-attention mechanisms to draw global dependencies between input and output, dispensing with recurrence and convolutions entirely.",
            "topic": "transformers",
            "difficulty": "advanced",
            "category": "architecture",
        },
    ]


def get_sample_documents_for_weaviate() -> list[dict[str, str]]:
    """Get sample documents for Weaviate (flat string properties)."""
    return [
        {
            "content": "Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
            "topic": "artificial_intelligence",
            "difficulty": "beginner",
        },
        {
            "content": "Machine Learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
            "topic": "machine_learning",
            "difficulty": "beginner",
        },
        {
            "content": "Deep Learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks have been applied to computer vision, speech recognition, and natural language processing.",
            "topic": "deep_learning",
            "difficulty": "intermediate",
        },
        {
            "content": "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. The goal is to program computers to process and analyze large amounts of natural language data.",
            "topic": "nlp",
            "difficulty": "intermediate",
        },
        {
            "content": "Computer Vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos. From an engineering perspective, it seeks to understand and automate tasks that the human visual system can do.",
            "topic": "computer_vision",
            "difficulty": "intermediate",
        },
        {
            "content": "Transformers are a deep learning architecture that has revolutionized natural language processing. They rely entirely on self-attention mechanisms to draw global dependencies between input and output, dispensing with recurrence and convolutions entirely.",
            "topic": "transformers",
            "difficulty": "advanced",
        },
    ]


def create_training_data() -> tuple[list[RAGDataInst], list[RAGDataInst]]:
    """Create training and validation datasets for RAG optimization."""
    # Training examples
    train_data = [
        RAGDataInst(
            query="What is machine learning?",
            ground_truth_answer="Machine Learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
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
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL)
    cleaned = cleaned.strip()

    # If still empty or starts with <think> without closing tag, try to find content after
    if not cleaned or cleaned.startswith("<think>"):
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


def setup_vector_store(store_name: str):
    """Factory function to set up the specified vector store."""
    setup_functions = {
        "chromadb": setup_chromadb_store,
        "lancedb": setup_lancedb_store,
        "milvus": setup_milvus_store,
        "qdrant": setup_qdrant_store,
        "weaviate": setup_weaviate_store,
    }

    if store_name not in setup_functions:
        raise ValueError(f"Unknown vector store: {store_name}. Supported: {list(setup_functions.keys())}")

    return setup_functions[store_name]()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GEPA RAG Optimization Example with Multiple Vector Stores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rag_optimization.py --vector-store chromadb
  python rag_optimization.py --vector-store lancedb --model ollama/llama3.1:8b
  python rag_optimization.py --vector-store qdrant --max-iterations 10
  python rag_optimization.py --vector-store weaviate --model gpt-4o-mini

Supported Vector Stores:
  chromadb  - Local/persistent, simple setup (default)
  lancedb   - Serverless, no Docker required
  milvus    - Cloud-native, uses Lite mode locally
  qdrant    - High-performance, advanced filtering
  weaviate  - Hybrid search capabilities (requires Docker)
        """,
    )

    parser.add_argument(
        "--vector-store",
        type=str,
        default="chromadb",
        choices=["chromadb", "lancedb", "milvus", "qdrant", "weaviate"],
        help="Vector store to use (default: chromadb)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/qwen3:8b",
        help="LLM model (default: ollama/qwen3:8b)"
    )
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
    """Main function demonstrating RAG optimization with multiple vector stores."""
    args = parse_arguments()

    print("🚀 GEPA RAG Optimization with Multiple Vector Stores")
    print("=" * 60)
    print(f"🗄️ Vector Store: {args.vector_store}")
    print(f"📊 Model: {args.model}")
    print(f"🔗 Embeddings: {args.embedding_model}")
    print(f"🔄 Max Iterations: {args.max_iterations}")

    try:
        # Step 1: Setup vector store
        print(f"\n1️⃣ Setting up {args.vector_store} vector store...")
        vector_store = setup_vector_store(args.vector_store)

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
        rag_config = {
            "retrieval_strategy": "similarity",
            "top_k": 3,
            "retrieval_weight": 0.3,
            "generation_weight": 0.7,
        }

        # Add hybrid search for Weaviate
        if args.vector_store == "weaviate":
            rag_config["retrieval_strategy"] = "hybrid"
            rag_config["hybrid_alpha"] = 0.7

        rag_adapter = GenericRAGAdapter(
            vector_store=vector_store,
            llm_model=llm_client,
            embedding_model=args.embedding_model,
            rag_config=rag_config,
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

        print(f"\n✅ {args.vector_store.title()} RAG optimization completed successfully!")

        # Clean up connections
        try:
            if hasattr(vector_store, "client") and hasattr(vector_store.client, "close"):
                vector_store.client.close()
        except Exception:
            pass

    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

        print("\n🔧 Troubleshooting tips:")
        if args.vector_store == "weaviate":
            print("  • Ensure Weaviate is running: curl http://localhost:8080/v1/meta")
            print("  • Start Weaviate: docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.1")
        elif args.vector_store == "qdrant":
            print("  • For external Qdrant: docker run -p 6333:6333 qdrant/qdrant")

        print("  • Ensure Ollama is running: ollama list")
        print("  • Check models are available: ollama pull qwen3:8b")
        print("  • For cloud models: set API keys (OPENAI_API_KEY, etc.)")
        print(f"  • Install dependencies: pip install {get_install_command(args.vector_store)}")

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
