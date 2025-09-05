# Generic RAG Adapter for GEPA

A vector store-agnostic RAG (Retrieval-Augmented Generation) adapter that enables GEPA to optimize RAG systems across any vector store implementation.

## 🎯 Overview

The Generic RAG Adapter brings GEPA's evolutionary optimization to the world of RAG systems. With its pluggable vector store architecture, you can optimize RAG prompts once and deploy across any vector store—from local ChromaDB instances to production Weaviate clusters.

**Key Benefits:**
- **Vector Store Agnostic**: Write once, run anywhere (ChromaDB, Weaviate, Qdrant, Pinecone, Milvus)
- **Multi-Component Optimization**: Simultaneously optimize query reformulation, context synthesis, answer generation, and reranking
- **Comprehensive Evaluation**: Both retrieval quality (precision, recall, MRR) and generation quality (F1, BLEU, faithfulness) metrics
- **Production Ready**: Battle-tested with proper error handling, logging, and performance monitoring

## 🚀 Quick Start

### Installation

```bash
# Basic RAG optimization support
pip install gepa[rag]

# With specific vector stores
pip install gepa[chromadb]    # For ChromaDB
pip install gepa[weaviate]    # For Weaviate

# Setup local Ollama models for examples
# Standard models (higher quality)
ollama pull gpt-oss:20b   # Task execution model
ollama pull gpt-oss:120b  # Reflection model for optimization

# Memory-friendly alternatives (for low-resource machines)
ollama pull llama3.2:1b   # Lightweight task model (~1GB RAM)
ollama pull llama3.1:8b   # Lightweight reflection model (~5GB RAM)
```

### 5-Minute Example

```python
import gepa
from gepa.adapters.generic_rag_adapter import GenericRAGAdapter, ChromaVectorStore

# 1. Setup your vector store
vector_store = ChromaVectorStore.create_local("./knowledge_base", "documents")

# 2. Create the RAG adapter
adapter = GenericRAGAdapter(
    vector_store=vector_store,
    llm_model="ollama/llama3.2:1b",  # Memory-friendly local model
    # llm_model="gpt-4",  # For cloud-based models (requires API key)
    rag_config={
        "retrieval_strategy": "similarity",
        "top_k": 3  # Reduced for faster local testing
    }
)

# 3. Define your training data
train_data = [
    {
        "query": "What is machine learning?",
        "ground_truth_answer": "Machine learning is...",
        "relevant_doc_ids": ["doc_001"],
        "metadata": {"category": "AI"}
    }
    # ... more examples
]

# 4. Define initial prompts to optimize
initial_prompts = {
    "answer_generation": "Answer the question based on the provided context.",
    "context_synthesis": "Synthesize the following documents into a coherent context."
}

# 5. Run GEPA optimization (local-friendly settings)
result = gepa.optimize(
    seed_candidate=initial_prompts,
    trainset=train_data,
    valset=validation_data,
    adapter=adapter,
    max_metric_calls=10,  # Small number for local testing
    reflection_llm_model="ollama/llama3.1:8b"  # Memory-friendly reflection model
    # reflection_llm_model="gpt-4"  # For cloud-based optimization
)

print("Optimized RAG prompts:", result.best_candidate)
```

### Vector Store Support

```python
# ChromaDB (local development)
vector_store = ChromaVectorStore.create_local("./kb", "docs")

# Weaviate (production with hybrid search)
vector_store = WeaviateVectorStore.create_cloud(
    cluster_url="https://your-cluster.weaviate.network",
    auth_credentials=weaviate.AuthApiKey("your-key"),
    collection_name="KnowledgeBase"
)

# Same optimization pipeline works with both!
```

**Optimizable Components:**
- **Query Reformulation**: Improve user query understanding and reformulation
- **Context Synthesis**: Optimize document combination and summarization
- **Answer Generation**: Enhance final answer quality and accuracy
- **Document Reranking**: Improve retrieved document relevance ordering

## 🏗️ Architecture

### Vector Store Interface

The adapter uses a clean abstraction that any vector store can implement:

```python
class VectorStoreInterface(ABC):
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Semantic similarity search"""

    @abstractmethod
    def vector_search(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Direct vector search"""

    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Hybrid semantic + keyword search (optional)"""

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Collection metadata and statistics"""
```

### RAG Pipeline Flow

```
User Query → Query Reformulation → Document Retrieval → Document Reranking → Context Synthesis → Answer Generation → Final Answer
     ↑              ↑                     ↑                    ↑                  ↑                  ↑
  Original      Optimizable          Vector Store        Optimizable        Optimizable        Optimizable
   Input          Prompt             Operations            Prompt            Prompt             Prompt
```

## 🔧 Supported Vector Stores

### ChromaDB
Perfect for local development, prototyping, and smaller deployments.

```python
from gepa.adapters.generic_rag_adapter import ChromaVectorStore

# Local persistence
vector_store = ChromaVectorStore.create_local(
    persist_directory="./chroma_db",
    collection_name="documents"
)

# In-memory (testing)
vector_store = ChromaVectorStore.create_memory(
    collection_name="test_docs"
)
```

### Weaviate
Production-grade with advanced features like hybrid search and multi-tenancy.

```python
from gepa.adapters.generic_rag_adapter import WeaviateVectorStore

# Local Weaviate instance
vector_store = WeaviateVectorStore.create_local(
    host="localhost",
    port=8080,
    collection_name="Documents"
)

# Weaviate Cloud Services (WCS)
vector_store = WeaviateVectorStore.create_cloud(
    cluster_url="https://your-cluster.weaviate.network",
    auth_credentials=weaviate.AuthApiKey("your-api-key"),
    collection_name="Documents"
)

# Custom deployment
vector_store = WeaviateVectorStore.create_custom(
    url="https://your-weaviate-instance.com",
    auth_credentials=auth_credentials,
    collection_name="Documents"
)
```

### Adding New Vector Stores

Implement the `VectorStoreInterface` for your vector store:

```python
class MyVectorStore(VectorStoreInterface):
    def similarity_search(self, query: str, k: int = 5, filters=None):
        # Your implementation
        return documents

    def vector_search(self, query_vector: List[float], k: int = 5, filters=None):
        # Your implementation
        return documents

    def get_collection_info(self):
        return {
            "name": self.collection_name,
            "document_count": self.count(),
            "dimension": self.vector_dim,
            "vector_store_type": "my_store"
        }
```

## 🎛️ Configuration Options

### Model Configuration

**Local Ollama Models (Recommended for Testing):**
```python
# Memory-friendly options
adapter = GenericRAGAdapter(
    vector_store=vector_store,
    llm_model="ollama/llama3.2:1b",  # ~1GB RAM - Fast inference
    rag_config=config
)

# Higher quality local models
adapter = GenericRAGAdapter(
    vector_store=vector_store,
    llm_model="ollama/llama3.1:8b",  # ~5GB RAM - Better quality
    rag_config=config
)

# GEPA optimization with local models
result = gepa.optimize(
    seed_candidate=initial_prompts,
    trainset=train_data,
    valset=validation_data,
    adapter=adapter,
    max_metric_calls=5,  # Small for local testing
    reflection_llm_model="ollama/llama3.1:8b"
)
```

**Cloud Models (Production Use):**
```python
# OpenAI models
adapter = GenericRAGAdapter(
    vector_store=vector_store,
    llm_model="gpt-4o",  # Requires OPENAI_API_KEY
    rag_config=config
)

# Anthropic models
adapter = GenericRAGAdapter(
    vector_store=vector_store,
    llm_model="claude-3-5-sonnet-20241022",  # Requires ANTHROPIC_API_KEY
    rag_config=config
)

# GEPA optimization with cloud models
result = gepa.optimize(
    seed_candidate=initial_prompts,
    trainset=train_data,
    valset=validation_data,
    adapter=adapter,
    max_metric_calls=50,  # Higher for production optimization
    reflection_llm_model="gpt-4o"
)
```

### RAG Pipeline Configuration

```python
rag_config = {
    # Retrieval Strategy
    "retrieval_strategy": "similarity",    # "similarity", "hybrid", "vector"
    "top_k": 5,                          # Documents to retrieve

    # Evaluation Weights
    "retrieval_weight": 0.3,             # Weight for retrieval metrics
    "generation_weight": 0.7,            # Weight for generation metrics

    # Hybrid Search (Weaviate)
    "hybrid_alpha": 0.5,                 # 0.0=keyword, 1.0=semantic, 0.5=balanced

    # Filtering
    "filters": {"category": "technical"}  # Metadata filters
}

adapter = GenericRAGAdapter(
    vector_store=vector_store,
    llm_model="ollama/llama3.2:1b",  # Local model for testing
    # llm_model="gpt-4o",  # For cloud-based usage
    rag_config=rag_config
)
```

### Advanced Configuration Examples

**ChromaDB Optimized:**
```python
config = {
    "retrieval_strategy": "similarity",
    "top_k": 7,
    "retrieval_weight": 0.35,
    "generation_weight": 0.65
}
```

**Weaviate with Hybrid Search:**
```python
config = {
    "retrieval_strategy": "hybrid",
    "top_k": 5,
    "hybrid_alpha": 0.7,               # More semantic than keyword
    "retrieval_weight": 0.25,
    "generation_weight": 0.75,
    "filters": {"confidence": {"$gt": 0.8}}
}
```

## 🔍 Optimizable Components

### 1. Query Reformulation
Transform user queries for better retrieval:

```python
"query_reformulation": """
You are an expert at reformulating user queries for information retrieval.
Your task is to enhance the query while preserving the original intent.

Guidelines:
- Add relevant technical terms and synonyms
- Make the query more specific and focused
- Optimize for both semantic and keyword matching
- Preserve key concepts from the original query

Reformulate the following query for better retrieval:
"""
```

### 2. Context Synthesis
Combine retrieved documents into coherent context:

```python
"context_synthesis": """
You are an expert at synthesizing information from multiple documents.
Your task is to create a comprehensive context that directly addresses the query.

Guidelines:
- Focus on information most relevant to the user's question
- Integrate information from multiple sources seamlessly
- Remove redundant or conflicting information
- Maintain factual accuracy and important details

Query: {query}

Synthesize the following retrieved documents:
"""
```

### 3. Answer Generation
Generate accurate, well-structured final answers:

```python
"answer_generation": """
You are an AI assistant providing expert-level answers.
Your task is to generate accurate, comprehensive responses based on the provided context.

Guidelines:
- Base your answer primarily on the provided context
- Structure your response with clear explanations
- Include specific details and examples when available
- If context is insufficient, acknowledge the limitation clearly

Context: {context}
Question: {query}

Provide a thorough, accurate answer:
"""
```

### 4. Document Reranking
Optimize retrieved document relevance ordering:

```python
"reranking_criteria": """
You are an expert at evaluating document relevance for question answering.
Your task is to rank documents by their relevance to the specific query.

Ranking Criteria:
- Documents with direct answers get highest priority
- Comprehensive explanations rank second
- Supporting examples and context rank third
- Off-topic or tangential content ranks lowest

Query: {query}

Rank these documents by relevance (most relevant first):
"""
```

## 📊 Evaluation Metrics

### Retrieval Quality Metrics

- **Precision**: Fraction of retrieved documents that are relevant
- **Recall**: Fraction of relevant documents that were retrieved
- **F1 Score**: Harmonic mean of precision and recall
- **MRR (Mean Reciprocal Rank)**: Quality of ranking for relevant documents

### Generation Quality Metrics

- **Exact Match**: Whether generated answer exactly matches ground truth
- **Token F1**: F1 score based on token overlap with ground truth
- **BLEU Score**: N-gram overlap similarity measure
- **Answer Relevance**: How well the answer relates to retrieved context
- **Faithfulness**: How well the answer is supported by the context

### Combined Scoring

The adapter computes a weighted combination of retrieval and generation metrics:

```
final_score = (retrieval_weight × retrieval_f1) + (generation_weight × generation_score)
```

Where `generation_score` combines token F1, answer relevance, and faithfulness.

## 🚀 Production Examples

### Multi-Vector Store Deployment

```python
def create_rag_adapter(env: str):
    if env == "development":
        vector_store = ChromaVectorStore.create_local("./local_kb", "docs")
        config = {"retrieval_strategy": "similarity", "top_k": 3}
        llm_model = "ollama/llama3.2:1b"  # Memory-friendly for local dev
    elif env == "production":
        vector_store = WeaviateVectorStore.create_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=weaviate.AuthApiKey(os.getenv("WEAVIATE_KEY")),
            collection_name="ProductionKB"
        )
        config = {
            "retrieval_strategy": "hybrid",
            "hybrid_alpha": 0.75,
            "top_k": 5,
            "filters": {"status": "approved"}
        }
        llm_model = os.getenv("LLM_MODEL", "gpt-4o")  # Cloud models for production

    return GenericRAGAdapter(
        vector_store=vector_store,
        llm_model=llm_model,
        rag_config=config
    )
```

### Performance Monitoring

```python
# Enable detailed tracing for analysis
eval_batch = adapter.evaluate(
    batch=test_data,
    candidate=optimized_prompts,
    capture_traces=True
)

# Analyze performance
for i, (trajectory, score) in enumerate(zip(eval_batch.trajectories, eval_batch.scores)):
    print(f"Query {i+1}: Score = {score:.3f}")
    print(f"  Retrieved: {len(trajectory['retrieved_docs'])} documents")
    print(f"  Token usage: {trajectory['execution_metadata']['total_tokens']}")

    # Access detailed metrics
    retrieval_metrics = trajectory['execution_metadata']['retrieval_metrics']
    generation_metrics = trajectory['execution_metadata']['generation_metrics']

    print(f"  Retrieval F1: {retrieval_metrics['retrieval_f1']:.3f}")
    print(f"  Generation F1: {generation_metrics['token_f1']:.3f}")
    print(f"  Faithfulness: {generation_metrics['faithfulness']:.3f}")
```

## 📚 Complete Examples

### Basic Usage
- **[Basic RAG Optimization](examples/rag_optimization_example.py)** - Start here with ChromaDB
- **[Weaviate Integration](examples/weaviate_rag_optimization.py)** - Production deployment example
- **[Multi-Store Comparison](examples/multi_vector_store_comparison.py)** - Compare vector stores

### Advanced Patterns
- **[Production Deployment](examples/production_rag_setup.py)** - Scalable production configuration
- **[Custom Vector Store](examples/custom_vector_store_example.py)** - Implement your own vector store
- **[Evaluation Analysis](examples/rag_evaluation_analysis.py)** - Deep-dive into metrics and performance

## 🤝 Contributing

We welcome contributions! Priority areas:

### New Vector Store Implementations
- **Qdrant**: Cloud-native vector database with advanced filtering
- **Pinecone**: Managed vector database with high performance
- **Milvus**: Open-source vector database for production scale
- **Elasticsearch**: Search engine with vector capabilities
- **OpenSearch**: Open-source alternative to Elasticsearch

### Enhancement Areas
- Advanced reranking algorithms (learning-to-rank, neural rerankers)
- Multi-modal RAG support (text + images)
- Streaming evaluation for large datasets
- Integration with embedding providers (OpenAI, Cohere, HuggingFace)
- Performance optimizations and caching strategies

### Implementation Guidelines

1. **Follow the Interface**: Implement `VectorStoreInterface` completely
2. **Add Factory Methods**: Provide `create_local()`, `create_cloud()` class methods
3. **Error Handling**: Graceful degradation and clear error messages
4. **Documentation**: Comprehensive docstrings and usage examples
5. **Testing**: Unit tests for all public methods

See our [contribution guide](CONTRIBUTING.md) for detailed instructions.

## 📄 API Reference

### Core Classes

- **[`GenericRAGAdapter`](generic_rag_adapter.py)**: Main adapter class for GEPA integration
- **[`VectorStoreInterface`](vector_store_interface.py)**: Abstract base class for vector stores
- **[`RAGPipeline`](rag_pipeline.py)**: RAG execution engine
- **[`RAGEvaluationMetrics`](evaluation_metrics.py)**: Comprehensive evaluation metrics

### Vector Store Implementations

- **[`ChromaVectorStore`](vector_stores/chroma_store.py)**: ChromaDB implementation
- **[`WeaviateVectorStore`](vector_stores/weaviate_store.py)**: Weaviate implementation

### Type Definitions

- **`RAGDataInst`**: Training/validation example structure
- **`RAGTrajectory`**: Detailed execution trace
- **`RAGOutput`**: Final system output with metadata

## 🔒 License

Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
Licensed under the MIT License - see [LICENSE](../../../LICENSE) for details.
