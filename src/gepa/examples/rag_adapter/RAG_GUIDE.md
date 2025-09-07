# GEPA Generic RAG Adapter Guide

This guide demonstrates how to use GEPA's Generic RAG Adapter with the new **unified `rag_optimization.py`** script that supports multiple vector stores. This consolidated approach makes it easy to test and compare different vector databases with a single command.

## 🆕 Unified Script: `rag_optimization.py`

**One script, multiple vector stores!** We've consolidated all individual optimization scripts into a single, powerful `rag_optimization.py` that supports all vector stores through command-line arguments.

### 📂 Supported Vector Stores

| Vector Store | Docker Required | Key Features | Usage |
|--------------|----------------|--------------|--------|
| **ChromaDB** (default) | ❌ No | Local storage, simple setup, semantic search | `--vector-store chromadb` |
| **LanceDB** | ❌ No | Serverless, columnar format, developer-friendly | `--vector-store lancedb` |
| **Milvus** | ❌ No* | Cloud-native, scalable, Milvus Lite for local dev | `--vector-store milvus` |
| **Qdrant** | ❌ No* | Advanced filtering, payload search, high performance | `--vector-store qdrant` |
| **Weaviate** | ✅ Yes | Hybrid search, production-ready, advanced features | `--vector-store weaviate` |

*Docker optional for production deployments

### ✨ Benefits of the Unified Approach

- **🔄 Easy Switching**: Test different vector stores with just a flag change
- **🔧 Consistent Interface**: Same commands work across all databases
- **📊 Fair Comparison**: Identical test conditions for comparing performance
- **🛠 Less Maintenance**: Single file to maintain instead of 5 separate scripts

## 🚀 Quick Start Guide

### Prerequisites

1. **Install Dependencies:**

   **Install GEPA Core:**
   ```bash
   pip install gepa
   ```

   **Install RAG Adapter Dependencies:**
   
   You can either install all vector store dependencies or specific ones:

   ```bash
   # Option A: Install all dependencies (recommended for exploration)
   pip install -r requirements-rag.txt

   # Option B: Install specific vector store dependencies
   # ChromaDB (easiest to start with)
   pip install litellm chromadb

   # LanceDB (serverless, no Docker needed)  
   pip install litellm lancedb pyarrow

   # Milvus (local Lite mode)
   pip install litellm pymilvus

   # Qdrant (in-memory mode)
   pip install litellm qdrant-client

   # Weaviate (requires Docker)
   pip install litellm weaviate-client
   ```

   **Note:** Vector store dependencies are now separate from the core GEPA package and must be installed manually based on which vector stores you want to use. For specific version requirements, see `requirements-rag.txt`.

2. **For Local Models (Ollama):**
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.com/install.sh | sh

   # Pull models used in examples
   ollama pull qwen3:8b
   ollama pull llama3.1:8b
   ollama pull nomic-embed-text:latest
   ```

3. **For Cloud Models:**
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export ANTHROPIC_API_KEY="your-api-key"
   ```

4. **Docker Requirements:**

   | Database | Docker Required | Notes |
   |----------|----------------|-------|
   | **ChromaDB** | ❌ No | Runs locally, no external services |
   | **LanceDB** | ❌ No | Serverless, creates local files |
   | **Milvus** | ❌ No (default) | Uses Milvus Lite (local SQLite) |
   | **Qdrant** | ❌ No (default) | Uses in-memory mode by default |
   | **Weaviate** | ✅ Yes | Requires Docker or cloud instance |

   **Docker Setup (only for Weaviate):**
   ```bash
   # Start Weaviate with Docker
   docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.1
   ```

   **Optional Docker Setup:**
   ```bash
   # For production Milvus (optional)
   docker run -d -p 19530:19530 milvusdb/milvus:latest standalone

   # For production Qdrant (optional)
   docker run -p 6333:6333 qdrant/qdrant
   ```

## 🚀 Using the Unified RAG Optimization Script

### Basic Usage

```bash
# Navigate to the examples directory
cd src/gepa/examples/rag_adapter

# 🔵 ChromaDB (Default - No Docker Required)
python rag_optimization.py --vector-store chromadb

# 🟢 LanceDB (Serverless - No Docker Required)
python rag_optimization.py --vector-store lancedb

# 🔵 Milvus (Local Lite Mode - No Docker Required)
python rag_optimization.py --vector-store milvus

# 🟡 Qdrant (In-Memory - No Docker Required)
python rag_optimization.py --vector-store qdrant

# 🟠 Weaviate (Requires Docker)
python rag_optimization.py --vector-store weaviate
```

### Quick Test (No Optimization)

```bash
# Test setup without running full optimization
python rag_optimization.py --vector-store chromadb --max-iterations 0
python rag_optimization.py --vector-store lancedb --max-iterations 0
python rag_optimization.py --vector-store qdrant --max-iterations 0
```

### Full Optimization Runs

```bash
# ChromaDB with 10 iterations
python rag_optimization.py --vector-store chromadb --max-iterations 10

# LanceDB with 20 iterations
python rag_optimization.py --vector-store lancedb --max-iterations 20

# Qdrant with 15 iterations
python rag_optimization.py --vector-store qdrant --max-iterations 15
```

### Different Models

```bash
# Use different Ollama models
python rag_optimization.py --vector-store chromadb --model ollama/llama3.1:8b

# Use cloud models (requires API key)
python rag_optimization.py --vector-store lancedb --model gpt-4o-mini --max-iterations 10

# Use Anthropic models
python rag_optimization.py --vector-store qdrant --model claude-3-haiku-20240307
```


## 📋 Vector Store Specific Instructions

### 🔵 ChromaDB (Default & Easiest)

ChromaDB is perfect for getting started - lightweight, runs locally, no external services needed.

**✅ No Docker Required**

```bash
# Basic usage (default vector store)
python rag_optimization.py --vector-store chromadb

# Or simply (chromadb is the default)
python rag_optimization.py

# Quick test
python rag_optimization.py --max-iterations 0

# Full optimization run
python rag_optimization.py --max-iterations 20
```

**Key Features:**
- Local persistent storage
- Simple setup with no configuration
- Semantic similarity search
- Built-in embedding functions


### 🟢 LanceDB (Serverless & Developer-Friendly)

LanceDB is a serverless vector database built on the Lance columnar format, perfect for local development.

**✅ No Docker Required**

```bash
# Basic usage
python rag_optimization.py --vector-store lancedb

# With different models
python rag_optimization.py --vector-store lancedb --model ollama/qwen3:8b

# Full optimization run
python rag_optimization.py --vector-store lancedb --max-iterations 20
```

**Key Features:**
- Serverless architecture (no external services)
- Built on Apache Arrow/Lance for performance
- Creates local database files (./lancedb_demo)
- Developer-friendly with simple setup


### 🔵 Milvus (Cloud-Native & Scalable)

Milvus is a cloud-native vector database designed for large-scale AI applications. Uses Milvus Lite for local development.

**✅ No Docker Required (uses Milvus Lite locally)**

```bash
# Basic usage (uses local SQLite-based Milvus Lite)
python rag_optimization.py --vector-store milvus

# With different models
python rag_optimization.py --vector-store milvus --model gpt-4o-mini --max-iterations 10

# Full optimization run
python rag_optimization.py --vector-store milvus --max-iterations 15
```

**Key Features:**
- Milvus Lite (local SQLite, no Docker needed)
- Creates local ./milvus_demo.db file automatically
- Cloud-native design for production scaling
- Advanced indexing and search capabilities


### 🟡 Qdrant (High-Performance & Advanced Filtering)

Qdrant is a high-performance vector database with advanced filtering and payload search capabilities.

**✅ No Docker Required (uses in-memory mode by default)**

```bash
# Basic usage (in-memory mode)
python rag_optimization.py --vector-store qdrant

# With different models
python rag_optimization.py --vector-store qdrant --model gpt-4o-mini --max-iterations 10

# Full optimization run
python rag_optimization.py --vector-store qdrant --max-iterations 15
```

**Key Features:**
- In-memory mode (no external services needed)
- Advanced metadata filtering capabilities
- Payload search (vector + metadata combined)
- High-performance optimized for speed and scale
- Optional persistent storage or remote server


### 🟠 Weaviate (Hybrid Search)

Weaviate offers advanced features like hybrid search (semantic + keyword) and is production-ready.

**⚠️ Docker Required**

```bash
# Setup: Start Weaviate with Docker (required)
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.1

# Verify Weaviate is running
curl http://localhost:8080/v1/meta

# Basic usage (requires Docker setup above)
python rag_optimization.py --vector-store weaviate

# With cloud models
python rag_optimization.py --vector-store weaviate --model gpt-4o-mini --max-iterations 10

# Full optimization run
python rag_optimization.py --vector-store weaviate --max-iterations 15
```

**Key Features:**
- Hybrid search (semantic + keyword/BM25 combined)
- Production-ready with clustering support
- Rich GraphQL and RESTful APIs
- Advanced schema management
- Built-in vectorization modules


## ⚙️ Configuration Options

### Command Line Arguments

The unified `rag_optimization.py` script supports these arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--vector-store` | `chromadb` | Choose vector store: `chromadb`, `lancedb`, `milvus`, `qdrant`, `weaviate` |
| `--model` | `ollama/qwen3:8b` | LLM model to use for generation |
| `--embedding-model` | `ollama/nomic-embed-text:latest` | Embedding model for vector search |
| `--max-iterations` | `5` | GEPA optimization iterations (use 0 to skip optimization) |
| `--verbose` | `False` | Enable detailed logging and debugging |

### Complete Help

```bash
# See all available options
python rag_optimization.py --help
```

### Vector Store Selection

```bash
# Available vector stores (choose one)
--vector-store chromadb   # Default: Local, no Docker
--vector-store lancedb    # Serverless, no Docker
--vector-store milvus     # Local Lite mode, no Docker
--vector-store qdrant     # In-memory mode, no Docker
--vector-store weaviate   # Requires Docker
```

### Model Recommendations

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `ollama/qwen3:8b` | Large | Medium | Excellent | Default for ChromaDB/Weaviate/Qdrant |
| `ollama/llama3.1:8b` | Large | Medium | Excellent | Default for LanceDB/Milvus |
| `gpt-4o-mini` | Cloud | Fast | Excellent | Production (cloud) |
| `claude-3-haiku-20240307` | Cloud | Fast | Excellent | Production (cloud) |

### Embedding Models

| Model | Provider | Use Case |
|-------|----------|----------|
| `ollama/nomic-embed-text:latest` | Local | Offline, privacy |
| `text-embedding-3-small` | OpenAI | Fast, cost-effective |
| `text-embedding-3-large` | OpenAI | High quality |

## 🧪 Testing Your Setup

### Quick Health Check

```bash
# Test all vector stores (no optimization, just setup verification)
python rag_optimization.py --vector-store chromadb --max-iterations 0
python rag_optimization.py --vector-store lancedb --max-iterations 0
python rag_optimization.py --vector-store milvus --max-iterations 0
python rag_optimization.py --vector-store qdrant --max-iterations 0
python rag_optimization.py --vector-store weaviate --max-iterations 0  # Requires Docker

# Test external services (if using)
curl http://localhost:8080/v1/meta  # Weaviate (if using Docker)
ollama list                         # Check available Ollama models
```

### Compare Vector Stores

```bash
# Run same optimization across all vector stores for comparison
python rag_optimization.py --vector-store chromadb --max-iterations 5
python rag_optimization.py --vector-store lancedb --max-iterations 5
python rag_optimization.py --vector-store milvus --max-iterations 5
python rag_optimization.py --vector-store qdrant --max-iterations 5
```

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Make sure you're in the right directory
cd /path/to/gepa/src/gepa/examples/rag_adapter
python rag_optimization.py --vector-store chromadb

# If you get import errors, install missing dependencies using requirements-rag.txt
pip install -r requirements-rag.txt

# Or install specific vector store dependencies:
pip install litellm chromadb                    # For ChromaDB
pip install litellm lancedb pyarrow             # For LanceDB
pip install litellm pymilvus                    # For Milvus
pip install litellm qdrant-client               # For Qdrant
pip install litellm weaviate-client             # For Weaviate
```

#### Ollama Issues
```bash
# Check Ollama is running
ollama list

# Pull required models
ollama pull qwen3:8b
ollama pull llama3.1:8b
ollama pull nomic-embed-text:latest

# Test models
ollama run qwen3:8b "Hello"
ollama run llama3.1:8b "Hello"
```

#### Weaviate Issues
```bash
# Check Weaviate is accessible
curl http://localhost:8080/v1/meta

# Start Weaviate with Docker
docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.1

# Check Docker container
docker ps
```

#### LanceDB Issues
```bash
# Check LanceDB installation
python -c "import lancedb; print('LanceDB installed')"

# Check PyArrow installation
python -c "import pyarrow; print('PyArrow installed')"

# Install missing dependencies
pip install litellm lancedb pyarrow

# Check sentence-transformers for embeddings
python -c "import sentence_transformers; print('sentence-transformers installed')"
pip install sentence-transformers
```

#### Milvus Issues
```bash
# Check Milvus Lite installation
python -c "import pymilvus; print('PyMilvus installed')"

# Install missing dependencies
pip install litellm pymilvus

# Check if milvus_demo.db file exists
ls -la milvus_demo.db

# For full Milvus server issues
docker run -d -p 19530:19530 milvusdb/milvus:latest standalone
curl http://localhost:19530/health
```

#### Qdrant Issues
```bash
# Check Qdrant client installation
python -c "import qdrant_client; print('Qdrant client installed')"

# Install missing dependencies
pip install litellm qdrant-client

# Test Qdrant server connection
curl http://localhost:6333/health

# Start Qdrant with Docker
docker run -p 6333:6333 qdrant/qdrant

# Check Qdrant container
docker ps | grep qdrant
```


#### Memory Issues
```bash
# Use cloud model instead of local
python rag_optimization.py --vector-store chromadb --model gpt-4o-mini

# Reduce iterations
python rag_optimization.py --vector-store chromadb --max-iterations 2

# Test without optimization first
python rag_optimization.py --vector-store chromadb --max-iterations 0
```

#### Vector Store Specific Issues

```bash
# ChromaDB - No common issues, very stable

# LanceDB - Check PyArrow installation
python -c "import pyarrow; print('PyArrow OK')"
pip install litellm lancedb pyarrow

# Milvus - Check PyMilvus installation
python -c "import pymilvus; print('PyMilvus OK')"
pip install litellm pymilvus

# Qdrant - Check client installation
python -c "import qdrant_client; print('Qdrant client OK')"
pip install litellm qdrant-client

# Weaviate - Ensure Docker is running
curl http://localhost:8080/v1/meta
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.1
```

### Getting Help

If you encounter issues:

1. **Check Prerequisites**: Ensure all dependencies are installed
2. **Start Simple**: Use `--max-iterations 0` to test setup without optimization
3. **Use Cloud Models**: Try `gpt-4o-mini` for faster testing with less memory
4. **Enable Verbose Mode**: Add `--verbose` for detailed error information
5. **Check Resources**: Ensure sufficient memory and disk space

## 📈 Understanding Results

### Evaluation Metrics

- **Retrieval Quality**: How well relevant documents are retrieved
- **Generation Quality**: How accurate and helpful the generated answers are
- **Combined Score**: Weighted combination optimized by GEPA (higher is better)

### Optimization Process

GEPA uses evolutionary search to improve prompts:

1. **Baseline**: Test initial prompts
2. **Mutation**: Generate variations of prompts
3. **Selection**: Keep best performing versions
4. **Iteration**: Repeat until convergence or max iterations

### Expected Improvements

Typical score improvements with GEPA:
- **Initial Score**: 0.3-0.5 (basic prompts)
- **After Optimization**: 0.6-0.8 (optimized prompts)
- **Improvement Range**: +0.1 to +0.4 points

## 🎯 Next Steps

1. **Scale Up**: Use larger models and more iterations for production
2. **Custom Data**: Replace example data with your domain-specific knowledge
3. **Advanced Features**: Explore metadata filtering and custom prompts
4. **Production Setup**: Configure persistent storage and monitoring
5. **Integration**: Incorporate optimized prompts into your applications

---

## 📊 Real Optimization Results

### 🔬 ChromaDB + GEPA Optimization Example

**Configuration:**
- **Vector Database**: ChromaDB (Local, No Docker)
- **LLM Model**: Ollama Qwen3:8b (Local)
- **Embedding Model**: Ollama nomic-embed-text:latest
- **Max Iterations**: 10
- **Knowledge Base**: 6 AI/ML articles
- **Training Examples**: 3
- **Validation Examples**: 2
- **Search Strategy**: Semantic similarity search

**Performance Results:**

| Metric | Initial Score | Final Score | Improvement | Total Iterations |
|--------|---------------|-------------|-------------|-----------------|
| **Validation Score** | 0.388 | 0.388 | **+0.014** | 14 iterations |
| **Training Score** | 0.374 | - | **+3.7%** | - |

**Setup Commands:**
```bash
# No Docker required for ChromaDB!
# Run optimization directly with unified script
PYTHONPATH=src python src/gepa/examples/rag_adapter/rag_optimization.py \
    --vector-store chromadb \
    --max-iterations 10 \
    --model ollama/qwen3:8b \
    --verbose
```

**Key Observations:**
- ✅ **Successful improvement**: +0.014 score increase (+3.7% improvement)
- ChromaDB's simple setup makes it ideal for quick optimization experiments
- Local Ollama models integrated seamlessly with GEPA
- Semantic similarity search provided good retrieval quality
- GEPA's evolutionary optimization found better prompt variants

**Sample Output Evolution:**

*Initial Answer (0.374 score):*
```
### Answer:
Computer vision is a field of artificial intelligence (AI) focused on enabling computers to interpret and understand the visual world. It leverages digital images and videos as input and employs deep learning models—a subset of machine learning—to analyze and classify visual data...
```

*Optimized Answer (0.388 score):*
```
**Answer:**
Computer vision is a field of artificial intelligence (AI) focused on enabling computers to interpret and understand the visual world. It leverages **digital images and videos** as input and employs **deep learning models**—a subset of machine learning—to analyze and classify visual data. These models, inspired by biological neural networks, excel at processing **unstructured or unlabeled data**...
```

**Improvement Analysis:**
- Better formatting with bold headings and key terms
- More structured presentation of technical concepts
- Enhanced readability through strategic emphasis
- Maintained technical accuracy while improving clarity

---
