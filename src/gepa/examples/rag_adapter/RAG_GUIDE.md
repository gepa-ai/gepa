# GEPA Generic RAG Adapter Examples

This directory contains focused examples demonstrating how to use GEPA's Generic RAG Adapter with different vector stores. Each example supports both local Ollama models and cloud-based LLMs with real AI/ML knowledge bases.

## 📂 Examples Overview

| Example | Vector Store | Docker Required | Key Features |
|---------|--------------|----------------|--------------|
| `chromadb_optimization.py` | ChromaDB | ❌ No | Local storage, simple setup, semantic search |
| `weaviate_optimization.py` | Weaviate | ✅ Yes | Hybrid search, production-ready, advanced features |
| `lancedb_optimization.py` | LanceDB | ❌ No | Serverless, columnar format, developer-friendly |
| `milvus_optimization.py` | Milvus | ❌ No* | Cloud-native, scalable, Milvus Lite for local dev |
| `qdrant_optimization.py` | Qdrant | ❌ No* | Advanced filtering, payload search, high performance |

*Docker optional for production deployments

## 🚀 Quick Start Guide

### Prerequisites

1. **Install Dependencies:**
   
   **Option A: Install with GEPA (Recommended)**
   ```bash
   # Install GEPA with specific vector database support
   pip install gepa[chromadb]    # For ChromaDB
   pip install gepa[weaviate]    # For Weaviate  
   pip install gepa[lancedb]     # For LanceDB
   pip install gepa[milvus]      # For Milvus
   pip install gepa[qdrant]      # For Qdrant
   
   # Or install all at once
   pip install gepa[chromadb,weaviate,lancedb,milvus,qdrant]
   ```
   
   **Option B: Manual Installation**
   ```bash
   # Core GEPA
   pip install gepa
   
   # Vector database clients (install as needed)
   pip install chromadb          # For ChromaDB
   pip install weaviate-client   # For Weaviate
   pip install lancedb pyarrow   # For LanceDB
   pip install pymilvus          # For Milvus
   pip install qdrant-client     # For Qdrant
   
   # LLM support
   pip install litellm
   ```
   
   **Embedding Models (choose one):**
   ```bash
   pip install sentence-transformers  # For local embeddings
   # OR use cloud embeddings (no additional install needed)
   ```

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

## 🔵 ChromaDB Example

ChromaDB is perfect for getting started - it's lightweight, runs locally, and requires no external services.

**✅ No Docker Required:** Runs completely locally.

### Basic Usage

```bash
# Navigate to the examples directory
cd src/gepa/examples/rag_adapter

# Run with default settings (Ollama qwen3:8b)
python chromadb_optimization.py

# Quick test (no optimization)
python chromadb_optimization.py --max-iterations 0

# Full optimization run
python chromadb_optimization.py --max-iterations 20
```

### With Different Models

```bash
# Use different Ollama model
python chromadb_optimization.py --model ollama/llama3.1:8b

# Use cloud model (requires API key)
python chromadb_optimization.py --model gpt-4o-mini --max-iterations 10

# Use Anthropic model
python chromadb_optimization.py --model claude-3-haiku-20240307
```


## 🟠 Weaviate Example

Weaviate offers advanced features like hybrid search (semantic + keyword) and is production-ready.

**⚠️ Docker Required:** This example requires a running Weaviate instance.

### Setup Weaviate

```bash
# Start Weaviate with Docker (required)
docker run -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:1.26.1

# Verify Weaviate is running
curl http://localhost:8080/v1/meta
```

### Basic Usage

```bash
# Navigate to the examples directory
cd src/gepa/examples/rag_adapter

# Run with default settings (Ollama qwen3:8b)
python weaviate_optimization.py

# Skip data setup (use existing collection)
python weaviate_optimization.py --skip-setup

# Full optimization run
python weaviate_optimization.py --max-iterations 15
```

### With Different Models

```bash
# Use cloud model with Weaviate
python weaviate_optimization.py --model gpt-4o-mini --max-iterations 10

# Use different local model
python weaviate_optimization.py --model ollama/llama3.1:8b
```


## 🟢 LanceDB Example

LanceDB is a serverless vector database built on the Lance columnar format, perfect for developer-friendly local development.

**✅ No Docker Required:** Serverless, creates local files.

### Key Features
- **Serverless**: No external services required
- **Columnar Format**: Built on Apache Arrow/Lance for performance
- **Developer-Friendly**: Simple setup, works out of the box
- **Local Storage**: Creates local database files

### Basic Usage

```bash
# Navigate to the examples directory
cd src/gepa/examples/rag_adapter

# Run with default settings (Ollama llama3.1:8b)
python lancedb_optimization.py

# Quick test (no optimization)
python lancedb_optimization.py --max-iterations 0

# Full optimization run
python lancedb_optimization.py --max-iterations 20
```

### With Different Models

```bash
# Use different Ollama model
python lancedb_optimization.py --model ollama/qwen3:8b

# Use cloud model (requires API key)
python lancedb_optimization.py --model gpt-4o-mini --max-iterations 10

# Use Anthropic model
python lancedb_optimization.py --model claude-3-haiku-20240307
```


## 🔵 Milvus Example

Milvus is a cloud-native vector database designed for large-scale AI applications. This example uses Milvus Lite for local development.

**✅ No Docker Required:** Uses Milvus Lite (local SQLite) by default.

### Key Features
- **Milvus Lite**: Local SQLite-based version (no Docker required)
- **Cloud-Native**: Production-ready with full Milvus server option
- **Scalable**: Designed for large-scale applications
- **Local Development**: Perfect for testing and development

### Setup Milvus

```bash
# Default: Milvus Lite (no setup required)
# Creates local ./milvus_demo.db file automatically

# Optional: Full Milvus Server (for production)
docker run -d -p 19530:19530 milvusdb/milvus:latest standalone
```

### Basic Usage

```bash
# Navigate to the examples directory
cd src/gepa/examples/rag_adapter

# Run with default settings (Ollama llama3.1:8b, Milvus Lite)
python milvus_optimization.py

# Quick test (no optimization)
python milvus_optimization.py --max-iterations 0

# Full optimization run
python milvus_optimization.py --max-iterations 15
```

### With Different Models

```bash
# Use cloud model with Milvus
python milvus_optimization.py --model gpt-4o-mini --max-iterations 10

# Use different local model
python milvus_optimization.py --model ollama/qwen3:8b
```


## 🟡 Qdrant Example

Qdrant is a high-performance vector database with advanced filtering and payload search capabilities.

**✅ No Docker Required:** Uses in-memory mode by default.

### Key Features
- **Advanced Filtering**: Rich metadata filtering capabilities
- **Payload Search**: Search by both vectors and metadata
- **High Performance**: Optimized for speed and scale
- **Flexible Deployment**: In-memory, local file, or remote server

### Setup Qdrant

```bash
# Default: In-memory mode (no setup required)
python qdrant_optimization.py

# Option 1: Local persistent storage
python qdrant_optimization.py --path ./qdrant_db

# Option 2: Remote Qdrant server (optional)
docker run -p 6333:6333 qdrant/qdrant
python qdrant_optimization.py --host localhost --port 6333
```

### Basic Usage

```bash
# Navigate to the examples directory
cd src/gepa/examples/rag_adapter

# Run with default settings (Ollama qwen3:8b, in-memory)
python qdrant_optimization.py

# With persistent local storage
python qdrant_optimization.py --path ./qdrant_db

# With remote Qdrant server
python qdrant_optimization.py --host localhost --port 6333

# Full optimization run
python qdrant_optimization.py --max-iterations 15
```

### With Different Models

```bash
# Use cloud model with Qdrant
python qdrant_optimization.py --model gpt-4o-mini --max-iterations 10

# Use different local model
python qdrant_optimization.py --model ollama/llama3.1:8b

# With API key for secured instances
python qdrant_optimization.py --api-key your-api-key
```


## ⚙️ Configuration Options

### Command Line Arguments

All examples support these common arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `ollama/qwen3:8b` (ChromaDB/Weaviate/Qdrant)<br>`ollama/llama3.1:8b` (LanceDB/Milvus) | LLM model to use |
| `--embedding-model` | `ollama/nomic-embed-text:latest` | Embedding model |
| `--max-iterations` | `5` | GEPA optimization iterations |
| `--verbose` | `False` | Enable detailed logging |

**Database-specific arguments:**

| Database | Argument | Default | Description |
|----------|----------|---------|-------------|
| **Weaviate** | `--skip-setup` | `False` | Use existing Weaviate collection |
| **Qdrant** | `--path` | `:memory:` | Local Qdrant database path |
| **Qdrant** | `--host` | `None` | Qdrant server host (for remote) |
| **Qdrant** | `--port` | `6333` | Qdrant server port |
| **Qdrant** | `--api-key` | `None` | Qdrant API key (for secured instances) |

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
# Test all examples (no optimization)
python chromadb_optimization.py --max-iterations 0
python weaviate_optimization.py --max-iterations 0
python lancedb_optimization.py --max-iterations 0
python milvus_optimization.py --max-iterations 0
python qdrant_optimization.py --max-iterations 0

# Test external services
curl http://localhost:8080/v1/meta  # Weaviate
curl http://localhost:6333/health   # Qdrant
ollama list                         # Ollama models
```

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Make sure you're in the right directory
cd /path/to/gepa/src/gepa/examples/rag_adapter
python chromadb_optimization.py
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
pip install lancedb pyarrow

# Check sentence-transformers for embeddings
python -c "import sentence_transformers; print('sentence-transformers installed')"
pip install sentence-transformers
```

#### Milvus Issues
```bash
# Check Milvus Lite installation
python -c "import pymilvus; print('PyMilvus installed')"

# Install missing dependencies
pip install pymilvus

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
pip install qdrant-client

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
python chromadb_optimization.py --model gpt-4o-mini

# Reduce iterations
python chromadb_optimization.py --max-iterations 2

# Test without optimization first
python chromadb_optimization.py --max-iterations 0
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


### Quick Decision Guide

- **New to vector databases?** → Start with **ChromaDB** or **LanceDB**
- **Need advanced filtering?** → Choose **Qdrant**
- **Building for scale?** → Use **Milvus** or **Weaviate**
- **Want hybrid search?** → Go with **Weaviate**
- **Prefer serverless?** → Try **LanceDB**