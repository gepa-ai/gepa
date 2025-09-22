# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from .evaluation_metrics import RAGEvaluationMetrics
from .generic_rag_adapter import (
    GenericRAGAdapter,
    RAGDataInst,
    RAGOutput,
    RAGTrajectory,
)
from .rag_pipeline import RAGPipeline
from .vector_store_interface import VectorStoreInterface
from .vector_stores.chroma_store import ChromaVectorStore
from .vector_stores.weaviate_store import WeaviateVectorStore

# Optional vector stores - import only if dependencies are available
try:
    from .vector_stores.qdrant_store import QdrantVectorStore
    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False

try:
    from .vector_stores.milvus_store import MilvusVectorStore
    _MILVUS_AVAILABLE = True
except ImportError:
    _MILVUS_AVAILABLE = False

try:
    from .vector_stores.lancedb_store import LanceDBVectorStore
    _LANCEDB_AVAILABLE = True
except ImportError:
    _LANCEDB_AVAILABLE = False

__all__ = [
    "GenericRAGAdapter",
    "RAGDataInst",
    "RAGOutput",
    "RAGTrajectory",
    "VectorStoreInterface",
    "ChromaVectorStore",
    "WeaviateVectorStore",
    "RAGPipeline",
    "RAGEvaluationMetrics",
]

# Add optional vector stores to __all__ if available
if _QDRANT_AVAILABLE:
    __all__.append("QdrantVectorStore")
if _MILVUS_AVAILABLE:
    __all__.append("MilvusVectorStore")
if _LANCEDB_AVAILABLE:
    __all__.append("LanceDBVectorStore")
