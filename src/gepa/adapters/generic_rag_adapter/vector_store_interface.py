# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from abc import ABC, abstractmethod
from typing import Any


class VectorStoreInterface(ABC):
    """
    Abstract interface for vector store operations in RAG systems.

    This interface defines the core operations needed for retrieval-augmented generation,
    enabling GEPA to work with any vector store implementation (ChromaDB, Weaviate, Qdrant,
    Pinecone, Milvus, etc.) through a unified API.

    The interface supports:
    - Semantic similarity search
    - Vector-based search with pre-computed embeddings
    - Hybrid search (semantic + keyword, where supported)
    - Metadata filtering and collection introspection

    Implementing this interface allows your vector store to be used with GEPA's
    evolutionary prompt optimization for RAG systems.

    Example:
        .. code-block:: python

            class MyVectorStore(VectorStoreInterface):
                def similarity_search(self, query, k=5, filters=None):
                    # Your implementation
                    return documents

            vector_store = MyVectorStore()
            adapter = GenericRAGAdapter(vector_store=vector_store)
    """

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search for documents semantically similar to the query text.

        This method performs semantic similarity search using the vector store's
        default embedding model or configured vectorizer.

        Args:
            query: Text query to search for similar documents
            k: Maximum number of documents to return (default: 5)
            filters: Optional metadata filters to constrain search.
                Format: {"key": "value"} or {"key": {"$op": value}}

        Returns:
            List of documents ordered by similarity score (highest first).
            Each document is a dictionary with:
            - "content" (str): The document text content
            - "metadata" (dict): Document metadata including any doc_id
            - "score" (float): Similarity score between 0.0 and 1.0 (higher = more similar)

        Raises:
            NotImplementedError: Must be implemented by concrete vector store classes

        Example:
            .. code-block:: python

                results = vector_store.similarity_search(
                    query="machine learning algorithms",
                    k=3,
                    filters={"category": "AI"}
                )
                print(results[0]["content"])  # Most similar document
        """
        pass

    @abstractmethod
    def vector_search(
        self,
        query_vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search using a pre-computed query embedding vector.

        This method allows direct vector similarity search when you already have
        the query embedding, avoiding the need for additional embedding computation.

        Args:
            query_vector: Pre-computed embedding vector for the query.
                Must match the dimensionality of vectors in the collection.
            k: Maximum number of documents to return (default: 5)
            filters: Optional metadata filters to constrain search

        Returns:
            List of documents ordered by vector similarity (highest first).
            Same format as similarity_search().

        Raises:
            NotImplementedError: Must be implemented by concrete vector store classes
            ValueError: If query_vector dimensions don't match collection

        Example:
            .. code-block:: python

                import numpy as np
                query_vector = embedding_model.encode("machine learning")
                results = vector_store.vector_search(query_vector.tolist(), k=5)
        """
        pass

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid semantic + keyword search combining vector and text-based matching.

        This method combines semantic similarity (vector search) with keyword-based
        search (like BM25) to leverage both approaches. The alpha parameter controls
        the balance between the two search methods.

        Args:
            query: Text query to search for
            k: Maximum number of documents to return (default: 5)
            alpha: Weight for semantic vs keyword search (default: 0.5)
                - 0.0 = pure keyword/BM25 search
                - 1.0 = pure semantic/vector search
                - 0.5 = balanced hybrid search
            filters: Optional metadata filters to constrain search

        Returns:
            List of documents ordered by hybrid similarity score (highest first).
            Same format as similarity_search().

        Note:
            If hybrid search is not supported by the vector store implementation,
            this method falls back to similarity_search() with a warning.
            Use supports_hybrid_search() to check availability.

        Example:
            .. code-block:: python

                # Balanced hybrid search
                results = vector_store.hybrid_search("AI algorithms", alpha=0.5)

                # More semantic-focused
                results = vector_store.hybrid_search("AI algorithms", alpha=0.8)
        """
        # Default fallback implementation
        return self.similarity_search(query, k, filters)

    @abstractmethod
    def get_collection_info(self) -> dict[str, Any]:
        """
        Get metadata and statistics about the vector store collection.

        This method provides introspection capabilities for the collection,
        returning key information about its configuration and contents.

        Returns:
            Dictionary containing collection metadata with keys:
            - "name" (str): Collection/index name
            - "document_count" (int): Total number of documents
            - "dimension" (int): Vector embedding dimension (0 if unknown)
            - "vector_store_type" (str): Type of vector store (e.g., "chromadb", "weaviate")
            - Additional store-specific metadata as available

        Raises:
            NotImplementedError: Must be implemented by concrete vector store classes

        Example:
            .. code-block:: python

                info = vector_store.get_collection_info()
                print(f"Collection {info['name']} has {info['document_count']} documents")
                print(f"Vector dimension: {info['dimension']}")
        """
        pass

    def get_embedding_dimension(self) -> int:
        """
        Get the embedding dimension of the vector store.

        Returns:
            Dimension of the vectors in the collection
        """
        info = self.get_collection_info()
        return info.get("dimension", 0)

    def supports_hybrid_search(self) -> bool:
        """
        Check if the vector store supports hybrid search.

        Returns:
            True if hybrid search is supported, False otherwise
        """
        return False

    def supports_metadata_filtering(self) -> bool:
        """
        Check if the vector store supports metadata filtering.

        Returns:
            True if metadata filtering is supported, False otherwise
        """
        return True
