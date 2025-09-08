# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any

from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface


class ChromaVectorStore(VectorStoreInterface):
    """
    ChromaDB implementation of the VectorStoreInterface.

    ChromaDB is an open-source embedding database that's easy to use
    and perfect for local development and prototyping.
    """

    def __init__(self, client, collection_name: str, embedding_function=None):
        """
        Initialize ChromaVectorStore.

        Args:
            client: ChromaDB client instance
            collection_name: Name of the collection to use
            embedding_function: Optional embedding function for text queries
        """
        import importlib.util

        if importlib.util.find_spec("chromadb") is None:
            raise ImportError("ChromaDB is required for ChromaVectorStore. Install with: pip install litellm chromadb")

        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        # Get or create the collection
        try:
            self.collection = self.client.get_collection(name=collection_name, embedding_function=embedding_function)
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(name=collection_name, embedding_function=embedding_function)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for documents similar to the query text."""
        # Convert filters to ChromaDB format if provided
        where_clause = self._convert_filters(filters) if filters else None

        results = self.collection.query(
            query_texts=[query], n_results=k, where=where_clause, include=["documents", "metadatas", "distances"]
        )

        return self._format_results(results)

    def vector_search(
        self,
        query_vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using a pre-computed query vector."""
        where_clause = self._convert_filters(filters) if filters else None

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        return self._format_results(results)

    def get_collection_info(self) -> dict[str, Any]:
        """Get metadata about the ChromaDB collection."""
        count = self.collection.count()

        # Try to get a sample document to determine embedding dimension
        dimension = 0
        if count > 0:
            sample = self.collection.peek(limit=1)
            embeddings = sample.get("embeddings")
            if embeddings is not None and len(embeddings) > 0:
                dimension = len(embeddings[0])

        return {
            "name": self.collection_name,
            "document_count": count,
            "dimension": dimension,
            "vector_store_type": "chromadb",
        }

    def supports_metadata_filtering(self) -> bool:
        """ChromaDB supports metadata filtering."""
        return True

    def _convert_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
        """
        Convert generic filters to ChromaDB where clause format.

        Generic format: {"key": "value", "key2": {"$gt": 5}}
        ChromaDB format: {"key": {"$eq": "value"}, "key2": {"$gt": 5}}
        """
        chroma_filters = {}

        for key, value in filters.items():
            if isinstance(value, dict):
                # Already in operator format
                chroma_filters[key] = value
            else:
                # Convert to equality operator
                chroma_filters[key] = {"$eq": value}

        return chroma_filters

    def _format_results(self, results) -> list[dict[str, Any]]:
        """Convert ChromaDB results to standardized format."""
        documents = []

        if not results["documents"] or not results["documents"][0]:
            return documents

        docs = results["documents"][0]
        metadatas = results.get("metadatas", [None] * len(docs))[0] or [{}] * len(docs)
        distances = results.get("distances", [0.0] * len(docs))[0]

        for doc, metadata, distance in zip(docs, metadatas, distances, strict=False):
            # Convert distance to similarity score (higher is better)
            # ChromaDB uses cosine distance, so similarity = 1 - distance
            distance_val = self._extract_distance_value(distance)
            similarity_score = max(0.0, 1.0 - distance_val)

            documents.append({"content": doc, "metadata": metadata or {}, "score": similarity_score})

        return documents

    def _extract_distance_value(self, distance) -> float:
        """
        Helper to extract a scalar float from a distance value returned by ChromaDB.
        Handles numpy scalars, single-element lists/arrays, and plain floats.
        """
        try:
            if hasattr(distance, "item"):  # numpy scalar
                return distance.item()
            elif hasattr(distance, "__len__") and not isinstance(distance, (str, bytes)) and len(distance) == 1:
                return float(distance[0])
            else:
                return float(distance)
        except (TypeError, ValueError, IndexError) as e:
            import logging
            logging.warning(f"Unexpected distance format: {type(distance)}, value: {distance}, error: {e}")
            return 0.0  # Default fallback
    @classmethod
    def create_local(cls, persist_directory: str, collection_name: str, embedding_function=None) -> "ChromaVectorStore":
        """
        Create a ChromaVectorStore with local persistence.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            embedding_function: Optional embedding function

        Returns:
            ChromaVectorStore instance
        """
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("ChromaDB is required. Install with: pip install litellm chromadb") from e

        client = chromadb.PersistentClient(path=persist_directory)
        return cls(client, collection_name, embedding_function)

    @classmethod
    def create_memory(cls, collection_name: str, embedding_function=None) -> "ChromaVectorStore":
        """
        Create a ChromaVectorStore in memory (for testing).

        Args:
            collection_name: Name of the collection
            embedding_function: Optional embedding function

        Returns:
            ChromaVectorStore instance
        """
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("ChromaDB is required. Install with: pip install litellm chromadb") from e

        client = chromadb.Client()
        return cls(client, collection_name, embedding_function)
