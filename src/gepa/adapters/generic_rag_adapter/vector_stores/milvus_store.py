# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any

from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface


class MilvusVectorStore(VectorStoreInterface):
    """
    Milvus implementation of the VectorStoreInterface.

    Milvus is a cloud-native vector database built for scalable similarity search
    and AI applications. It provides excellent performance for large-scale deployments
    and supports various index types and distance metrics.
    """

    def __init__(self, client, collection_name: str, embedding_function=None):
        """
        Initialize MilvusVectorStore.

        Args:
            client: MilvusClient instance
            collection_name: Name of the collection to use
            embedding_function: Optional function to compute embeddings for queries
        """
        import importlib.util

        if importlib.util.find_spec("pymilvus") is None:
            raise ImportError("Milvus client is required for MilvusVectorStore. Install with: pip install litellm pymilvus")

        self.client = client
        self.collection_name = collection_name
        self.embedding_function = embedding_function

        # Verify collection exists
        if not self.client.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' not found. Please create the collection first.")

        # Load collection into memory for search operations
        try:
            self.client.load_collection(collection_name)
        except Exception:
            # Collection might already be loaded, which is fine
            pass

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search for documents similar to the query text using embeddings."""
        if self.embedding_function is None:
            raise ValueError("No embedding function provided for similarity search")

        # Compute embeddings for the query
        try:
            query_vector = self.embedding_function(query)
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to compute embeddings for query: {e!s}") from e

        # Use vector search with computed embeddings
        return self.vector_search(query_vector, k, filters)

    def vector_search(
        self,
        query_vector: list[float],
        k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Search using a pre-computed query vector."""
        try:
            # Convert filters to Milvus expression format
            filter_expr = self._convert_filters(filters) if filters else None

            # Perform vector search
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],  # Milvus expects list of vectors
                limit=k,
                filter=filter_expr,
                output_fields=["*"],  # Return all fields
            )

            return self._format_results(results)

        except Exception as e:
            raise RuntimeError(f"Milvus vector search failed: {e!s}") from e

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents with their embeddings to the collection."""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        elif len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")

        # Prepare data for insertion
        data_to_insert = []
        for doc_id, doc, embedding in zip(ids, documents, embeddings, strict=False):
            # Milvus requires consistent field structure
            record = {
                "id": doc_id,
                "vector": embedding,
                **doc,  # Include all document fields
            }
            data_to_insert.append(record)

        try:
            # Insert data into collection
            result = self.client.insert(collection_name=self.collection_name, data=data_to_insert)

            # Check if insertion was successful
            if "insert_count" in result and result["insert_count"] == len(data_to_insert):
                return ids
            else:
                raise RuntimeError(
                    f"Insertion failed. Expected {len(data_to_insert)}, got {result.get('insert_count', 0)}"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to add documents to Milvus: {e!s}") from e

    def delete_documents(self, ids: list[str]) -> bool:
        """Delete documents by their IDs."""
        try:
            # Use parameterized ID-based deletion to avoid injection
            result = self.client.delete(collection_name=self.collection_name, ids=ids)

            return "delete_count" in result and result["delete_count"] > 0

        except Exception as e:
            raise RuntimeError(f"Failed to delete documents from Milvus: {e!s}") from e

    def get_collection_info(self) -> dict[str, Any]:
        """Get metadata about the Milvus collection."""
        try:
            # Get collection description
            description = self.client.describe_collection(self.collection_name)

            # Get collection statistics
            stats = self.client.get_collection_stats(self.collection_name)

            # Extract vector field information
            vector_field = None
            dimension = 0
            for field in description.get("fields", []):
                if field.get("type") == "FloatVector":
                    vector_field = field.get("name", "vector")
                    dimension = field.get("params", {}).get("dim", 0)
                    break

            return {
                "name": self.collection_name,
                "document_count": stats.get("row_count", 0),
                "dimension": dimension,
                "vector_store_type": "milvus",
                "vector_field": vector_field,
                "schema": description,
            }

        except Exception as e:
            # Fallback info if detailed info fails
            return {
                "name": self.collection_name,
                "document_count": 0,
                "dimension": 0,
                "vector_store_type": "milvus",
                "error": str(e),
            }

    def supports_hybrid_search(self) -> bool:
        """Milvus supports hybrid search through dense + sparse vectors."""
        return True

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining vector similarity and other search methods.

        Note: This implementation focuses on vector search with filtering.
        Full hybrid search with sparse vectors would require additional setup.
        """
        # For now, implement as filtered vector search
        # Future enhancement could include sparse vector search for text matching
        return self.similarity_search(query, k, filters)

    def _format_results(self, results) -> list[dict[str, Any]]:
        """Convert Milvus results to standardized format."""
        documents = []

        # Milvus returns nested list: results[0] contains hits for first query vector
        if not results or not results[0]:
            return documents

        hits = results[0]  # Get hits for our single query vector

        for hit in hits:
            # Extract fields from hit
            hit_id = hit.get("id", "")
            distance = hit.get("distance", 0.0)

            # Extract content - try different field names
            content = ""
            content_fields = ["content", "text", "document", "body", "description"]
            for field in content_fields:
                if field in hit:
                    content = hit[field]
                    break

            # If no content field found, combine text fields
            if not content:
                text_properties = []
                for key, value in hit.items():
                    if isinstance(value, str) and value.strip() and key not in ["id", "distance"]:
                        text_properties.append(f"{key}: {value}")
                content = " | ".join(text_properties)

            # Create metadata (all fields except content and system fields)
            metadata = {}
            system_fields = ["id", "distance", "vector"] + content_fields
            for key, value in hit.items():
                if key not in system_fields:
                    metadata[key] = value
            metadata["doc_id"] = str(hit_id)

            # Convert distance to similarity score (Milvus returns distance, lower is better)
            # For cosine distance: similarity = 1 - distance
            # For L2 distance: similarity = 1 / (1 + distance)
            score = max(0.0, 1.0 - distance) if distance <= 1.0 else 1.0 / (1.0 + distance)

            documents.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "score": score,
                }
            )

        return documents

    def _convert_filters(self, filters: dict[str, Any]) -> str:
        """Convert generic filters to Milvus expression format."""
        if not filters:
            return None

        expressions = []

        for key, value in filters.items():
            if isinstance(value, str):
                # String exact match
                expressions.append(f'{key} == "{value}"')
            elif isinstance(value, int | float):
                # Numeric exact match
                expressions.append(f"{key} == {value}")
            elif isinstance(value, list):
                # IN clause for multiple values
                if all(isinstance(v, str) for v in value):
                    values_str = '", "'.join(value)
                    expressions.append(f'{key} in ["{values_str}"]')
                else:
                    values_str = ", ".join(str(v) for v in value)
                    expressions.append(f"{key} in [{values_str}]")
            elif isinstance(value, dict):
                # Range queries
                range_conditions = []
                if "gte" in value:
                    range_conditions.append(f"{key} >= {value['gte']}")
                if "gt" in value:
                    range_conditions.append(f"{key} > {value['gt']}")
                if "lte" in value:
                    range_conditions.append(f"{key} <= {value['lte']}")
                if "lt" in value:
                    range_conditions.append(f"{key} < {value['lt']}")

                if range_conditions:
                    expressions.append(" and ".join(range_conditions))

        if expressions:
            return " and ".join(expressions)

        return None

    @classmethod
    def create_local(
        cls, collection_name: str, embedding_function=None, vector_size: int = 384, uri: str = "./milvus_demo.db"
    ):
        """Create a local Milvus vector store (Milvus Lite)."""
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as e:
            raise ImportError("Milvus client is required. Install with: pip install litellm pymilvus") from e

        client = MilvusClient(uri=uri)

        # Create collection if it doesn't exist
        if not client.has_collection(collection_name):
            # Create collection with explicit schema to avoid max_length issues
            schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

            # Add ID field (string)
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=512,  # Maximum length for ID strings
            )

            # Add vector field
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=vector_size)

            # Add content field
            schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=65535,  # Large max length for content
            )

            # Create collection with schema
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )

            # Create index for vector field
            index_params = client.prepare_index_params()
            index_params.add_index(field_name="vector", metric_type="COSINE")
            client.create_index(collection_name=collection_name, index_params=index_params)

        return cls(client, collection_name, embedding_function)

    @classmethod
    def create_remote(
        cls,
        collection_name: str,
        embedding_function=None,
        uri: str = "http://localhost:19530",
        user: str = "",
        password: str = "",
        token: str = "",
        vector_size: int = 384,
    ):
        """Create a remote Milvus vector store."""
        try:
            from pymilvus import DataType, MilvusClient
        except ImportError as e:
            raise ImportError("Milvus client is required. Install with: pip install litellm pymilvus") from e

        # Connect to remote Milvus
        client = MilvusClient(
            uri=uri,
            user=user if user else None,
            password=password if password else None,
            token=token if token else None,
        )

        # Create collection if it doesn't exist
        if not client.has_collection(collection_name):
            # Create collection with explicit schema to avoid max_length issues
            schema = client.create_schema(auto_id=False, enable_dynamic_field=True)

            # Add ID field (string)
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=512,  # Maximum length for ID strings
            )

            # Add vector field
            schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=vector_size)

            # Add content field
            schema.add_field(
                field_name="content",
                datatype=DataType.VARCHAR,
                max_length=65535,  # Large max length for content
            )

            # Create collection with schema
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
            )

            # Create index for vector field
            index_params = client.prepare_index_params()
            index_params.add_index(field_name="vector", metric_type="COSINE")
            client.create_index(collection_name=collection_name, index_params=index_params)

        return cls(client, collection_name, embedding_function)
