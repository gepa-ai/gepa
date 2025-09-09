# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from typing import Any

from gepa.adapters.generic_rag_adapter.vector_store_interface import VectorStoreInterface


class LanceDBVectorStore(VectorStoreInterface):
    """
    LanceDB implementation of the VectorStoreInterface.

    LanceDB is a developer-friendly, serverless vector database for AI applications.
    It provides excellent performance for both local development and production
    deployments with support for SQL-like filtering and modern PyArrow integration.
    """

    def __init__(self, db, table_name: str, embedding_function=None):
        """
        Initialize LanceDBVectorStore.

        Args:
            db: LanceDB database connection
            table_name: Name of the table to use
            embedding_function: Optional function to compute embeddings for queries
        """
        import importlib.util

        if importlib.util.find_spec("lancedb") is None:
            raise ImportError("LanceDB is required for LanceDBVectorStore. Install with: pip install litellm lancedb pyarrow")

        self.db = db
        self.table_name = table_name
        self.embedding_function = embedding_function

        # Try to open table if it exists, otherwise it will be created in add_documents
        try:
            self.table = self.db.open_table(table_name)
        except Exception:
            self.table = None  # Will be created when first adding documents

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
        if self.table is None:
            return []  # No documents added yet

        try:
            # Build query with vector search
            query_builder = self.table.search(query_vector).limit(k)

            # Add filters if provided
            if filters:
                filter_expr = self._convert_filters(filters)
                if filter_expr:
                    query_builder = query_builder.where(filter_expr)

            # Execute query and get results
            results = query_builder.to_pandas()

            return self._format_results(results)

        except Exception as e:
            raise RuntimeError(f"LanceDB vector search failed: {e!s}") from e

    def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add documents with their embeddings to the table."""
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
            # LanceDB requires consistent field structure
            record = {
                "id": doc_id,
                "vector": embedding,
                **doc,  # Include all document fields
            }
            data_to_insert.append(record)

        try:
            # Create table if it doesn't exist yet
            if self.table is None:
                self.table = self.db.create_table(self.table_name, data=data_to_insert)
            else:
                # Add data to existing table
                self.table.add(data_to_insert, mode="append")
            return ids

        except Exception as e:
            raise RuntimeError(f"Failed to add documents to LanceDB: {e!s}") from e

    def delete_documents(self, ids: list[str]) -> bool:
        """Delete documents by their IDs."""
        try:
            # Use parameterized filter to avoid injection
            if len(ids) == 1:
                filter_expr = {"id": ids[0]}
            else:
                filter_expr = {"id": {"$in": ids}}

            # Delete documents
            self.table.delete(filter_expr)
            return True

        except Exception as e:
            raise RuntimeError(f"Failed to delete documents from LanceDB: {e!s}") from e

    def get_collection_info(self) -> dict[str, Any]:
        """Get metadata about the LanceDB table."""
        try:
            # Get table schema
            schema = self.table.schema

            # Count rows (this might be approximate for large tables)
            try:
                count_result = self.table.count_rows()
                row_count = count_result if isinstance(count_result, int) else 0
            except Exception:
                # Fallback: count by querying
                try:
                    sample_df = self.table.to_pandas()
                    row_count = len(sample_df)
                except Exception:
                    row_count = 0

            # Extract vector field information
            vector_field = None
            dimension = 0
            for field in schema:
                # Check if field is a list type (vector field)
                if hasattr(field.type, "value_type") and "float" in str(field.type).lower():
                    vector_field = field.name
                    # Try to get dimension from list type
                    if hasattr(field.type, "list_size"):
                        dimension = field.type.list_size
                    elif "vector" in field.name.lower():
                        # This is likely our vector field
                        vector_field = field.name

            # Get table version and other metadata
            version = getattr(self.table, "version", "unknown")

            return {
                "name": self.table_name,
                "document_count": row_count,
                "dimension": dimension,
                "vector_store_type": "lancedb",
                "vector_field": vector_field,
                "version": version,
                "schema": str(schema),
            }

        except Exception as e:
            # Fallback info if detailed info fails
            return {
                "name": self.table_name,
                "document_count": 0,
                "dimension": 0,
                "vector_store_type": "lancedb",
                "error": str(e),
            }

    def supports_hybrid_search(self) -> bool:
        """LanceDB supports hybrid search through full-text search + vector search."""
        return True

    def hybrid_search(
        self,
        query: str,
        k: int = 5,
        alpha: float = 0.5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining vector similarity and full-text search.
        """
        try:
            # Import FTS query if available
            try:
                from lancedb.query import FtsQuery

                # Build hybrid query
                FtsQuery(query=query)

                # Get embedding for vector search
                if self.embedding_function:
                    query_vector = self.embedding_function(query)
                    if hasattr(query_vector, "tolist"):
                        query_vector = query_vector.tolist()

                    # Perform hybrid search (this is a simplified version)
                    # In practice, you might want to combine scores differently
                    query_builder = self.table.search(query_vector, query_type="hybrid").limit(k)
                else:
                    # Fall back to FTS only
                    query_builder = self.table.search(query, query_type="fts").limit(k)

                # Add filters if provided
                if filters:
                    filter_expr = self._convert_filters(filters)
                    if filter_expr:
                        query_builder = query_builder.where(filter_expr)

                # Execute query
                results = query_builder.to_pandas()
                return self._format_results(results)

            except ImportError:
                # Fall back to vector search only
                return self.similarity_search(query, k, filters)

        except Exception:
            # Fall back to regular vector search
            return self.similarity_search(query, k, filters)

    def _format_results(self, results_df) -> list[dict[str, Any]]:
        """Convert LanceDB results to standardized format."""
        documents = []

        if results_df is None or len(results_df) == 0:
            return documents

        for _, row in results_df.iterrows():
            row_dict = row.to_dict()

            # Extract content - try different field names
            content = ""
            content_fields = ["content", "text", "document", "body", "description"]
            for field in content_fields:
                if row_dict.get(field):
                    content = str(row_dict[field])
                    break

            # If no content field found, combine text fields
            if not content:
                text_properties = []
                system_fields = ["id", "vector", "_distance"]
                for key, value in row_dict.items():
                    if (
                        isinstance(value, str)
                        and value.strip()
                        and key not in system_fields
                        and key not in content_fields
                    ):
                        text_properties.append(f"{key}: {value}")
                content = " | ".join(text_properties)

            # Create metadata (all fields except content and system fields)
            metadata = {}
            system_fields = ["id", "vector", "_distance"] + content_fields
            for key, value in row_dict.items():
                if key not in system_fields and value is not None:
                    # Convert numpy types to Python types for JSON serialization
                    if hasattr(value, "item"):
                        value = value.item()
                    elif hasattr(value, "tolist"):
                        value = value.tolist()
                    metadata[key] = value

            metadata["doc_id"] = str(row_dict.get("id", ""))

            # Extract similarity score (LanceDB includes _distance column)
            distance = row_dict.get("_distance", 0.0)
            if hasattr(distance, "item"):
                distance = distance.item()

            # Convert distance to similarity score (lower distance = higher similarity)
            # For L2 distance: similarity = 1 / (1 + distance)
            # For cosine distance: similarity = 1 - distance (if distance is in [0,2])
            if distance <= 1.0:
                score = max(0.0, 1.0 - distance)  # Cosine-like
            else:
                score = 1.0 / (1.0 + distance)  # L2-like

            documents.append(
                {
                    "content": content,
                    "metadata": metadata,
                    "score": float(score),
                }
            )

        return documents

    def _convert_filters(self, filters: dict[str, Any]) -> str:
        """Convert generic filters to LanceDB SQL-like expressions."""
        if not filters:
            return None

        expressions = []

        for key, value in filters.items():
            if isinstance(value, str):
                # String exact match - properly escape quotes and backslashes
                escaped_value = value.replace("\\", "\\\\").replace("'", "''")
                expressions.append(f"{key} = '{escaped_value}'")
            elif isinstance(value, int | float):
                # Numeric exact match
                expressions.append(f"{key} = {value}")
            elif isinstance(value, bool):
                # Boolean match
                expressions.append(f"{key} = {value}")
            elif isinstance(value, list):
                # IN clause for multiple values
                if all(isinstance(v, str) for v in value):
                    # String values - properly escape
                    escaped_values = []
                    for v in value:
                        escaped_v = v.replace("'", "''")
                        escaped_values.append(f"'{escaped_v}'")
                    values_str = ", ".join(escaped_values)
                    expressions.append(f"{key} IN ({values_str})")
                else:
                    # Numeric values
                    values_str = ", ".join(str(v) for v in value)
                    expressions.append(f"{key} IN ({values_str})")
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
                    expressions.append("(" + " AND ".join(range_conditions) + ")")

        if expressions:
            return " AND ".join(expressions)

        return None

    @classmethod
    def create_local(cls, table_name: str, embedding_function=None, db_path: str = "./lancedb", vector_size: int = 384):
        """Create a local LanceDB vector store."""
        import importlib.util

        if importlib.util.find_spec("lancedb") is None or importlib.util.find_spec("pyarrow") is None:
            raise ImportError("LanceDB and PyArrow are required. Install with: pip install litellm lancedb pyarrow")

        import lancedb

        # Connect to local database
        db = lancedb.connect(db_path)

        # For LanceDB, we'll create the table when first adding documents
        # This allows LanceDB to infer the schema from actual data, avoiding conflicts

        return cls(db, table_name, embedding_function)

    @classmethod
    def create_remote(
        cls,
        table_name: str,
        embedding_function=None,
        uri: str | None = None,
        api_key: str | None = None,
        region: str = "us-east-1",
        vector_size: int = 384,
    ):
        """Create a remote LanceDB vector store (LanceDB Cloud)."""
        import importlib.util

        if importlib.util.find_spec("lancedb") is None or importlib.util.find_spec("pyarrow") is None:
            raise ImportError("LanceDB and PyArrow are required. Install with: pip install litellm lancedb pyarrow")

        import lancedb

        if not uri or not api_key:
            raise ValueError("URI and API key are required for remote LanceDB connection")

        # Connect to remote database
        db = lancedb.connect(uri, api_key=api_key, region=region)

        # For LanceDB, we'll create the table when first adding documents
        # This allows LanceDB to infer the schema from actual data, avoiding conflicts

        return cls(db, table_name, embedding_function)
