# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

from gepa.adapters.generic_rag_adapter.evaluation_metrics import RAGEvaluationMetrics
from gepa.adapters.generic_rag_adapter.vector_stores.chroma_store import ChromaVectorStore


def _store() -> ChromaVectorStore:
    # _format_results does not touch the client, so skip __init__ (which needs chromadb installed)
    return object.__new__(ChromaVectorStore)


def test_format_results_exposes_chroma_id_as_doc_id():
    results = {
        "ids": [["doc_a", "doc_b"]],
        "documents": [["text a", "text b"]],
        "metadatas": [[{"topic": "x"}, None]],
        "distances": [[0.1, 0.4]],
    }

    docs = _store()._format_results(results)

    assert [d["metadata"]["doc_id"] for d in docs] == ["doc_a", "doc_b"]
    assert docs[0]["metadata"]["topic"] == "x"
    # The caller's metadata dict is not mutated
    assert results["metadatas"][0][0] == {"topic": "x"}

    metrics = RAGEvaluationMetrics().evaluate_retrieval(docs, ["doc_b"])
    assert metrics["retrieval_recall"] == 1.0
    assert metrics["retrieval_mrr"] == 0.5


def test_format_results_keeps_explicit_doc_id_metadata():
    results = {
        "ids": [["chroma-1"]],
        "documents": [["text"]],
        "metadatas": [[{"doc_id": "custom"}]],
        "distances": [[0.0]],
    }

    docs = _store()._format_results(results)

    assert docs[0]["metadata"]["doc_id"] == "custom"
