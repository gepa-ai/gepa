# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import re
from typing import Any


class RAGEvaluationMetrics:
    """
    Evaluation metrics for RAG systems.

    Provides both retrieval and generation quality metrics
    for comprehensive RAG system evaluation.
    """

    def evaluate_retrieval(self, retrieved_docs: list[dict[str, Any]], relevant_doc_ids: list[str]) -> dict[str, float]:
        """
        Evaluate retrieval quality metrics.

        Args:
            retrieved_docs: List of retrieved documents with metadata
            relevant_doc_ids: List of ground truth relevant document IDs

        Returns:
            Dictionary with retrieval metrics (precision, recall, f1, mrr)
        """
        if not retrieved_docs or not relevant_doc_ids:
            return {"retrieval_precision": 0.0, "retrieval_recall": 0.0, "retrieval_f1": 0.0, "retrieval_mrr": 0.0}

        # Extract document IDs from retrieved docs
        retrieved_ids = []
        for doc in retrieved_docs:
            doc_id = doc.get("metadata", {}).get("doc_id") or doc.get("metadata", {}).get("id")
            if doc_id:
                retrieved_ids.append(str(doc_id))

        relevant_set = set(relevant_doc_ids)
        retrieved_set = set(retrieved_ids)

        # Calculate precision and recall
        if len(retrieved_set) == 0:
            precision = 0.0
        else:
            precision = len(relevant_set.intersection(retrieved_set)) / len(retrieved_set)

        if len(relevant_set) == 0:
            recall = 0.0
        else:
            recall = len(relevant_set.intersection(retrieved_set)) / len(relevant_set)

        # Calculate F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        # Calculate Mean Reciprocal Rank (MRR)
        mrr = 0.0
        for i, retrieved_id in enumerate(retrieved_ids):
            if retrieved_id in relevant_set:
                mrr = 1.0 / (i + 1)
                break

        return {"retrieval_precision": precision, "retrieval_recall": recall, "retrieval_f1": f1, "retrieval_mrr": mrr}

    def evaluate_generation(self, generated_answer: str, ground_truth: str, context: str) -> dict[str, float]:
        """
        Evaluate generation quality metrics.

        Args:
            generated_answer: Generated answer text
            ground_truth: Ground truth answer
            context: Retrieved context used for generation

        Returns:
            Dictionary with generation metrics
        """
        # Exact match (case-insensitive)
        exact_match = self._exact_match(generated_answer, ground_truth)

        # F1 score based on token overlap
        f1_score = self._token_f1(generated_answer, ground_truth)

        # BLEU-like score
        bleu_score = self._simple_bleu(generated_answer, ground_truth)

        # Answer relevance (simple keyword overlap with context)
        relevance_score = self._answer_relevance(generated_answer, context)

        # Faithfulness (how well the answer is supported by context)
        faithfulness_score = self._faithfulness_score(generated_answer, context)

        return {
            "exact_match": float(exact_match),
            "token_f1": f1_score,
            "bleu_score": bleu_score,
            "answer_relevance": relevance_score,
            "faithfulness": faithfulness_score,
            "answer_confidence": (f1_score + relevance_score + faithfulness_score) / 3.0,
        }

    def combined_rag_score(
        self,
        retrieval_metrics: dict[str, float],
        generation_metrics: dict[str, float],
        retrieval_weight: float = 0.3,
        generation_weight: float = 0.7,
    ) -> float:
        """
        Combine retrieval and generation metrics into a single score.

        Args:
            retrieval_metrics: Output from evaluate_retrieval
            generation_metrics: Output from evaluate_generation
            retrieval_weight: Weight for retrieval score
            generation_weight: Weight for generation score

        Returns:
            Combined score between 0 and 1
        """
        # Primary retrieval metric: F1 score
        retrieval_score = retrieval_metrics.get("retrieval_f1", 0.0)

        # Primary generation metric: weighted combination
        generation_score = (
            generation_metrics.get("token_f1", 0.0) * 0.4
            + generation_metrics.get("answer_relevance", 0.0) * 0.3
            + generation_metrics.get("faithfulness", 0.0) * 0.3
        )

        return retrieval_weight * retrieval_score + generation_weight * generation_score

    def _exact_match(self, prediction: str, ground_truth: str) -> bool:
        """Check if prediction exactly matches ground truth (case-insensitive)."""
        return prediction.strip().lower() == ground_truth.strip().lower()

    def _token_f1(self, prediction: str, ground_truth: str) -> float:
        """Calculate F1 score based on token overlap."""
        pred_tokens = set(self._normalize_text(prediction).split())
        truth_tokens = set(self._normalize_text(ground_truth).split())

        if len(pred_tokens) == 0 and len(truth_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0.0

        intersection = pred_tokens.intersection(truth_tokens)
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(truth_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def _simple_bleu(self, prediction: str, ground_truth: str, n: int = 2) -> float:
        """Simple BLEU-like score for n-gram overlap."""
        pred_words = self._normalize_text(prediction).split()
        truth_words = self._normalize_text(ground_truth).split()

        if len(pred_words) < n or len(truth_words) < n:
            return self._token_f1(prediction, ground_truth)

        pred_ngrams = {tuple(pred_words[i : i + n]) for i in range(len(pred_words) - n + 1)}
        truth_ngrams = {tuple(truth_words[i : i + n]) for i in range(len(truth_words) - n + 1)}

        if len(pred_ngrams) == 0 or len(truth_ngrams) == 0:
            return 0.0

        intersection = pred_ngrams.intersection(truth_ngrams)
        return len(intersection) / len(pred_ngrams)

    def _answer_relevance(self, answer: str, context: str) -> float:
        """Measure how well the answer relates to the provided context."""
        answer_words = set(self._normalize_text(answer).split())
        context_words = set(self._normalize_text(context).split())

        if len(answer_words) == 0:
            return 0.0

        overlap = answer_words.intersection(context_words)
        return len(overlap) / len(answer_words)

    def _faithfulness_score(self, answer: str, context: str) -> float:
        """
        Measure how well the answer is supported by the context.
        Simple implementation based on shared key phrases.
        """
        # Extract key phrases (sequences of 2+ words)
        answer_phrases = self._extract_phrases(answer)
        context_phrases = self._extract_phrases(context)

        if len(answer_phrases) == 0:
            return 1.0  # Empty answer is technically faithful

        supported_phrases = answer_phrases.intersection(context_phrases)
        return len(supported_phrases) / len(answer_phrases)

    def _extract_phrases(self, text: str, min_length: int = 2) -> set[str]:
        """Extract meaningful phrases from text."""
        words = self._normalize_text(text).split()
        phrases = set()

        # Add individual significant words (length > 3)
        for word in words:
            if len(word) > 3:
                phrases.add(word)

        # Add bi-grams and tri-grams
        for n in range(min_length, min(4, len(words) + 1)):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if len(phrase) > 5:  # Only meaningful phrases
                    phrases.add(phrase)

        return phrases

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase and remove extra whitespace
        text = text.lower().strip()
        # Remove punctuation and special characters
        text = re.sub(r"[^\w\s]", " ", text)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text
