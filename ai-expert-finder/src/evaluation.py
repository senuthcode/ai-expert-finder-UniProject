"""
Evaluation Metrics Module for Information Retrieval (MAP@K, Recall@K, NDCG@K).
"""

import math
from typing import List, Any


def precision_at_k(actual: List[Any], predicted: List[Any], k: int) -> float:
    """Computes Precision at rank K."""
    if not predicted or k <= 0:
        return 0.0
    predicted_k = predicted[:k]
    actual_set = set(actual)
    hits = sum(1 for item in predicted_k if item in actual_set)
    return hits / float(k)


def recall_at_k(actual: List[Any], predicted: List[Any], k: int) -> float:
    """Computes Recall at rank K."""
    if not actual or k <= 0:
        return 0.0
    predicted_k = predicted[:k]
    actual_set = set(actual)
    hits = sum(1 for item in predicted_k if item in actual_set)
    return hits / float(len(actual_set))


def average_precision_at_k(actual: List[Any], predicted: List[Any], k: int) -> float:
    """Computes Average Precision (AP) at rank K."""
    if not actual or not predicted:
        return 0.0
    predicted_k = predicted[:k]
    actual_set = set(actual)
    hits = 0
    sum_precisions = 0.0

    for i, item in enumerate(predicted_k):
        if item in actual_set:
            hits += 1
            sum_precisions += hits / float(i + 1)

    return sum_precisions / float(min(len(actual_set), k)) if actual_set else 0.0


def ndcg_at_k(actual: List[Any], predicted: List[Any], k: int) -> float:
    """Computes Normalized Discounted Cumulative Gain (NDCG) at rank K."""
    if not actual or not predicted:
        return 0.0
    predicted_k = predicted[:k]
    actual_set = set(actual)

    dcg = 0.0
    for i, item in enumerate(predicted_k):
        if item in actual_set:
            dcg += 1.0 / math.log2(i + 2)

    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(actual_set), k)))
    return dcg / idcg if idcg > 0.0 else 0.0
