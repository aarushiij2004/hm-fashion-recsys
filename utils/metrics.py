"""
utils/metrics.py
─────────────────
Standard IR metrics for recommendation evaluation.
All functions take sets/lists of item indices.
"""

import numpy as np


def recall_at_k(positives: set, ranked: list, k: int) -> float:
    """Fraction of relevant items found in top-K predictions."""
    if not positives:
        return 0.0
    hits = len(set(ranked[:k]) & positives)
    return hits / min(len(positives), k)


def precision_at_k(positives: set, ranked: list, k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(set(ranked[:k]) & positives)
    return hits / k


def average_precision_at_k(positives: set, ranked: list, k: int) -> float:
    """Average precision: area under precision-recall curve up to K."""
    if not positives:
        return 0.0
    score, hits = 0.0, 0
    for i, item in enumerate(ranked[:k]):
        if item in positives:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(positives), k)


def map_at_k(all_positives: list[set], all_ranked: list[list], k: int) -> float:
    """Mean Average Precision @ K over a list of queries."""
    return float(np.mean([
        average_precision_at_k(pos, ranked, k)
        for pos, ranked in zip(all_positives, all_ranked)
    ]))


def dcg_at_k(positives: set, ranked: list, k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(ranked[:k]):
        if item in positives:
            dcg += 1.0 / np.log2(i + 2)
    return dcg


def ndcg_at_k(positives: set, ranked: list, k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K."""
    ideal = dcg_at_k(positives, list(positives), k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(positives, ranked, k) / ideal


def evaluate_all(val_interactions: dict, predictions: dict, k_values: list = None) -> dict:
    """
    Compute all metrics for a set of predictions.

    Args:
        val_interactions: {user_idx: [positive_item_idxs]}
        predictions:      {user_idx: [ranked_item_idxs]}
        k_values:         list of K values to evaluate at

    Returns:
        dict of metric_name → value
    """
    if k_values is None:
        k_values = [10, 20, 50, 100]

    results = {}
    users = [u for u in val_interactions if u in predictions and val_interactions[u]]

    for k in k_values:
        recalls, ndcgs, aps = [], [], []
        for u in users:
            pos  = set(val_interactions[u])
            pred = predictions[u]
            recalls.append(recall_at_k(pos, pred, k))
            ndcgs.append(ndcg_at_k(pos, pred, k))
            aps.append(average_precision_at_k(pos, pred, k))

        results[f"Recall@{k}"]    = round(float(np.mean(recalls)), 4)
        results[f"NDCG@{k}"]      = round(float(np.mean(ndcgs)), 4)
        results[f"MAP@{k}"]       = round(float(np.mean(aps)), 4)

    return results
