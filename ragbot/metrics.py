"""Retrieval metrics. All functions take ranked retrieved ids and a set of relevant ids."""

from __future__ import annotations


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant chunks that appear in the top k results."""
    if not relevant:
        return 0.0
    found = len(set(retrieved[:k]) & relevant)
    return found / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k results that are relevant."""
    if k <= 0:
        return 0.0
    found = len(set(retrieved[:k]) & relevant)
    return found / k


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """1.0 if at least one relevant chunk is in the top k, else 0.0."""
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant result (0.0 if none). Averaged over queries this is MRR."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0
