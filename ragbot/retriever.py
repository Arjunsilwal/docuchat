"""Hybrid retrieval: BM25 keyword search + vector search, merged with Reciprocal Rank Fusion."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from ragbot.chunking import Chunk
from ragbot.store import VectorStore

_STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in", "is", "it",
    "many", "much", "my", "of", "on", "or", "the", "to", "what", "when", "where", "which",
    "who", "with", "you", "your",
}  # fmt: skip


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Keeps codes like 'exp-204' together as 'exp', '204'."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge several ranked id lists. score(d) = sum over lists of 1 / (k + rank).

    Uses ranks only, so it does not matter that BM25 scores and cosine
    similarities are on completely different scales.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class HybridRetriever:
    MODES = ("hybrid", "vector", "keyword")

    def __init__(self, store: VectorStore, candidates: int = 20):
        self.store = store
        self.candidates = candidates
        chunks = store.all_chunks()
        self.chunks_by_id = {c.id: c for c in chunks}
        self._ids = [c.id for c in chunks]
        # BM25 needs a non-empty corpus
        self._bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def keyword_search(self, query: str, n: int) -> list[str]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._ids, scores), key=lambda item: item[1], reverse=True)
        return [chunk_id for chunk_id, score in ranked[:n] if score > 0]

    def vector_search(self, query: str, n: int) -> list[str]:
        return [chunk_id for chunk_id, _ in self.store.search(query, n)]

    def retrieve(self, query: str, top_k: int = 5, mode: str = "hybrid") -> list[RetrievedChunk]:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        rankings = []
        if mode in ("hybrid", "vector"):
            rankings.append(self.vector_search(query, self.candidates))
        if mode in ("hybrid", "keyword"):
            rankings.append(self.keyword_search(query, self.candidates))
        fused = reciprocal_rank_fusion(rankings)[:top_k]
        return [RetrievedChunk(self.chunks_by_id[cid], score) for cid, score in fused]