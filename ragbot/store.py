"""Vector store: ChromaDB holds each chunk's text, metadata and embedding."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import chromadb
import numpy as np

from ragbot.chunking import Chunk


class Embedder(Protocol):
    """Anything that turns a list of texts into a list of vectors."""

    def __call__(self, texts: Sequence[str]) -> list[list[float]]: ...


def default_embedder() -> Embedder:
    """all-MiniLM-L6-v2 (384 dimensions), run locally through ONNX.

    The model (about 80 MB) downloads automatically the first time it is used.
    """
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    model = DefaultEmbeddingFunction()
    return lambda texts: [list(map(float, v)) for v in model(list(texts))]


class VectorStore:
    def __init__(self, db_dir: str, collection: str, embedder: Embedder | None = None):
        self.client = chromadb.PersistentClient(path=db_dir)
        self.collection_name = collection
        self.embedder = embedder or default_embedder()
        self.collection = self._get_collection()

    def _get_collection(self):
        # We compute embeddings ourselves, so Chroma gets no embedding function.
        # Cosine distance = 1 - cosine similarity, the usual choice for text embeddings.
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Delete everything so re-ingesting never leaves stale chunks behind."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:  # collection did not exist yet
            pass
        self.collection = self._get_collection()

    def add(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self.collection.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self._vectors([c.text for c in batch]),
                metadatas=[
                    {"source": c.source, "section": c.section, "page": c.page} for c in batch
                ],
            )

    def _vectors(self, texts: Sequence[str]) -> np.ndarray:
        """Embed texts as a float32 NumPy array, the format ChromaDB stores."""
        return np.asarray(self.embedder(texts), dtype=np.float32)

    def count(self) -> int:
        return self.collection.count()

    def search(self, query: str, n_results: int) -> list[tuple[str, float]]:
        """Return (chunk_id, cosine_similarity) pairs, most similar first."""
        n_results = min(n_results, self.count())
        if n_results == 0:
            return []
        result = self.collection.query(
            query_embeddings=self._vectors([query]),
            n_results=n_results,
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = result["distances"][0] if result["distances"] else []
        return [(chunk_id, 1.0 - distance) for chunk_id, distance in zip(ids, distances)]

    def all_chunks(self) -> list[Chunk]:
        """Load every chunk (used to build the in-memory BM25 keyword index)."""
        result = self.collection.get(include=["documents", "metadatas"])
        documents = result["documents"] or []
        metadatas = result["metadatas"] or []
        chunks = []
        for chunk_id, text, meta in zip(result["ids"], documents, metadatas):
            page = meta.get("page")
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    source=str(meta["source"]),
                    section=str(meta.get("section", "")),
                    page=page if isinstance(page, int) else 0,
                )
            )
        return chunks
