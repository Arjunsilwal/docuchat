"""Shared test fixtures. Tests never call the real embedding model or the Claude API."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import pytest

from ragbot.chunking import chunk_documents
from ragbot.loaders import load_documents
from ragbot.store import VectorStore


class HashEmbedder:
    """Deterministic bag-of-words embedding: each word is hashed into one of 256 slots.

    Not semantic like a real model, but fast, offline and good enough to test plumbing.
    """

    dims = 256

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dims
            for word in text.lower().split():
                slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims
                vector[slot] += 1.0
            norm = math.sqrt(sum(v * v for v in vector)) or 1.0
            vectors.append([v / norm for v in vector])
        return vectors


class FakeLLM:
    """Records what it was sent and streams a canned answer."""

    def __init__(self, answer: str = "Employees get 16 weeks of parental leave [1]."):
        self.answer = answer
        self.calls: list[dict] = []

    def stream(self, system: str, messages: list[dict]):
        self.calls.append({"system": system, "messages": messages})
        yield from self.answer.split(" ")


@pytest.fixture
def store(tmp_path) -> VectorStore:
    """A vector store filled with the sample documents in data/."""
    vector_store = VectorStore(str(tmp_path / "db"), "test", embedder=HashEmbedder())
    chunks = chunk_documents(load_documents("data"), chunk_size=800, chunk_overlap=150)
    vector_store.add(chunks)
    return vector_store