"""Ingest pipeline: load -> chunk -> embed -> store.

Run:  python -m ragbot.ingest
"""

from __future__ import annotations

import argparse
import time

from ragbot.chunking import chunk_documents
from ragbot.config import settings
from ragbot.loaders import load_documents
from ragbot.store import VectorStore


def ingest(data_dir: str, store: VectorStore, chunk_size: int, chunk_overlap: int) -> int:
    documents = load_documents(data_dir)
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    store.reset()
    store.add(chunks)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into the vector store.")
    parser.add_argument("--data-dir", default=settings.data_dir)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    args = parser.parse_args()

    started = time.perf_counter()
    store = VectorStore(settings.db_dir, settings.collection)
    count = ingest(args.data_dir, store, args.chunk_size, args.chunk_overlap)
    elapsed = time.perf_counter() - started
    print(f"Indexed {count} chunks from '{args.data_dir}' in {elapsed:.1f}s -> {settings.db_dir}/")


if __name__ == "__main__":
    main()
