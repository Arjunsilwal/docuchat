"""Central settings. Every value can be overridden with an environment variable or .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

# pyrefly: ignore[missing-import]
from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # Where documents live and where the vector database is saved
    data_dir: str = os.getenv("DATA_DIR", "data")
    db_dir: str = os.getenv("DB_DIR", ".chroma")
    collection: str = os.getenv("COLLECTION", "docuchat")

    # Chunking, measured in characters (about 4 characters per token,
    # so 1200 characters is roughly 300 tokens)
    chunk_size: int = _int("CHUNK_SIZE", 1200)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 200)

    # Retrieval: fetch `candidates` from each search, keep the best `top_k` after fusion
    candidates: int = _int("CANDIDATES", 20)
    top_k: int = _int("TOP_K", 5)

    # Generation
    model: str = os.getenv("CLAUDE_MODEL", "claude-haiku-5-5")
    max_tokens: int = _int("MAX_TOKENS", 800)
    max_history_turns: int = _int("MAX_HISTORY_TURNS", 6)

    # Which pipeline the app uses: "custom" (our own code) or "langchain"
    pipeline: str = os.getenv("PIPELINE", "custom")


settings = Settings()
