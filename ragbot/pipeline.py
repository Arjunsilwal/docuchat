"""The RAG pipeline: retrieve -> build prompt -> generate."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

from ragbot.prompts import SYSTEM_PROMPT, build_user_message, trim_history
from ragbot.retriever import HybridRetriever, RetrievedChunk


class ChatModel(Protocol):
    def stream(self, system: str, messages: list[dict]) -> Iterator[str]: ...


@dataclass
class RAGResponse:
    sources: list[RetrievedChunk]  # known before generation starts, so the UI can show them
    stream: Iterator[str]  # the answer, token by token


class RAGPipeline:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm: ChatModel,
        top_k: int = 5,
        max_history_turns: int = 6,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.max_history_turns = max_history_turns

    def ask(self, question: str, history: list[dict] | None = None) -> RAGResponse:
        """Answer `question`. `history` is a list of {"role", "content"} plain-text turns."""
        sources = self.retriever.retrieve(question, top_k=self.top_k)
        messages = trim_history(history or [], self.max_history_turns)
        messages = messages + [{"role": "user", "content": build_user_message(question, sources)}]
        return RAGResponse(sources=sources, stream=self.llm.stream(SYSTEM_PROMPT, messages))