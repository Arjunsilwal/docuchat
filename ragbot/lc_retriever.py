"""DocuChat's hybrid retriever wrapped as a LangChain retriever.

LangChain chains and agents work with any `BaseRetriever`: an object that takes
a query and returns `Document`s. Wrapping our own HybridRetriever means the
LangChain pipeline searches exactly the same way (BM25 + vectors + RRF).
"""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from ragbot.chunking import Chunk
from ragbot.retriever import HybridRetriever, RetrievedChunk


class DocuChatRetriever(BaseRetriever):
    hybrid: HybridRetriever
    top_k: int = 5

    # HybridRetriever is a plain Python class, not a pydantic model
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [to_document(item) for item in self.hybrid.retrieve(query, top_k=self.top_k)]


def to_document(item: RetrievedChunk) -> Document:
    """Our RetrievedChunk -> LangChain Document (text + metadata)."""
    chunk = item.chunk
    return Document(
        page_content=chunk.text,
        metadata={
            "id": chunk.id,
            "source": chunk.source,
            "section": chunk.section,
            "page": chunk.page,
            "score": item.score,
        },
    )


def to_retrieved_chunk(doc: Document) -> RetrievedChunk:
    """LangChain Document -> our RetrievedChunk, so the UI and prompt code can be reused."""
    meta = doc.metadata
    chunk = Chunk(
        id=meta["id"],
        text=doc.page_content,
        source=meta["source"],
        section=meta.get("section", ""),
        page=meta.get("page", 0),
    )
    return RetrievedChunk(chunk=chunk, score=meta.get("score", 0.0))
