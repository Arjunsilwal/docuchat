"""Streamlit chat UI.  Run:  streamlit run app.py"""

from __future__ import annotations

import os

import anthropic
import streamlit as st

from ragbot.config import settings
from ragbot.ingest import ingest
from ragbot.llm import ClaudeChat
from ragbot.pipeline import RAGPipeline
from ragbot.retriever import HybridRetriever, RetrievedChunk
from ragbot.store import VectorStore

st.set_page_config(page_title="DocuChat", page_icon="📄")


@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_pipeline() -> RAGPipeline | None:
    """Built once per server process, then reused for every message."""
    store = VectorStore(settings.db_dir, settings.collection)
    if store.count() == 0:
        # First run (e.g. a fresh cloud deployment): build the index automatically
        ingest(settings.data_dir, store, settings.chunk_size, settings.chunk_overlap)
    if store.count() == 0:
        return None
    retriever = HybridRetriever(store, candidates=settings.candidates)
    llm = ClaudeChat(settings.model, settings.max_tokens)
    return RAGPipeline(retriever, llm, settings.top_k, settings.max_history_turns)


def render_sources(sources: list[RetrievedChunk]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for number, item in enumerate(sources, start=1):
            chunk = item.chunk
            location = chunk.section or chunk.source
            if chunk.page:
                location += f", page {chunk.page}"
            st.markdown(f"**[{number}] {chunk.source}**: {location}")
            st.caption(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))


# ---- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.header("📄 DocuChat")
    st.write(
        "Ask questions about Acme Corp's handbook, IT security policy and product FAQ. "
        "Answers come only from those documents, with numbered citations."
    )
    st.caption(f"Model: `{settings.model}` · top-k: {settings.top_k}")
    st.caption("Retrieval: BM25 + vector search, fused with RRF")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("Chat with your documents")

if not os.getenv("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")
    st.stop()

pipeline = load_pipeline()
if pipeline is None:
    st.warning("No documents found. Add .md, .txt or .pdf files to the data/ folder, then reload.")
    st.stop()

# ---- Chat ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {"role", "content", "sources"?}

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

if question := st.chat_input("e.g. How much parental leave do we get?"):
    with st.chat_message("user"):
        st.markdown(question)

    # Send only plain question/answer text as history, not old retrieved documents
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

    with st.chat_message("assistant"):
        try:
            response = pipeline.ask(question, history)
            answer = st.write_stream(response.stream)
            render_sources(response.sources)
        except anthropic.APIError as error:
            st.error(f"Claude API error: {error}")
            st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": response.sources}
    )