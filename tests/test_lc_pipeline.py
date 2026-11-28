from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ragbot.lc_pipeline import LangChainRAGPipeline
from ragbot.lc_retriever import DocuChatRetriever, to_document, to_retrieved_chunk
from ragbot.prompts import SYSTEM_PROMPT
from ragbot.retriever import HybridRetriever


def make_pipeline(store, answer="Employees get 16 weeks of parental leave [1]."):
    retriever = DocuChatRetriever(hybrid=HybridRetriever(store), top_k=3)
    return LangChainRAGPipeline(retriever, FakeListChatModel(responses=[answer]))


def test_retriever_returns_langchain_documents_with_metadata(store):
    retriever = DocuChatRetriever(hybrid=HybridRetriever(store), top_k=3)
    docs = retriever.invoke("How many weeks of parental leave?")
    assert len(docs) == 3
    assert {"id", "source", "section", "page", "score"} <= set(docs[0].metadata)
    assert any("16 weeks" in doc.page_content for doc in docs)


def test_document_conversion_round_trip(store):
    item = HybridRetriever(store).retrieve("parental leave", top_k=1)[0]
    back = to_retrieved_chunk(to_document(item))
    assert back.chunk == item.chunk
    assert back.score == item.score


def test_prompt_has_system_history_and_documents(store):
    pipeline = make_pipeline(store)
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! Ask me about Acme documents."},
    ]
    _, inputs = pipeline.build_inputs("How many weeks of parental leave?", history)
    messages = pipeline.prompt.invoke(inputs).to_messages()

    assert [m.type for m in messages] == ["system", "human", "ai", "human"]
    assert messages[0].content == SYSTEM_PROMPT
    assert "<documents>" in messages[-1].content and "<question>" in messages[-1].content


def test_ask_streams_answer_and_returns_sources(store):
    response = make_pipeline(store).ask("How many weeks of parental leave?")
    assert len(response.sources) == 3
    assert "".join(response.stream) == "Employees get 16 weeks of parental leave [1]."