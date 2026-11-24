from ragbot.pipeline import RAGPipeline
from ragbot.prompts import NO_ANSWER, SYSTEM_PROMPT, build_user_message, trim_history
from ragbot.retriever import HybridRetriever
from tests.conftest import FakeLLM


def test_system_prompt_has_grounding_rules():
    assert "ONLY" in SYSTEM_PROMPT
    assert NO_ANSWER in SYSTEM_PROMPT


def test_user_message_puts_documents_before_question(store):
    sources = HybridRetriever(store).retrieve("parental leave", top_k=2)
    message = build_user_message("How long is parental leave?", sources)
    assert message.index("<documents>") < message.index("<question>")
    assert '<document index="1"' in message and '<document index="2"' in message


def test_trim_history_keeps_last_turns_and_starts_with_user():
    history = []
    for i in range(10):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    trimmed = trim_history(history, max_turns=2)
    assert [m["content"] for m in trimmed] == ["q8", "a8", "q9", "a9"]
    assert trim_history(history, max_turns=0) == []


def test_pipeline_streams_answer_and_returns_sources(store):
    llm = FakeLLM()
    pipeline = RAGPipeline(HybridRetriever(store), llm, top_k=3)
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! Ask me about Acme documents."},
    ]
    response = pipeline.ask("How many weeks of parental leave?", history)

    assert len(response.sources) == 3
    answer = " ".join(response.stream)
    assert "16 weeks" in answer

    sent = llm.calls[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert "<question>" in sent[-1]["content"]