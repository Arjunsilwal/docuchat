"""The same RAG pipeline built with LangChain components.

retriever (DocuChatRetriever) -> prompt (ChatPromptTemplate) -> model (ChatAnthropic)
-> output parser, joined with the pipe operator (LCEL). It has the same `ask()`
method as RAGPipeline, so the Streamlit app can use either one.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ragbot.lc_retriever import DocuChatRetriever, to_retrieved_chunk
from ragbot.pipeline import RAGResponse
from ragbot.prompts import SYSTEM_PROMPT, build_user_message, trim_history


def default_chat_model(model: str, max_tokens: int) -> BaseChatModel:
    # Reads ANTHROPIC_API_KEY from the environment, like our own ClaudeChat.
    # These are the field names behind the shorter aliases model= and max_tokens=;
    # spelling them out (with timeout and stop) keeps VS Code's type checker happy.
    return ChatAnthropic(
        model_name=model,
        max_tokens_to_sample=max_tokens,
        max_retries=3,
        timeout=None,
        stop=None,
    )


class LangChainRAGPipeline:
    def __init__(
        self,
        retriever: DocuChatRetriever,
        llm: BaseChatModel,
        max_history_turns: int = 6,
    ):
        self.retriever = retriever
        self.max_history_turns = max_history_turns
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("history"),
                ("human", "{user_message}"),
            ]
        )
        # LCEL: each step's output flows into the next with the | operator
        self.chain = self.prompt | llm | StrOutputParser()

    def build_inputs(self, question: str, history: list[dict] | None = None):
        """Retrieve documents and prepare the prompt variables."""
        documents = self.retriever.invoke(question)
        sources = [to_retrieved_chunk(doc) for doc in documents]
        trimmed = trim_history(history or [], self.max_history_turns)
        inputs = {
            "history": [(m["role"], m["content"]) for m in trimmed],
            "user_message": build_user_message(question, sources),
        }
        return sources, inputs

    def ask(self, question: str, history: list[dict] | None = None) -> RAGResponse:
        sources, inputs = self.build_inputs(question, history)
        return RAGResponse(sources=sources, stream=self.chain.stream(inputs))