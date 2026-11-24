"""Thin wrapper around the Claude Messages API with streaming."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import anthropic
from anthropic.types import MessageParam


class ClaudeChat:
    def __init__(
        self,
        model: str,
        max_tokens: int = 800,
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        # Reads ANTHROPIC_API_KEY from the environment; retries 429/5xx errors automatically
        self.client = client or anthropic.Anthropic(max_retries=3)

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        """Yield text as it is generated (Server-Sent Events under the hood)."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            # our messages are plain dicts in exactly the shape the API expects
            messages=cast(list[MessageParam], messages),
        ) as stream:
            yield from stream.text_stream