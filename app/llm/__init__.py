"""LLM provider selection."""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.llm.base import (
    Completion,
    LLMError,
    LLMProvider,
    Message,
    assistant,
    system,
    user,
)


@lru_cache(maxsize=1)
def get_llm() -> LLMProvider:
    """The configured provider. One client per process - it pools connections."""
    if settings.llm.llm_provider == "ollama":
        from app.llm.ollama_provider import OllamaProvider

        return OllamaProvider()

    from app.llm.openai_provider import OpenAIProvider

    return OpenAIProvider()


__all__ = [
    "Completion",
    "LLMError",
    "LLMProvider",
    "Message",
    "assistant",
    "get_llm",
    "system",
    "user",
]
