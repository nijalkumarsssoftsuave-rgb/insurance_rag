"""LLMProvider protocol: complete, stream, structured_output.

The protocol exists so the zero-cost path is a config change rather than a
refactor: ``LLM_PROVIDER=ollama`` swaps the implementation and nothing upstream
knows. Whether that swap is acceptable is a question for the eval harness, not
for the code (ARCHITECTURE 4.1).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]
T = TypeVar("T", bound=BaseModel)


@dataclass(slots=True)
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def system(content: str) -> Message:
    return Message(role="system", content=content)


def user(content: str) -> Message:
    return Message(role="user", content=content)


def assistant(content: str) -> Message:
    return Message(role="assistant", content=content)


@dataclass(slots=True)
class Completion:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMError(RuntimeError):
    """Provider call failed after retries."""


@runtime_checkable
class LLMProvider(Protocol):
    """Implementations: ``OpenAIProvider``, ``OllamaProvider``."""

    @property
    def model(self) -> str: ...

    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion: ...

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        """Constrained decoding into ``schema``.

        Used by the router and the verifier, where a free-text answer that has to
        be parsed defensively is a source of silent failure.
        """
        ...
