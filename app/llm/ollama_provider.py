"""Local Ollama models - the zero-cost fallback path.

Ollama exposes an OpenAI-compatible endpoint at ``/v1``, so this reuses the whole
OpenAI provider and overrides only what genuinely differs: structured output.
Ollama supports ``format: json`` but not OpenAI's strict json_schema mode, so the
schema goes into the prompt and the response is parsed defensively.

Expect this path to be worse at citation discipline and at abstaining. Measure it
against gpt-4o-mini with the eval harness before switching (ARCHITECTURE 4.1).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from app.config import settings
from app.llm.base import Message, system
from app.llm.openai_provider import OpenAIProvider, _json_fallback

T = TypeVar("T", bound=BaseModel)


class OllamaProvider(OpenAIProvider):
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 180.0,  # local generation is slower than an API hop
    ) -> None:
        base = (base_url or settings.llm.ollama_base_url).rstrip("/")
        super().__init__(
            model=model or settings.llm.ollama_model,
            api_key="ollama",  # required by the client, ignored by the server
            base_url=f"{base}/v1",
            timeout=timeout,
        )

    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        instruction = system(
            "Respond with a single JSON object and nothing else. No prose, no code "
            f"fences. It must match this JSON Schema:\n{json.dumps(schema.model_json_schema())}"
        )
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[instruction.as_dict()] + [m.as_dict() for m in messages],
            temperature=0.0 if temperature is None else temperature,
            response_format={"type": "json_object"},
        )
        return _json_fallback(response.choices[0].message.content or "", schema)
