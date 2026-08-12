"""gpt-4o-mini via the OpenAI API.

Uses the ``openai`` SDK directly rather than ``langchain-openai``. The wrapper
buys nothing here - we need token accounting, native structured output and
streaming, all of which the SDK exposes and the wrapper hides.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import TypeVar

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.llm.base import Completion, LLMError, Message

T = TypeVar("T", bound=BaseModel)

# Retry only what is genuinely transient. A 400 for a malformed request will
# fail identically on every attempt, and retrying it just burns latency.
_RETRYABLE = (RateLimitError, APIConnectionError)


class OpenAIProvider:
    """OpenAI-compatible chat provider.

    Also serves any endpoint that speaks the same wire format - Ollama subclasses
    this and only overrides the parts that genuinely differ.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._model = model or settings.llm.llm_model
        self._client = AsyncOpenAI(
            api_key=api_key or settings.llm.openai_api_key.get_secret_value() or "not-set",
            base_url=base_url,
            timeout=timeout,
            max_retries=0,  # tenacity owns the retry policy
        )

    @property
    def model(self) -> str:
        return self._model

    def _payload(
        self,
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict:
        return {
            "model": self._model,
            "messages": [m.as_dict() for m in messages],
            "temperature": (settings.llm.llm_temperature if temperature is None else temperature),
            "max_tokens": max_tokens or settings.llm.llm_max_tokens,
        }

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        try:
            response = await self._client.chat.completions.create(
                **self._payload(messages, temperature, max_tokens)
            )
        except APIStatusError as exc:
            raise LLMError(f"{self._model} returned {exc.status_code}: {exc.message}") from exc

        usage = response.usage
        return Completion(
            text=response.choices[0].message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Token stream. Not retried - a partially delivered answer cannot be
        replayed without the caller seeing it twice."""
        stream = await self._client.chat.completions.create(
            **self._payload(messages, temperature, max_tokens), stream=True
        )
        async for chunk in stream:
            if chunk.choices and (delta := chunk.choices[0].delta.content):
                yield delta

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    async def structured(
        self,
        messages: Sequence[Message],
        schema: type[T],
        *,
        temperature: float | None = None,
    ) -> T:
        """Native JSON-schema constrained decoding."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[m.as_dict() for m in messages],
            temperature=0.0 if temperature is None else temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "strict": True,
                    "schema": _strict_schema(schema),
                },
            },
        )
        raw = response.choices[0].message.content or "{}"
        try:
            return schema.model_validate_json(raw)
        except ValidationError as exc:
            raise LLMError(f"{schema.__name__} validation failed: {exc}") from exc

    async def health(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False


def _strict_schema(schema: type[BaseModel]) -> dict:
    """Coerce a Pydantic schema into OpenAI strict mode.

    Strict mode requires ``additionalProperties: false`` on every object and every
    property listed as required - optionality is expressed with a nullable type
    instead. Pydantic does not emit that shape, so we adjust it here rather than
    constraining how the rest of the codebase writes its models.
    """
    raw = schema.model_json_schema()

    def walk(node: object) -> None:
        if isinstance(node, dict):
            # A `$ref` must stand alone. Pydantic emits enum fields as
            # {"$ref": "#/$defs/Intent", "description": "..."} and OpenAI rejects
            # the whole request with "$ref cannot have keywords {'description'}".
            #
            # This failed silently for every routed turn: the router fell back to
            # heuristics, so intent classification and entity extraction were not
            # running at all while the system looked healthy.
            if "$ref" in node and len(node) > 1:
                ref = node["$ref"]
                node.clear()
                node["$ref"] = ref
                return

            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                if props := node.get("properties"):
                    node["required"] = list(props.keys())

            for value in list(node.values()):
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(raw)
    return raw


def _json_fallback(raw: str, schema: type[T]) -> T:
    """Recover a JSON object from a model that wrapped it in prose or fences.

    Only used by providers without native constrained decoding.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"no JSON object in response: {raw[:200]}")
    try:
        return schema.model_validate(json.loads(text[start : end + 1]))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise LLMError(f"{schema.__name__} parse failed: {exc}") from exc
