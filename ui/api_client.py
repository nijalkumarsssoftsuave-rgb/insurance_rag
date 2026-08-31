"""Typed HTTP client for the FastAPI backend, including SSE streaming.

The only module in the UI that knows HTTP exists. Keeping it that way is what
makes the eventual React port a rewrite of one file rather than the whole
frontend (ARCHITECTURE 9.3).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from typing import Any

import httpx

# 8000 is a common default and was already taken on the dev machine by an
# unrelated service. Keep this in step with API_PORT in .env.
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8010")

# Ingestion is slow on CPU, but the upload call itself returns 202 immediately.
# A long timeout here would only mask a backend that is actually wedged.
UPLOAD_TIMEOUT = 120.0
DEFAULT_TIMEOUT = 30.0
# A turn runs the embedder and cross-encoder locally on CPU; the
# first one also loads ~3.4 GB of weights.
CHAT_TIMEOUT = 300.0


class ApiError(RuntimeError):
    """Backend returned an error, with a message worth showing the user."""


@dataclass(slots=True)
class ChatAnswer:
    conversation_id: str
    answer: str
    citations: list[dict]
    intent: str | None = None
    confidence: float = 0.0
    abstained: bool = False
    needs_human: bool = False
    verified: bool = True
    below_threshold: bool = False
    top_score: float = 0.0
    query_variants: list[str] | None = None
    timings_ms: dict | None = None
    latency_ms: int = 0
    retrieved_chunk_ids: list[str] | None = None
    context_blocks: list[dict] | None = None
    broadened: bool = False

    @classmethod
    def from_payload(cls, data: dict) -> ChatAnswer:
        """Build from the API response, ignoring fields this client does not know.

        `cls(**data)` raises TypeError the moment the backend adds a response
        field - which it did, and it took the chat page down with a traceback in
        front of the user. The UI should degrade by ignoring what it cannot use,
        not crash: it is a presentation layer, not a schema validator.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass(slots=True)
class UploadResult:
    document_id: str
    filename: str
    status: str
    duplicate: bool
    message: str


class ApiClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or API_BASE_URL).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=DEFAULT_TIMEOUT)

    # ─────────────────────────────────────────────────────────── health

    def health(self) -> dict[str, Any]:
        return self._get("/api/v1/health")

    def is_up(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────── chat

    def chat_stream(self, message: str, conversation_id: str | None = None):
        """Yield ("stage", label) events, then ("answer", ChatAnswer).

        Stage events exist because a CPU-local turn takes seconds. Reporting what
        the system is doing is more honest, and more legible, than a spinner.
        """
        payload = {"message": message}
        if conversation_id:
            payload["conversation_id"] = conversation_id

        try:
            with self._client.stream(
                "POST", "/api/v1/chat/stream", json=payload, timeout=CHAT_TIMEOUT
            ) as response:
                if response.status_code >= 400:
                    response.read()
                    raise ApiError(_detail(response))

                event = None
                for line in response.iter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        data = json.loads(line.split(":", 1)[1].strip())
                        if event == "stage":
                            yield "stage", data.get("label", "")
                        elif event == "answer":
                            yield "answer", ChatAnswer.from_payload(data)
                        elif event == "error":
                            raise ApiError(data.get("message", "Unknown error"))
        except httpx.RequestError as exc:
            raise ApiError(f"Could not reach the API at {self.base_url}: {exc}") from exc

    def history(self, conversation_id: str) -> list[dict]:
        return self._get(f"/api/v1/chat/{conversation_id}/history")

    # ──────────────────────────────────────────────────────── documents

    def upload(
        self, filename: str, data: bytes, *, doc_type: str | None = None
    ) -> UploadResult:
        """Upload a document for indexing.

        ``doc_type=None`` means "let the server work it out": the API defaults to
        ``other``, and the ingestion pipeline then overwrites it with the type
        detected from the filename and cover page. Passing an explicit value
        suppresses that detection, because a caller who states the type is taken
        at their word.
        """
        form = {} if doc_type is None else {"doc_type": doc_type}
        try:
            response = self._client.post(
                "/api/v1/documents",
                files={"file": (filename, data)},
                data=form,
                timeout=UPLOAD_TIMEOUT,
            )
        except httpx.RequestError as exc:
            raise ApiError(f"Could not reach the API at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise ApiError(_detail(response))

        payload = response.json()
        return UploadResult(
            document_id=payload["document_id"],
            filename=payload["filename"],
            status=payload["status"],
            duplicate=payload.get("duplicate", False),
            message=payload.get("message", ""),
        )

    def list_documents(self) -> list[dict[str, Any]]:
        return self._get("/api/v1/documents")

    def document(self, document_id: str) -> dict[str, Any]:
        return self._get(f"/api/v1/documents/{document_id}")

    def stats(self) -> dict[str, Any]:
        return self._get("/api/v1/documents/stats")

    def delete_document(self, document_id: str) -> None:
        response = self._client.delete(f"/api/v1/documents/{document_id}")
        if response.status_code >= 400:
            raise ApiError(_detail(response))

    # ──────────────────────────────────────────────────────── internals

    def _get(self, path: str) -> Any:
        try:
            response = self._client.get(path)
        except httpx.RequestError as exc:
            raise ApiError(f"Could not reach the API at {self.base_url}: {exc}") from exc
        if response.status_code >= 400:
            raise ApiError(_detail(response))
        return response.json()


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        return str(body.get("detail") or body)
    except Exception:
        return f"{response.status_code}: {response.text[:300]}"
