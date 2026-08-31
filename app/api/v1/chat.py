"""POST /chat and /chat/stream - drives the LangGraph conversation.

Two endpoints for one graph. The streaming variant emits **stage events**, not
tokens: generation uses constrained decoding to enforce the citation contract, so
there is no partial text to stream. On CPU a turn takes seconds, and telling the
user *"searching your policy documents"* is far better than an idle spinner - it
also makes the pipeline legible during a walkthrough.

Every turn is written to `conversations` and `messages` with its retrieved chunk
ids, so an answer can be reconstructed later (ARCHITECTURE 10.4).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.core.enums import MessageRole
from app.db.models import Conversation, Message
from app.deps import DbSession, Subject
from app.graph.builder import get_graph
from app.graph.state import initial_state
from app.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_MESSAGE_CHARS = 4_000

# Node name -> what the user is told is happening. Anything unlisted is silent,
# so internal nodes do not leak into the interface.
STAGE_LABELS: dict[str, str] = {
    "guard": "Checking your message",
    "condense": "Reading the conversation so far",
    "route": "Understanding the question",
    "claim_lookup": "Looking up your claim",
    "clause_handoff": "Finding the clause behind the decision",
    "retrieve": "Searching your policy documents",
    "generate": "Composing the answer",
    "verify": "Checking every statement against the source",
    "abstain": "No reliable answer found",
    "scope_reply": "Answering",
}


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    conversation_id: str | None = Field(
        default=None, description="Omit to start a new conversation"
    )


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: list[dict] = Field(default_factory=list)
    intent: str | None = None
    confidence: float = 0.0
    abstained: bool = False
    needs_human: bool = False
    verified: bool = True
    below_threshold: bool = False
    top_score: float = 0.0
    query_variants: list[str] = Field(default_factory=list)
    timings_ms: dict[str, int] = Field(default_factory=dict)
    latency_ms: int = 0
    # The inspection view. The graph has always computed these; they used to die
    # in the node, which meant a wrong answer could not be told apart from a
    # wrong *retrieval* without re-running the query by hand. `context_blocks`
    # is everything that reached the model; `citations` is the subset the model
    # actually used, so the two together separate the two failure classes.
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    context_blocks: list[dict] = Field(default_factory=list)
    broadened: bool = False


# How many prior messages are replayed into the graph. `condense_node` only reads
# the last six, so fetching more would cost a wider query for text it discards.
HISTORY_LIMIT = 6


async def _load_history(session: DbSession, conversation_id: str) -> list:
    """Prior turns of this conversation, oldest first, as LangChain messages.

    Read from Postgres rather than the checkpointer so history survives an API
    restart and is correct when more than one worker serves the same conversation.
    Failure is non-fatal: a turn answered without history is worse than one
    answered with it, but far better than a turn that errors.
    """
    try:
        rows = (
            await session.scalars(
                select(Message)
                .join(Conversation, Message.conversation_id == Conversation.id)
                .where(Conversation.thread_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(HISTORY_LIMIT)
            )
        ).all()
    except Exception as exc:
        log.warning("Could not load conversation history", error=str(exc))
        return []

    return [
        HumanMessage(content=row.content)
        if row.role is MessageRole.USER
        else AIMessage(content=row.content)
        for row in reversed(rows)
    ]


async def _run_graph(question: str, subject, conversation_id: str, history: list) -> dict:
    graph = get_graph()
    state = initial_state(
        question=question,
        subject=subject,
        conversation_id=conversation_id,
        thread_id=conversation_id,
        history=history,
    )
    return await graph.ainvoke(state, config={"configurable": {"thread_id": conversation_id}})


async def _persist(
    session: DbSession,
    *,
    subject,
    conversation_id: str,
    question: str,
    result: dict,
    latency_ms: int,
) -> None:
    """Record the turn. Never raises - a failed audit write must not lose the
    answer the customer already has, but it is logged loudly."""
    try:
        conversation = await session.scalar(
            select(Conversation).where(Conversation.thread_id == conversation_id)
        )
        if conversation is None:
            conversation = Conversation(
                tenant_id=subject.tenant_id,
                thread_id=conversation_id,
                title=question[:200],
            )
            session.add(conversation)
            await session.flush()

        session.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=question,
            )
        )
        session.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.ASSISTANT,
                content=result.get("answer") or "",
                intent=result.get("intent"),
                model=result.get("model"),
                prompt_version=result.get("prompt_version"),
                retrieved_chunk_ids=result.get("retrieved_chunk_ids") or [],
                citations=result.get("citations") or [],
                confidence=result.get("confidence", 0.0),
                abstained=bool(result.get("abstained")),
                latency_ms=latency_ms,
            )
        )
        await session.commit()
    except Exception as exc:
        log.error("Could not persist conversation turn", error=str(exc))


def _to_response(conversation_id: str, result: dict, latency_ms: int) -> ChatResponse:
    intent = result.get("intent")
    return ChatResponse(
        conversation_id=conversation_id,
        answer=result.get("answer") or "",
        citations=result.get("citations") or [],
        intent=intent.value if hasattr(intent, "value") else intent,
        confidence=float(result.get("confidence") or 0.0),
        abstained=bool(result.get("abstained")),
        needs_human=bool(result.get("needs_human")),
        verified=bool(result.get("verified", True)),
        below_threshold=bool(result.get("below_threshold")),
        top_score=float(result.get("top_score") or 0.0),
        query_variants=result.get("query_variants") or [],
        timings_ms=result.get("timings_ms") or {},
        latency_ms=latency_ms,
        retrieved_chunk_ids=result.get("retrieved_chunk_ids") or [],
        context_blocks=result.get("context_blocks") or [],
        broadened=bool(result.get("broadened")),
    )


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest, session: DbSession, subject: Subject) -> ChatResponse:
    conversation_id = payload.conversation_id or str(uuid.uuid4())
    started = time.perf_counter()

    history = await _load_history(session, conversation_id)

    try:
        result = await _run_graph(payload.message, subject, conversation_id, history)
    except Exception as exc:
        log.error("Chat turn failed", error=str(exc))
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"Could not answer: {exc}"
        ) from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    await _persist(
        session,
        subject=subject,
        conversation_id=conversation_id,
        question=payload.message,
        result=result,
        latency_ms=latency_ms,
    )
    return _to_response(conversation_id, result, latency_ms)


@router.post("/stream")
async def chat_stream(
    payload: ChatRequest, session: DbSession, subject: Subject
) -> EventSourceResponse:
    """Server-sent events: a `stage` per graph node, then one `answer`."""
    conversation_id = payload.conversation_id or str(uuid.uuid4())

    async def events() -> AsyncIterator[dict]:
        started = time.perf_counter()
        graph = get_graph()
        state = initial_state(
            question=payload.message,
            subject=subject,
            conversation_id=conversation_id,
            thread_id=conversation_id,
            history=await _load_history(session, conversation_id),
        )
        config = {"configurable": {"thread_id": conversation_id}}
        merged: dict = {}

        try:
            async for update in graph.astream(state, config=config, stream_mode="updates"):
                for node, delta in update.items():
                    if delta:
                        merged.update(delta)
                    if label := STAGE_LABELS.get(node):
                        yield {
                            "event": "stage",
                            "data": json.dumps({"node": node, "label": label}),
                        }
        except Exception as exc:
            log.error("Chat stream failed", error=str(exc))
            yield {"event": "error", "data": json.dumps({"message": str(exc)})}
            return

        latency_ms = int((time.perf_counter() - started) * 1000)
        response = _to_response(conversation_id, merged, latency_ms)
        yield {"event": "answer", "data": response.model_dump_json()}

        await _persist(
            session,
            subject=subject,
            conversation_id=conversation_id,
            question=payload.message,
            result=merged,
            latency_ms=latency_ms,
        )

    return EventSourceResponse(events())


@router.get("/{conversation_id}/history")
async def history(conversation_id: str, session: DbSession, subject: Subject) -> list[dict]:
    """Replay a conversation from Postgres.

    Distinct from the graph checkpoint: this survives a restart and a checkpoint
    purge, because it is the audit record rather than working state.
    """
    conversation = await session.scalar(
        select(Conversation).where(
            Conversation.thread_id == conversation_id,
            Conversation.tenant_id == subject.tenant_id,
        )
    )
    if conversation is None:
        return []

    messages = await session.scalars(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at)
    )
    return [
        {
            "role": m.role.value,
            "content": m.content,
            "citations": m.citations or [],
            "abstained": m.abstained,
            "created_at": m.created_at.isoformat(),
        }
        for m in messages.all()
    ]
