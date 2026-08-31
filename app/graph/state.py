"""ConversationState TypedDict - the single source of truth for a turn.

Everything a node needs and everything the audit trail records travels here. Two
fields deserve attention:

* ``subject`` is an ``AuthSubject`` built from a verified JWT before the graph
  runs. No node may construct or modify one. It is the reason a claim lookup
  cannot be steered by the model (ARCHITECTURE 10.1).
* ``retrieved_chunk_ids`` and ``citations`` are not debug output. They are the
  evidence a grievance officer needs to reconstruct why the assistant said what
  it said (ARCHITECTURE 10.4).
"""

from __future__ import annotations

import operator
from datetime import date
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages

from app.core.enums import Intent
from app.security.authz import AuthSubject


class RouteDecision(TypedDict, total=False):
    """Router output. Entities are extracted, never inferred."""

    intent: Intent
    claim_number: str | None
    policy_number: str | None
    product_name: str | None
    date_of_loss: date | None
    is_coverage_question: bool
    confidence: float


class ConversationState(TypedDict, total=False):
    # ── identity, set before the graph runs ──────────────────────────────
    subject: AuthSubject
    conversation_id: str
    thread_id: str

    # ── input ────────────────────────────────────────────────────────────
    messages: Annotated[list, add_messages]
    question: str  # raw user text
    standalone_question: str  # after history-aware condensation
    masked_question: str  # PII-masked, for logs and traces

    # ── guard ────────────────────────────────────────────────────────────
    injection_severity: str
    pii_mapping: dict[str, str]
    blocked: bool
    block_reason: str | None

    # ── routing ──────────────────────────────────────────────────────────
    route: RouteDecision
    intent: Intent

    # ── lane A: retrieval ────────────────────────────────────────────────
    query_variants: list[str]
    retrieved_chunk_ids: list[str]
    context_text: str
    context_blocks: list[dict[str, Any]]
    top_score: float
    below_threshold: bool
    broadened: bool

    # ── lane B: claims ───────────────────────────────────────────────────
    claim_found: bool
    claim_answer: str | None
    clause_followup: str | None

    # ── output ───────────────────────────────────────────────────────────
    answer: str
    citations: list[dict[str, Any]]
    confidence: float
    abstained: bool
    needs_human: bool
    verified: bool
    verification_notes: list[str]

    # ── bookkeeping ──────────────────────────────────────────────────────
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    # Accumulated across nodes rather than overwritten, so one turn yields one
    # complete timing breakdown.
    timings_ms: Annotated[dict[str, int], operator.or_]
    retry_count: int
    error: str | None


def initial_state(
    *,
    question: str,
    subject: AuthSubject,
    conversation_id: str,
    thread_id: str,
    history: list | None = None,
) -> ConversationState:
    """Build the state for one turn.

    ``history`` is the prior turns of this conversation, oldest first. It must be
    supplied by the caller: no node writes to the ``messages`` channel, so leaving
    it empty makes every turn look like the first one and ``condense_node`` returns
    the raw question unchanged - which is how "And what about ICU?" ended up being
    classified out of scope instead of resolving against the previous answer.
    """
    return ConversationState(
        subject=subject,
        conversation_id=conversation_id,
        thread_id=thread_id,
        question=question,
        messages=history or [],
        blocked=False,
        abstained=False,
        needs_human=False,
        verified=False,
        verification_notes=[],
        retrieved_chunk_ids=[],
        citations=[],
        timings_ms={},
        retry_count=0,
        confidence=0.0,
    )
