"""Assembles nodes and conditional edges into the compiled graph.

Shape:

    guard ──▶ condense ──▶ route ──┬─▶ claim_lookup ─┬─▶ clause_handoff ─▶ retrieve
                 │                 │                 └─▶ END
                 │                 ├─▶ retrieve ─▶ [gate] ─┬─▶ generate ─▶ verify ─▶ END
                 │                 │                       └─▶ abstain ─▶ END
                 │                 └─▶ scope_reply ─▶ END
                 └─(blocked)──────────────────────────────────▶ END

Lane B can fall through into Lane A: a rejected claim carries a clause reference,
and explaining that clause is a document question.
"""

from __future__ import annotations

from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.core.enums import Intent
from app.graph.nodes.claim_lookup import (
    claim_lookup_node,
    clause_handoff_node,
    needs_clause_explanation,
)
from app.graph.nodes.condense import condense_node
from app.graph.nodes.generate import generate_node
from app.graph.nodes.guard import guard_node
from app.graph.nodes.rerank import abstain_node, confidence_gate
from app.graph.nodes.retrieve import retrieve_node
from app.graph.nodes.route import route_node
from app.graph.nodes.verify import verify_node
from app.graph.state import ConversationState
from app.logging import get_logger

log = get_logger(__name__)

SCOPE_MESSAGE = (
    "I can help with questions about your insurance policies and the status of "
    "your claims. What would you like to know?"
)


async def scope_reply_node(state: ConversationState) -> dict:
    """Smalltalk and out-of-scope. Deliberately does not retrieve.

    Running the pipeline for "hello" would spend an embedding pass and a
    cross-encoder pass to answer a greeting.
    """
    return {
        "answer": SCOPE_MESSAGE,
        "citations": [],
        "abstained": False,
        "verified": True,
        "confidence": 1.0,
    }


def after_guard(state: ConversationState) -> Literal["condense", "end"]:
    return "end" if state.get("blocked") else "condense"


def by_intent(
    state: ConversationState,
) -> Literal["claim_lookup", "retrieve", "scope_reply"]:
    intent = state.get("intent")
    if intent is Intent.CLAIM_STATUS:
        return "claim_lookup"
    if intent in (Intent.SMALLTALK, Intent.OUT_OF_SCOPE):
        return "scope_reply"
    # POLICY_QA and CLAIM_INTAKE both need document grounding. Intake is not yet
    # a guided flow; until it is, answering the procedure from the SOP is right.
    return "retrieve"


def build_graph(checkpointer=None) -> StateGraph:
    """Compile the conversation graph.

    Args:
        checkpointer: a LangGraph checkpointer. Postgres in production, so a
            conversation survives a Streamlit restart; ``MemorySaver`` by default
            for tests and local runs.
    """
    graph = StateGraph(ConversationState)

    graph.add_node("guard", guard_node)
    graph.add_node("condense", condense_node)
    graph.add_node("route", route_node)
    graph.add_node("claim_lookup", claim_lookup_node)
    graph.add_node("clause_handoff", clause_handoff_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)
    graph.add_node("abstain", abstain_node)
    graph.add_node("scope_reply", scope_reply_node)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", after_guard, {"condense": "condense", "end": END})
    graph.add_edge("condense", "route")
    graph.add_conditional_edges(
        "route",
        by_intent,
        {
            "claim_lookup": "claim_lookup",
            "retrieve": "retrieve",
            "scope_reply": "scope_reply",
        },
    )

    # Lane B: a rejection with a clause reference falls through into Lane A.
    graph.add_conditional_edges(
        "claim_lookup",
        needs_clause_explanation,
        {"explain": "clause_handoff", "done": END},
    )
    graph.add_edge("clause_handoff", "retrieve")

    # Lane A: the confidence gate is the only path to generation.
    graph.add_conditional_edges(
        "retrieve", confidence_gate, {"generate": "generate", "abstain": "abstain"}
    )
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)
    graph.add_edge("abstain", END)
    graph.add_edge("scope_reply", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


_graph = None


def get_graph():
    """Process-wide compiled graph. Compilation is not free; the graph is stateless."""
    global _graph
    if _graph is None:
        _graph = build_graph()
        log.info("Conversation graph compiled")
    return _graph


async def build_postgres_graph(dsn: str):
    """Graph backed by a Postgres checkpointer.

    Call ``await checkpointer.setup()`` once per deployment - it creates the
    checkpoint tables, which are separate from our own ``conversations`` and
    ``messages`` (those exist for audit and must outlive a checkpoint purge).
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    # LangGraph's saver speaks raw psycopg, not the SQLAlchemy dialect prefix.
    raw_dsn = dsn.replace("postgresql+psycopg://", "postgresql://")
    async with AsyncPostgresSaver.from_conn_string(raw_dsn) as checkpointer:
        await checkpointer.setup()
        yield build_graph(checkpointer)
