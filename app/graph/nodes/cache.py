"""Semantic cache check and store - Lane A only (Week 11).

Two nodes, inserted around the expensive middle of Lane A:

    route ─▶ cache_check ─┬─ hit  ─▶ END
                           └─ miss ─▶ retrieve ─▶ ... ─▶ verify ─▶ cache_store ─▶ END

A hit skips retrieve, expand, generate and verify entirely - the four steps
`eval/week11/cost_report.py` found responsible for 93% of a turn's cost
(route 11%, retrieve_expand 7%, generate 43%, verify 39%). A miss costs one
extra query embedding (CPU, not a paid call) and falls through unchanged.

Never cache a claim answer: `cache_check_node` only runs on the Lane A path
(see `by_intent` in app/graph/builder.py), and `cache_store_node` only
fires for an answer that is verified and not abstained - gated on where
these nodes sit in the graph, not on a check inside app/cache/semantic.py
itself, which has no way to know what called it.
"""

from __future__ import annotations

import time

from app.cache import semantic
from app.graph.state import ConversationState
from app.logging import get_logger

log = get_logger(__name__)


async def cache_check_node(state: ConversationState) -> dict:
    started = time.perf_counter()
    question = state.get("standalone_question") or state["question"]
    subject = state["subject"]

    hit = await semantic.lookup(question, tenant_id=subject.tenant_id)
    elapsed = {"timings_ms": {"cache_check": int((time.perf_counter() - started) * 1000)}}

    if hit is None:
        return elapsed

    log.info("Serving cached answer", tenant_id=subject.tenant_id)
    return {
        **hit,
        "cache_hit": True,
        **elapsed,
    }


async def cache_store_node(state: ConversationState) -> dict:
    """Store this turn's answer for a future, similar-enough question.

    Runs after verify - a side effect for the *next* turn, not this one, so
    it never touches `answer` or any other field this turn's response uses.
    """
    if state.get("cache_hit"):
        return {}  # already served from the cache - do not re-cache a cache hit
    if not state.get("verified") or state.get("abstained"):
        return {}  # never cache a rejected or abstained answer

    question = state.get("standalone_question") or state["question"]
    subject = state["subject"]
    cacheable = {
        "answer": state.get("answer"),
        "citations": state.get("citations") or [],
        "confidence": state.get("confidence", 0.0),
        "verified": True,
        "abstained": False,
        "retrieved_chunk_ids": state.get("retrieved_chunk_ids") or [],
        "context_blocks": state.get("context_blocks") or [],
        "top_score": state.get("top_score", 0.0),
    }
    await semantic.store(question, tenant_id=subject.tenant_id, result=cacheable)
    return {}
