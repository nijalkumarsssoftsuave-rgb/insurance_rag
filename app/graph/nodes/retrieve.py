"""Lane A: filter, expand, hybrid search, fuse.

This node runs the whole retrieval pipeline - through reranking and packing -
rather than splitting retrieval and reranking into separate graph nodes.

That is a deliberate departure from the diagram, for a concrete reason: LangGraph
checkpoints state, so anything left in state between nodes must serialize. Passing
``SearchHit`` and ``RerankedHit`` objects across a node boundary would mean either
serializing every candidate's payload into the checkpoint on every turn, or
flattening them to dicts and rebuilding them - both cost more than they buy. The
pipeline keeps the rich objects in one process boundary and writes only
serializable summaries to state. The confidence gate lives in ``rerank.py`` as a
conditional edge over those summaries.
"""

from __future__ import annotations

import time
from datetime import date

from app.core.enums import DocType
from app.graph.state import ConversationState
from app.logging import get_logger
from app.retrieval.filters import RetrievalFilter
from app.retrieval.hybrid import get_pipeline
from app.security.injection import wrap_untrusted

log = get_logger(__name__)


async def retrieve_node(state: ConversationState) -> dict:
    started = time.perf_counter()
    question = state.get("standalone_question") or state["question"]
    route = state.get("route") or {}
    subject = state["subject"]

    # Default to the wording in force TODAY when the question carries no date of
    # loss. "Is dental covered?" means under the policy that applies now.
    #
    # Without this, every version of a policy is an equally valid candidate and
    # the reranker picks on similarity alone - which it did: a dental question
    # was answered from the superseded 2022-2024 wording ("excluded in all
    # circumstances") instead of the current one ("unless necessitated by an
    # accident"). Opposite answers, both well-retrieved. This is exactly the
    # failure ARCHITECTURE 5.3 describes, and a date filter is the control.
    effective_on = route.get("date_of_loss") or date.today()

    retrieval_filter = RetrievalFilter(
        tenant_id=subject.tenant_id,
        product_name=route.get("product_name"),
        date_of_loss=effective_on,
        doc_types=[
            DocType.POLICY_WORDING,
            DocType.ENDORSEMENT,
            DocType.SOP,
            DocType.CIRCULAR,
            DocType.BROCHURE,
        ],
    )

    # Coverage questions force the exclusions and definitions of the matched
    # documents into context. A positively-phrased question ("is dental
    # covered?") will not reliably retrieve its own exclusion by similarity, and
    # an answer that misses it is the failure mode that matters most here.
    is_coverage = bool(route.get("is_coverage_question"))

    outcome = await get_pipeline().retrieve(
        question,
        retrieval_filter=retrieval_filter,
        force_companions=is_coverage,
    )

    context_text = wrap_untrusted(outcome.context.as_pairs()) if outcome.has_context else ""

    log.info(
        "Retrieved",
        blocks=len(outcome.context.blocks),
        top_score=round(outcome.top_score, 4),
        gated=outcome.below_threshold,
        broadened=outcome.broadened,
        coverage=is_coverage,
    )

    return {
        "query_variants": outcome.variants,
        "retrieved_chunk_ids": outcome.context.chunk_ids,
        "context_text": context_text,
        "context_blocks": [b.citation() for b in outcome.context.blocks],
        "top_score": outcome.top_score,
        "below_threshold": outcome.below_threshold,
        "broadened": outcome.broadened,
        "timings_ms": {
            **outcome.timings_ms,
            "retrieve_total": int((time.perf_counter() - started) * 1000),
        },
    }
