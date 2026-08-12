"""Cross-encoder reranking and the confidence gate.

The reranking itself runs inside the retrieval pipeline (see ``retrieve.py`` for
why). What lives here is the **gate**: the decision to answer or to abstain.

That decision is the most consequential branch in the graph. Abstention is a
feature, not a failure. A wrong "yes, that's covered" costs a claim payout, a
complaint, and possibly a regulator's attention; "I couldn't find that in your
policy - let me connect you to an agent" costs a support minute
(ARCHITECTURE 10.4).
"""

from __future__ import annotations

from typing import Literal

from app.config import settings
from app.graph.state import ConversationState
from app.logging import get_logger

log = get_logger(__name__)

MAX_RETRIES = 1  # one broadened retry, then abstain

ABSTAIN_MESSAGE = (
    "I couldn't find this in the policy documents I have access to, so I'd rather "
    "not guess. Would you like me to connect you to an agent who can check?"
)

NO_DOCUMENTS_MESSAGE = (
    "I don't have any policy documents indexed yet that match your question, so I "
    "can't answer it accurately. Please upload the relevant policy document, or "
    "I can connect you to an agent."
)


def confidence_gate(
    state: ConversationState,
) -> Literal["generate", "abstain"]:
    """Conditional edge after retrieval.

    The pipeline has already performed its own broadened retry internally, so by
    the time state reaches here the decision is final: either the evidence clears
    the threshold or the honest answer is that we do not know.
    """
    if not state.get("retrieved_chunk_ids"):
        log.info("Gate: no context retrieved")
        return "abstain"

    if state.get("below_threshold"):
        log.info(
            "Gate: below threshold",
            top_score=round(state.get("top_score", 0.0), 4),
            threshold=settings.reranker.score_threshold,
        )
        return "abstain"

    return "generate"


async def abstain_node(state: ConversationState) -> dict:
    """Say so, plainly, and offer a human.

    Recorded with ``abstained=True`` so the eval harness can measure abstention
    correctness - answering when it should have abstained is the most dangerous
    error class in this domain (ARCHITECTURE 11.2).
    """
    has_any_context = bool(state.get("retrieved_chunk_ids"))
    return {
        "answer": ABSTAIN_MESSAGE if has_any_context else NO_DOCUMENTS_MESSAGE,
        "citations": [],
        "abstained": True,
        "needs_human": True,
        "confidence": 0.0,
        "verified": True,  # nothing was claimed, so nothing needs verifying
    }
