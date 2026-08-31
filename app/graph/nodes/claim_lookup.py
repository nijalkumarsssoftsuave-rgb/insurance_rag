"""Lane B: authorized claim status via the claims repository.

The critical design choice here is what is **absent**: claims are not exposed to
the LLM as callable tools.

The router classifies intent and this node performs the lookup directly, with the
subject taken from verified state. An LLM-callable claim tool would mean the model
decides when to read claim data and with what arguments - and the model reads
attacker-controlled text from both the user and retrieved documents. Removing the
tool removes the entire class of attack rather than trying to prompt around it
(ARCHITECTURE 10.1, 10.2).

The answer itself is templated in ``app.claims.service``. No figure a customer
sees passes through a language model.
"""

from __future__ import annotations

import time

from app.claims import repository, service
from app.core.enums import AuditAction
from app.db.session import async_session_scope
from app.graph.state import ConversationState
from app.logging import get_logger
from app.security.audit import record
from app.security.authz import AuthorizationError

log = get_logger(__name__)

# Deliberately says nothing about whether claims exist for anyone else, for the
# same reason `service.not_found_message` is identical for "no such claim" and
# "not yours".
NO_POLICY_HOLDER_MESSAGE = (
    "I can't see any policies linked to this account, so there are no claims for me "
    "to look up. If you believe your policy should be linked, please contact support "
    "and they can connect it for you."
)


async def claim_lookup_node(state: ConversationState) -> dict:
    started = time.perf_counter()
    subject = state["subject"]
    route = state.get("route") or {}
    claim_number = route.get("claim_number")

    # An unlinked account is an ordinary outcome, not a fault. `require_policy_holder`
    # raises by design so no caller can reach an unscoped query, but that exception
    # used to travel all the way out of the graph and surface as HTTP 500 - so the
    # customer saw a crash where the correct response was "I can't see any policies
    # on this account". Catch it at the lane boundary and answer normally; the
    # scoping guarantee is unchanged because we still never run the query.
    # Mirrors the condition in `repository._scoped`: staff are scoped by tenant and
    # never need a holder id, so only a non-staff subject is checked here.
    if not subject.can_read_any_claim:
        try:
            subject.require_policy_holder()
        except AuthorizationError as exc:
            log.info("Claim lookup refused - subject has no policy holder", reason=str(exc))
            return _done(answer=NO_POLICY_HOLDER_MESSAGE, found=False, started=started)

    async with async_session_scope() as session:
        # No claim number: list what this subject actually has, rather than
        # asking them to guess a number we could look up ourselves.
        if not claim_number:
            claims = await repository.list_claims(session, subject, limit=10)
            await record(
                session,
                subject=subject,
                action=AuditAction.CLAIM_VIEWED,
                resource_type="claim_list",
                resource_id=None,
                allowed=True,
                detail={"count": len(claims)},
            )
            return _done(
                answer=service.render_claim_list(claims),
                found=bool(claims),
                started=started,
            )

        claim = await repository.get_claim(session, subject, claim_number)

        if claim is None:
            # Distinguish "does not exist" from "not yours" in the AUDIT LOG only.
            # The user-facing message is identical either way, or the assistant
            # becomes an oracle for valid claim numbers.
            probed_real = await repository.claim_exists_anywhere(session, claim_number)
            await record(
                session,
                subject=subject,
                action=AuditAction.CLAIM_ACCESS_DENIED,
                resource_type="claim",
                resource_id=claim_number,
                allowed=False,
                detail={"reason": "not_owner" if probed_real else "not_found"},
            )
            if probed_real:
                log.warning(
                    "Subject probed a claim they do not own",
                    user_id=str(subject.user_id),
                    claim_number=claim_number,
                )
            return _done(
                answer=service.not_found_message(claim_number), found=False, started=started
            )

        await record(
            session,
            subject=subject,
            action=AuditAction.CLAIM_VIEWED,
            resource_type="claim",
            resource_id=claim.claim_number,
            allowed=True,
            detail={"status": claim.status.value},
        )

    return _done(
        answer=service.render_status(claim),
        found=True,
        started=started,
        followup=service.clause_followup_query(claim),
    )


def _done(*, answer: str, found: bool, started: float, followup: str | None = None) -> dict:
    return {
        "claim_found": found,
        "claim_answer": answer,
        "answer": answer,
        "clause_followup": followup,
        "citations": [],
        "abstained": False,
        # Rendered from database rows, so groundedness is structural rather than
        # something a checker needs to establish.
        "verified": True,
        "confidence": 1.0 if found else 0.0,
        "needs_human": not found,
        "timings_ms": {"claim_lookup": int((time.perf_counter() - started) * 1000)},
    }


def needs_clause_explanation(state: ConversationState) -> str:
    """Conditional edge: rejected claims hand off to Lane A for the clause.

    A reason code is not an explanation. "PED_WAITING_PERIOD" means nothing to a
    customer until the clause behind it is quoted.
    """
    return "explain" if state.get("clause_followup") else "done"


async def clause_handoff_node(state: ConversationState) -> dict:
    """Rewrite the question so Lane A retrieves the clause behind a rejection."""
    followup = state.get("clause_followup")
    if not followup:
        return {}
    return {
        "standalone_question": followup,
        "claim_answer": state.get("answer"),
    }
