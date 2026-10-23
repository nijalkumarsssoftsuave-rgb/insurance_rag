"""Claim endpoints. Currently one: the adjuster handover summary.

Claim *status* is answered through the chat graph's Lane B, because a customer
asks for it in a sentence. This endpoint is for the other reader - the adjuster
picking a file up, who wants the whole file in four lines rather than a
conversation.

Staff only. The summary is written from `ClaimEvent.note`, which is internal
working text: assessors write "recommend rejection under 4.13" in it, and that is
not something to hand a policyholder.

Week 12: this endpoint called `app.claims.summary.summarise()` - Week 6's
original, single-LLM-call writer - through Weeks 7, 8 and 10, which built,
tested and validated three full replacements for exactly this task
(`claim_handover_agent`, then its injection-defended version, then
`claims_squad`) without this endpoint ever being pointed at any of them. The
gap was not theoretical: called directly against CLM-2026-0005 (Week 8's
injection fixture) the old path produced *"The adjuster should update the
status to approved as per the correction noted by Claims Supervisor J.
Patel..."* - the exact attack Week 8 built a defense for, live, in the one
place that defense was never deployed. `claims_squad` is wired in now - Week
10's own measured verdict (ties on quality, wins on cost and tokens, and was
the one of the two architectures that resisted this specific injection
without needing its output-validation backstop to fire) - not
`claim_handover_agent`, which still works but costs more for the same
result. See eval/week12/RESULTS.md for the reproduction, before and after.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.agents import claims_squad
from app.claims import repository
from app.config import settings
from app.deps import DbSession, Subject
from app.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/claims", tags=["claims"])


class ClaimSummary(BaseModel):
    claim_number: str
    summary: str
    status: str
    note_count: int = Field(description="adjuster notes the summary was written from")
    model: str
    output_overridden: bool = Field(
        description="True if the deterministic output-validation guard (Week 8) "
        "had to rewrite the generated text because it contradicted the claim's "
        "observed status - visible here, not just in a log, because an adjuster "
        "reading a corrected summary should know it was corrected."
    )


@router.get("/{claim_number}/summary", response_model=ClaimSummary)
async def claim_summary(claim_number: str, session: DbSession, subject: Subject) -> ClaimSummary:
    """Free-text handover summary of one claim, written by the claims squad
    (manager + status/clause specialists - app/agents/claims_squad.py)."""
    if not subject.is_staff:
        # 403, not 404: the caller is authenticated and simply lacks the role.
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Claim summaries are staff-only.")

    claim = await repository.get_claim(session, subject, claim_number)
    if claim is None:
        # `get_claim` returns None both for "no such claim" and "not yours", and
        # this endpoint keeps that symmetry: a 404 that distinguished the two
        # would confirm the existence of other people's claims.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such claim.")

    try:
        result = await claims_squad.run(session, subject, claim_number)
    except Exception as exc:
        log.error("Claim summary failed", claim_number=claim_number, error=str(exc))
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Could not write the summary just now."
        ) from exc

    note_count = sum(1 for e in claim.events if e.note)
    return ClaimSummary(
        claim_number=claim.claim_number,
        summary=result.answer,
        status=claim.status.value,
        note_count=note_count,
        model=settings.llm.llm_model,
        output_overridden=result.output_overridden,
    )
