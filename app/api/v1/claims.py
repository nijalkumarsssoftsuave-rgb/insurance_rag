"""Claim endpoints. Currently one: the adjuster handover summary.

Claim *status* is answered through the chat graph's Lane B, because a customer
asks for it in a sentence. This endpoint is for the other reader - the adjuster
picking a file up, who wants the whole file in four lines rather than a
conversation. `eval/week6/` measures whether those summaries can be trusted.

Staff only. The summary is written from `ClaimEvent.note`, which is internal
working text: assessors write "recommend rejection under 4.13" in it, and that is
not something to hand a policyholder.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.claims import repository, summary
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


@router.get("/{claim_number}/summary", response_model=ClaimSummary)
async def claim_summary(claim_number: str, session: DbSession, subject: Subject) -> ClaimSummary:
    """Free-text handover summary of one claim, written from its record and notes."""
    if not subject.is_staff:
        # 403, not 404: the caller is authenticated and simply lacks the role.
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Claim summaries are staff-only.")

    claim = await repository.get_claim(session, subject, claim_number)
    if claim is None:
        # `get_claim` returns None both for "no such claim" and "not yours", and
        # this endpoint keeps that symmetry: a 404 that distinguished the two
        # would confirm the existence of other people's claims.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such claim.")

    record = summary.from_claim_view(claim)
    try:
        text = await summary.summarise(record)
    except Exception as exc:
        log.error("Claim summary failed", claim_number=claim_number, error=str(exc))
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Could not write the summary just now."
        ) from exc

    from app.config import settings

    return ClaimSummary(
        claim_number=claim.claim_number,
        summary=text,
        status=claim.status.value,
        note_count=len(record.get("notes") or []),
        model=settings.llm.llm_model,
    )
