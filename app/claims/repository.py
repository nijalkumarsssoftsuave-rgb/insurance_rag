"""Parameterized, ownership-enforcing claim queries.

Every function here takes an ``AuthSubject`` as its first argument and folds it
into the WHERE clause. There is no code path that reads a claim without a subject,
and no function accepts a holder id as a plain parameter - that would let a caller
pass one the model produced.

Not-found and forbidden are indistinguishable to the caller. A customer probing
claim numbers learns nothing about which ones exist (ARCHITECTURE 10.1).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.enums import ClaimStatus, ClaimType
from app.db.models import Claim, ClaimEvent, Policy
from app.security.authz import AuthSubject


@dataclass(slots=True)
class ClaimEventView:
    event_type: str
    note: str | None
    to_status: ClaimStatus | None
    occurred_at: datetime


@dataclass(slots=True)
class ClaimView:
    """A safe projection.

    Clinical detail (``diagnosis``, ``hospital_name``) is staff-only. A customer
    asking about their own claim does not need it echoed back through an LLM, and
    keeping it out of the context is cheaper than trusting a prompt to withhold it.
    """

    claim_number: str
    status: ClaimStatus
    claim_type: ClaimType
    date_of_loss: date
    reported_at: datetime
    policy_number: str
    product_name: str
    insurer: str
    claimed_amount: Decimal | None
    approved_amount: Decimal | None
    settled_amount: Decimal | None
    currency: str
    rejection_reason_code: str | None
    rejection_clause_ref: str | None
    updated_at: datetime
    events: list[ClaimEventView]
    hospital_name: str | None = None
    diagnosis: str | None = None

    @property
    def is_rejected(self) -> bool:
        return self.status is ClaimStatus.REJECTED

    @property
    def needs_clause_explanation(self) -> bool:
        """Rejected with a clause reference - the hand-off from Lane B to Lane A."""
        return self.is_rejected and bool(self.rejection_clause_ref)


def _scoped(subject: AuthSubject) -> Select[tuple[Claim]]:
    """Base query with the subject's scope already applied.

    Every read starts here. Tenant is always constrained; a customer is
    additionally constrained to their own holder id.
    """
    stmt = (
        select(Claim)
        .join(Policy, Claim.policy_id == Policy.id)
        .where(Claim.tenant_id == subject.tenant_id)
        .options(selectinload(Claim.events), selectinload(Claim.policy))
    )
    if not subject.can_read_any_claim:
        stmt = stmt.where(Claim.policy_holder_id == subject.require_policy_holder())
    return stmt


def _to_view(claim: Claim, *, include_clinical: bool) -> ClaimView:
    return ClaimView(
        claim_number=claim.claim_number,
        status=claim.status,
        claim_type=claim.claim_type,
        date_of_loss=claim.date_of_loss,
        reported_at=claim.reported_at,
        policy_number=claim.policy.policy_number,
        product_name=claim.policy.product_name,
        insurer=claim.policy.insurer,
        claimed_amount=claim.claimed_amount,
        approved_amount=claim.approved_amount,
        settled_amount=claim.settled_amount,
        currency=claim.currency,
        rejection_reason_code=claim.rejection_reason_code,
        rejection_clause_ref=claim.rejection_clause_ref,
        updated_at=claim.updated_at,
        events=[
            ClaimEventView(
                event_type=e.event_type,
                note=e.note,
                to_status=e.to_status,
                occurred_at=e.occurred_at,
            )
            for e in claim.events
        ],
        hospital_name=claim.hospital_name if include_clinical else None,
        diagnosis=claim.diagnosis if include_clinical else None,
    )


async def get_claim(
    session: AsyncSession, subject: AuthSubject, claim_number: str
) -> ClaimView | None:
    """Fetch one claim by number, scoped to the subject.

    Returns None both when the claim does not exist and when it belongs to
    somebody else. That symmetry is deliberate.
    """
    if not claim_number or not claim_number.strip():
        return None

    claim = await session.scalar(_scoped(subject).where(Claim.claim_number == claim_number.strip()))
    if claim is None:
        return None
    return _to_view(claim, include_clinical=subject.is_staff)


async def list_claims(
    session: AsyncSession, subject: AuthSubject, *, limit: int = 20
) -> list[ClaimView]:
    """The subject's claims, newest first. Used for 'what claims do I have open'."""
    claims = await session.scalars(
        _scoped(subject).order_by(Claim.reported_at.desc()).limit(min(limit, 100))
    )
    return [_to_view(c, include_clinical=subject.is_staff) for c in claims.all()]


async def get_claim_by_id(
    session: AsyncSession, subject: AuthSubject, claim_id: uuid.UUID
) -> ClaimView | None:
    claim = await session.scalar(_scoped(subject).where(Claim.id == claim_id))
    if claim is None:
        return None
    return _to_view(claim, include_clinical=subject.is_staff)


async def claim_exists_anywhere(session: AsyncSession, claim_number: str) -> bool:
    """Unscoped existence check - for auditing only.

    Lets the audit log distinguish "probed a real claim they do not own" from
    "typed a claim number that does not exist". Never call this on a request path
    that returns anything to the user: the difference is exactly what an attacker
    wants to learn.
    """
    found = await session.scalar(
        select(Claim.id).where(Claim.claim_number == claim_number.strip()).limit(1)
    )
    return found is not None


async def recent_events(
    session: AsyncSession, subject: AuthSubject, claim_number: str, *, limit: int = 10
) -> list[ClaimEventView]:
    claim = await session.scalar(_scoped(subject).where(Claim.claim_number == claim_number.strip()))
    if claim is None:
        return []
    events = await session.scalars(
        select(ClaimEvent)
        .where(ClaimEvent.claim_id == claim.id)
        .order_by(ClaimEvent.occurred_at.desc())
        .limit(limit)
    )
    return [
        ClaimEventView(
            event_type=e.event_type,
            note=e.note,
            to_status=e.to_status,
            occurred_at=e.occurred_at,
        )
        for e in events.all()
    ]
