"""Claim status business logic and answer templating.

The answer is **templated, not generated**. Every figure a customer sees - status,
amounts, dates - is rendered from the database row by Python. The LLM is never
given the opportunity to paraphrase a settlement amount, because a hallucinated
number here is a financial dispute, not a quality metric.

The LLM's only role in this lane is an optional tone pass over already-rendered
facts, and a hand-off to Lane A when a rejection needs its clause explained.
"""

from __future__ import annotations

from decimal import Decimal

from app.claims.repository import ClaimView
from app.core.enums import ClaimStatus

_STATUS_TEXT: dict[ClaimStatus, str] = {
    ClaimStatus.SUBMITTED: "has been received and is queued for review",
    ClaimStatus.UNDER_REVIEW: "is currently under review",
    ClaimStatus.INFO_REQUIRED: "is on hold pending additional information from you",
    ClaimStatus.APPROVED: "has been approved",
    ClaimStatus.PARTIALLY_APPROVED: "has been partially approved",
    ClaimStatus.REJECTED: "has been rejected",
    ClaimStatus.SETTLED: "has been settled",
    ClaimStatus.CLOSED: "is closed",
    ClaimStatus.WITHDRAWN: "was withdrawn",
}

# What the customer should do next. Absent for terminal states.
_NEXT_STEP: dict[ClaimStatus, str] = {
    ClaimStatus.INFO_REQUIRED: (
        "Please upload the requested documents so the review can continue."
    ),
    ClaimStatus.APPROVED: "Payment is usually released within 7-10 working days.",
    ClaimStatus.PARTIALLY_APPROVED: (
        "You can request a review of the unapproved portion within 30 days."
    ),
    ClaimStatus.REJECTED: (
        "If you believe this is incorrect, you can request a reconsideration or "
        "escalate to the grievance officer."
    ),
}


def _money(amount: Decimal | None, currency: str) -> str:
    if amount is None:
        return "not yet determined"
    symbol = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}.get(currency, f"{currency} ")
    return f"{symbol}{amount:,.2f}"


def render_status(claim: ClaimView) -> str:
    """Deterministic status answer. No model involved."""
    lines = [
        f"Claim **{claim.claim_number}** ({claim.product_name}, policy "
        f"{claim.policy_number}) {_STATUS_TEXT[claim.status]}."
    ]

    if claim.claimed_amount is not None:
        lines.append(f"- Amount claimed: {_money(claim.claimed_amount, claim.currency)}")
    if claim.approved_amount is not None:
        lines.append(f"- Amount approved: {_money(claim.approved_amount, claim.currency)}")
    if claim.settled_amount is not None:
        lines.append(f"- Amount settled: {_money(claim.settled_amount, claim.currency)}")

    lines.append(f"- Date of loss: {claim.date_of_loss:%d %b %Y}")
    lines.append(f"- Last updated: {claim.updated_at:%d %b %Y}")

    if claim.status is ClaimStatus.REJECTED and claim.rejection_reason_code:
        reason = claim.rejection_reason_code.replace("_", " ").lower()
        clause = f" (clause {claim.rejection_clause_ref})" if claim.rejection_clause_ref else ""
        lines.append(f"- Reason recorded: {reason}{clause}")

    if latest := _latest_note(claim):
        lines.append(f"- Latest update: {latest}")

    if step := _NEXT_STEP.get(claim.status):
        lines.append(f"\n{step}")

    return "\n".join(lines)


def _latest_note(claim: ClaimView) -> str | None:
    for event in sorted(claim.events, key=lambda e: e.occurred_at, reverse=True):
        if event.note:
            return f"{event.note} ({event.occurred_at:%d %b %Y})"
    return None


def render_claim_list(claims: list[ClaimView]) -> str:
    if not claims:
        return "I could not find any claims linked to your policies."
    lines = [f"You have {len(claims)} claim(s) on record:"]
    lines += [
        f"- **{c.claim_number}** ({c.product_name}) - {c.status.value.replace('_', ' ')}, "
        f"loss dated {c.date_of_loss:%d %b %Y}"
        for c in claims
    ]
    return "\n".join(lines)


def not_found_message(claim_number: str | None) -> str:
    """Identical for 'does not exist' and 'belongs to someone else'.

    Any wording that distinguishes the two turns the assistant into an oracle for
    valid claim numbers.
    """
    if claim_number:
        return (
            f"I could not find a claim **{claim_number}** on your policies. Please check "
            "the number, or contact support if you believe it should be there."
        )
    return (
        "I could not identify which claim you mean. Please share the claim number "
        "(for example CLM-12345)."
    )


def clause_followup_query(claim: ClaimView) -> str | None:
    """The Lane B to Lane A hand-off.

    A rejection reason code is not an explanation. This turns it into a document
    question so the clause behind the decision can be retrieved and quoted.
    """
    if not claim.needs_clause_explanation:
        return None
    return (
        f"What does clause {claim.rejection_clause_ref} of the {claim.product_name} "
        f"policy say, and what does '{(claim.rejection_reason_code or '').replace('_', ' ')}' mean?"
    )
