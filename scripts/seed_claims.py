"""Seeds synthetic claims and policy holders for the POC.

Creates one policy holder - the demo subject from ``app.deps`` - with two
policies and four claims chosen to exercise every branch the claim lane has:

    CLM-2026-0001  settled            the happy path, with amounts
    CLM-2026-0002  under_review       in flight, no amounts decided yet
    CLM-2026-0003  info_required      needs something from the customer
    CLM-2026-0004  rejected           carries a clause ref, so it triggers the
                                      Lane B -> Lane A hand-off that fetches and
                                      quotes the clause behind the decision

The clause on the rejected claim is 4.11 (Dental Treatment), which really exists
in the seeded corpus - so the hand-off retrieves a real wording rather than
abstaining.

Idempotent: re-running deletes the demo holder's rows and rewrites them, so it is
safe to run repeatedly while iterating.

    python scripts/seed_claims.py

Afterwards, set DEMO_POLICY_HOLDER=1 in .env so the demo subject is linked to
the seeded holder - without it the claim lane correctly refuses.
"""

from __future__ import annotations

import sys
import uuid
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, select  # noqa: E402

from app.core.enums import ClaimStatus, ClaimType, PolicyStatus, UserRole  # noqa: E402
from app.db.models import Claim, ClaimEvent, Policy, PolicyHolder, User  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.deps import _DEMO_USER_ID, DEMO_POLICY_HOLDER_ID  # noqa: E402
from app.logging import configure_logging  # noqa: E402

TENANT = "default"

POLICY_HEALTH_ID = uuid.UUID("00000000-0000-0000-0000-00000000c001")
POLICY_MOTOR_ID = uuid.UUID("00000000-0000-0000-0000-00000000c002")


def _dt(days_ago: int) -> datetime:
    return datetime.now(UTC) - timedelta(days=days_ago)


def _wipe(session) -> None:
    """Remove anything this script previously wrote, so re-runs are clean."""
    claim_ids = list(
        session.scalars(select(Claim.id).where(Claim.policy_holder_id == DEMO_POLICY_HOLDER_ID))
    )
    if claim_ids:
        session.execute(delete(ClaimEvent).where(ClaimEvent.claim_id.in_(claim_ids)))
    session.execute(delete(Claim).where(Claim.policy_holder_id == DEMO_POLICY_HOLDER_ID))
    session.execute(delete(Policy).where(Policy.policy_holder_id == DEMO_POLICY_HOLDER_ID))
    session.execute(delete(PolicyHolder).where(PolicyHolder.id == DEMO_POLICY_HOLDER_ID))
    session.flush()


def _seed(session) -> tuple[int, int]:
    # The demo user may already exist from an earlier run; only create it once.
    if session.get(User, _DEMO_USER_ID) is None:
        session.add(
            User(
                id=_DEMO_USER_ID,
                tenant_id=TENANT,
                email="demo.customer@example.com",
                # Not a credential - there is no login in the POC and this value
                # is never verified against anything. Kept obviously inert so it
                # cannot be mistaken for a real hash.
                hashed_password="!unusable-poc-no-login!",  # noqa: S106 - inert placeholder
                full_name="Demo Customer",
                role=UserRole.CUSTOMER,
                is_active=True,
            )
        )

    session.add(
        PolicyHolder(
            id=DEMO_POLICY_HOLDER_ID,
            tenant_id=TENANT,
            user_id=_DEMO_USER_ID,
            external_ref="PH-DEMO-0001",
            full_name="Demo Customer",
            email="demo.customer@example.com",
            phone="+91 98000 00000",
            date_of_birth=date(1988, 5, 14),
        )
    )

    session.add_all(
        [
            Policy(
                id=POLICY_HEALTH_ID,
                tenant_id=TENANT,
                policy_number="POL-FHO-2026-004417",
                policy_holder_id=DEMO_POLICY_HOLDER_ID,
                insurer="Acme General Insurance",
                product_name="Family Health Optima",
                uin="ACMEHLIP26001V032627",
                sum_insured=Decimal("500000.00"),
                currency="INR",
                inception_date=date(2026, 4, 1),
                expiry_date=date(2027, 3, 31),
                status=PolicyStatus.ACTIVE,
            ),
            Policy(
                id=POLICY_MOTOR_ID,
                tenant_id=TENANT,
                policy_number="POL-MSP-2026-118823",
                policy_holder_id=DEMO_POLICY_HOLDER_ID,
                insurer="Acme General Insurance",
                product_name="Motor Shield Private Car",
                uin="ACMEMOTP26004V011226",
                sum_insured=Decimal("650000.00"),
                currency="INR",
                inception_date=date(2026, 4, 1),
                expiry_date=date(2027, 3, 31),
                status=PolicyStatus.ACTIVE,
            ),
        ]
    )
    session.flush()

    claims = [
        dict(
            claim_number="CLM-2026-0001",
            policy_id=POLICY_HEALTH_ID,
            status=ClaimStatus.SETTLED,
            claim_type=ClaimType.CASHLESS,
            date_of_loss=date(2026, 5, 2),
            reported_at=_dt(100),
            claimed_amount=Decimal("184500.00"),
            approved_amount=Decimal("172300.00"),
            settled_amount=Decimal("172300.00"),
            hospital_name="Sunrise Multispeciality Hospital",
            diagnosis="Acute appendicitis, laparoscopic appendicectomy",
            events=[
                ("submitted", None, ClaimStatus.SUBMITTED,
                 "Cashless pre-authorisation received", 100),
                ("approved", ClaimStatus.UNDER_REVIEW, ClaimStatus.APPROVED,
                 "Pre-authorisation approved for Rs. 172,300", 96),
                ("settled", ClaimStatus.APPROVED, ClaimStatus.SETTLED,
                 "Amount settled directly with the network hospital", 92),
            ],
        ),
        dict(
            claim_number="CLM-2026-0002",
            policy_id=POLICY_HEALTH_ID,
            status=ClaimStatus.UNDER_REVIEW,
            claim_type=ClaimType.REIMBURSEMENT,
            date_of_loss=date(2026, 7, 18),
            reported_at=_dt(24),
            claimed_amount=Decimal("62800.00"),
            hospital_name="Greenfield Hospital",
            diagnosis="Dengue fever with thrombocytopenia",
            events=[
                ("submitted", None, ClaimStatus.SUBMITTED,
                 "Reimbursement claim submitted online", 24),
                ("assigned", ClaimStatus.SUBMITTED, ClaimStatus.UNDER_REVIEW,
                 "Assigned to claims assessor for medical review", 21),
            ],
        ),
        dict(
            claim_number="CLM-2026-0003",
            policy_id=POLICY_HEALTH_ID,
            status=ClaimStatus.INFO_REQUIRED,
            claim_type=ClaimType.REIMBURSEMENT,
            date_of_loss=date(2026, 6, 9),
            reported_at=_dt(56),
            claimed_amount=Decimal("28400.00"),
            hospital_name="City Care Clinic",
            diagnosis="Fracture of left radius following a fall",
            events=[
                ("submitted", None, ClaimStatus.SUBMITTED,
                 "Claim submitted with partial documents", 56),
                ("info_requested", ClaimStatus.UNDER_REVIEW, ClaimStatus.INFO_REQUIRED,
                 "Please upload the discharge summary and the original pharmacy bills", 50),
            ],
        ),
        dict(
            claim_number="CLM-2026-0004",
            policy_id=POLICY_HEALTH_ID,
            status=ClaimStatus.REJECTED,
            claim_type=ClaimType.REIMBURSEMENT,
            date_of_loss=date(2026, 5, 27),
            reported_at=_dt(74),
            claimed_amount=Decimal("41200.00"),
            approved_amount=Decimal("0.00"),
            # Clause 4.11 exists in the seeded v3.2 wording, so the Lane B -> Lane A
            # hand-off retrieves real text instead of abstaining.
            rejection_reason_code="EXCLUDED_TREATMENT",
            rejection_clause_ref="4.11",
            hospital_name="Bright Smile Dental Centre",
            diagnosis="Orthodontic treatment, elective",
            events=[
                ("submitted", None, ClaimStatus.SUBMITTED, "Reimbursement claim submitted", 74),
                ("rejected", ClaimStatus.UNDER_REVIEW, ClaimStatus.REJECTED,
                 "Rejected under the dental treatment exclusion", 68),
            ],
        ),
    ]

    n_events = 0
    for spec in claims:
        events = spec.pop("events")
        claim = Claim(
            id=uuid.uuid4(),
            tenant_id=TENANT,
            policy_holder_id=DEMO_POLICY_HOLDER_ID,
            currency="INR",
            **spec,
        )
        session.add(claim)
        session.flush()
        for event_type, from_status, to_status, note, days_ago in events:
            session.add(
                ClaimEvent(
                    id=uuid.uuid4(),
                    claim_id=claim.id,
                    event_type=event_type,
                    from_status=from_status,
                    to_status=to_status,
                    note=note,
                    actor="claims.system",
                    occurred_at=_dt(days_ago),
                )
            )
            n_events += 1

    return len(claims), n_events


def main() -> int:
    configure_logging(json_output=False)
    with session_scope() as session:
        _wipe(session)
        n_claims, n_events = _seed(session)

    print(f"\nSeeded 1 policy holder, 2 policies, {n_claims} claims, {n_events} claim events.")
    print(f"  policy_holder_id = {DEMO_POLICY_HOLDER_ID}")
    print("\nTo let the demo subject see them, add this to .env and restart the API:")
    print("  DEMO_POLICY_HOLDER=1")
    print("\nLeave it unset to demonstrate the authorization guard refusing instead.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
