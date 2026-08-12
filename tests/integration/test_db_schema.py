"""Relational schema round-trip.

Covers the two things the schema exists to guarantee: that a claim can only be
read by its owner via a single-table predicate, and that a chunk stays resolvable
back to its source document for audit.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.core.enums import (
    AuditAction,
    ChunkKind,
    ClaimStatus,
    ClaimType,
    DocType,
    IngestionStatus,
    UserRole,
)
from app.db.models import (
    AuditLog,
    Chunk,
    Claim,
    ClaimEvent,
    Document,
    DocumentVersion,
    Policy,
    PolicyHolder,
    User,
)

pytestmark = pytest.mark.integration


def _unique(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


@pytest.fixture
def holder_with_claim(db_session):
    """Two holders, one claim. The second holder exists to prove isolation."""
    owner = PolicyHolder(full_name="Asha Rao", external_ref=_unique("ph"))
    stranger = PolicyHolder(full_name="Ravi Menon", external_ref=_unique("ph"))
    db_session.add_all([owner, stranger])
    db_session.flush()

    policy = Policy(
        policy_number=_unique("POL"),
        policy_holder_id=owner.id,
        insurer="Acme General",
        product_name="Family Health Optima",
        sum_insured=Decimal("500000.00"),
        inception_date=date(2024, 4, 1),
        expiry_date=date(2025, 3, 31),
    )
    db_session.add(policy)
    db_session.flush()

    claim = Claim(
        claim_number=_unique("CLM"),
        policy_id=policy.id,
        policy_holder_id=owner.id,
        claim_type=ClaimType.REIMBURSEMENT,
        status=ClaimStatus.UNDER_REVIEW,
        date_of_loss=date(2024, 9, 12),
        reported_at=datetime(2024, 9, 14, tzinfo=UTC),
        claimed_amount=Decimal("48250.00"),
    )
    db_session.add(claim)
    db_session.flush()
    return owner, stranger, claim


def test_claim_is_readable_by_its_owner(db_session, holder_with_claim) -> None:
    owner, _, claim = holder_with_claim
    found = db_session.scalar(
        select(Claim).where(
            Claim.claim_number == claim.claim_number,
            Claim.policy_holder_id == owner.id,
        )
    )
    assert found is not None
    assert found.status is ClaimStatus.UNDER_REVIEW


def test_claim_is_invisible_to_a_stranger(db_session, holder_with_claim) -> None:
    """The ownership predicate from ARCHITECTURE 10.1.

    Knowing a valid claim number must not be enough. This is the query shape the
    claims repository is required to use - no join, no post-filter in Python.
    """
    _, stranger, claim = holder_with_claim
    found = db_session.scalar(
        select(Claim).where(
            Claim.claim_number == claim.claim_number,
            Claim.policy_holder_id == stranger.id,
        )
    )
    assert found is None, "a claim must never resolve for a non-owner"


def test_claim_events_record_status_history(db_session, holder_with_claim) -> None:
    _, _, claim = holder_with_claim
    db_session.add_all(
        [
            ClaimEvent(
                claim_id=claim.id,
                event_type="status_change",
                from_status=ClaimStatus.SUBMITTED,
                to_status=ClaimStatus.UNDER_REVIEW,
                occurred_at=datetime(2024, 9, 15, tzinfo=UTC),
            ),
            ClaimEvent(
                claim_id=claim.id,
                event_type="document_requested",
                note="Discharge summary pending",
                occurred_at=datetime(2024, 9, 18, tzinfo=UTC),
            ),
        ]
    )
    db_session.flush()
    db_session.refresh(claim)
    assert len(claim.events) == 2
    assert claim.events[0].to_status is ClaimStatus.UNDER_REVIEW


def test_document_version_chunk_chain_is_resolvable(db_session) -> None:
    """A citation must resolve to its source months later (ARCHITECTURE 7.3)."""
    document = Document(
        filename="family-health-optima-v3.2.pdf",
        mime_type="application/pdf",
        size_bytes=1_240_000,
        insurer="Acme General",
        product_name="Family Health Optima",
        doc_type=DocType.POLICY_WORDING,
        effective_from=date(2024, 4, 1),
        effective_to=date(2025, 3, 31),
    )
    db_session.add(document)
    db_session.flush()

    version = DocumentVersion(
        document_id=document.id,
        version_no=1,
        content_hash=uuid.uuid4().hex * 2,  # 64 chars, stands in for a sha256
        storage_key="raw/family-health-optima-v3.2.pdf",
        status=IngestionStatus.COMPLETED,
        parser="docling",
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
    )
    db_session.add(version)
    db_session.flush()

    parent = Chunk(
        document_id=document.id,
        document_version_id=version.id,
        kind=ChunkKind.PARENT,
        chunk_index=0,
        section_path="Section 4 > Exclusions",
        text="Section 4 - Exclusions. The following are not covered ...",
        text_hash=uuid.uuid4().hex * 2,
    )
    db_session.add(parent)
    db_session.flush()

    child = Chunk(
        document_id=document.id,
        document_version_id=version.id,
        parent_id=parent.id,
        kind=ChunkKind.CHILD,
        chunk_index=1,
        section_path="Section 4 > Exclusions > 4.11 Dental",
        text="Dental treatment is excluded unless necessitated by accident.",
        embedded_text="[Acme General | Family Health Optima | Section 4 > Exclusions > 4.11] "
        "Dental treatment is excluded unless necessitated by accident.",
        text_hash=uuid.uuid4().hex * 2,
        embedding_model="BAAI/bge-m3",
    )
    db_session.add(child)
    db_session.flush()

    resolved = db_session.get(Chunk, child.id)
    assert resolved is not None
    assert resolved.parent_id == parent.id
    assert resolved.document_version.document.product_name == "Family Health Optima"
    # The prefix is embedded but must not be what a citation displays.
    assert resolved.embedded_text.endswith(resolved.text)


def test_content_hash_is_globally_unique(db_session) -> None:
    """Re-uploading identical bytes must be rejected, not silently re-ingested."""
    document = Document(
        filename="dup.pdf", mime_type="application/pdf", size_bytes=10, doc_type=DocType.SOP
    )
    db_session.add(document)
    db_session.flush()

    shared_hash = uuid.uuid4().hex * 2
    db_session.add(
        DocumentVersion(
            document_id=document.id,
            version_no=1,
            content_hash=shared_hash,
            storage_key="raw/a.pdf",
        )
    )
    db_session.flush()

    db_session.add(
        DocumentVersion(
            document_id=document.id,
            version_no=2,
            content_hash=shared_hash,
            storage_key="raw/b.pdf",
        )
    )
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_audit_log_records_denied_access(db_session) -> None:
    """Denials are recorded too - a run of them is what enumeration looks like."""
    actor = User(
        email=f"{uuid.uuid4().hex[:8]}@example.com",
        hashed_password="x",
        role=UserRole.CUSTOMER,
    )
    db_session.add(actor)
    db_session.flush()

    db_session.add(
        AuditLog(
            actor_user_id=actor.id,
            action=AuditAction.CLAIM_ACCESS_DENIED,
            resource_type="claim",
            resource_id="CLM-99999",
            allowed=False,
            detail={"reason": "not_owner"},
        )
    )
    db_session.flush()

    entry = db_session.scalar(
        select(AuditLog).where(AuditLog.actor_user_id == actor.id, AuditLog.allowed.is_(False))
    )
    assert entry is not None
    assert entry.action is AuditAction.CLAIM_ACCESS_DENIED
    assert entry.detail["reason"] == "not_owner"
