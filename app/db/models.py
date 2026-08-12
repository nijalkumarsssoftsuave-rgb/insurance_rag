"""ORM models: users, documents, chunks, claims, feedback, audit.

Two design notes that are load-bearing rather than stylistic:

* ``Claim.policy_holder_id`` is denormalized from ``Policy`` on purpose. Ownership
  checks must be a single-table predicate so the authorization guard cannot be
  accidentally bypassed by a join that someone later rewrites (ARCHITECTURE 10.1).
* ``Chunk`` mirrors what is written to Qdrant. When a customer disputes an answer
  six months later, the cited chunk text must still be resolvable even if the
  vector index has been rebuilt since (ARCHITECTURE 7.3).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import (
    AuditAction,
    ChunkKind,
    ClaimStatus,
    ClaimType,
    DocType,
    FeedbackRating,
    IngestionStatus,
    Intent,
    MessageRole,
    PolicyStatus,
    UserRole,
)
from app.db.base import Base, TimestampMixin, UUIDPrimaryKey

DEFAULT_TENANT = "default"


def _enum(enum_cls: type, name: str) -> SAEnum:
    """Store the enum's *values*, not its member names, so the DB is readable."""
    return SAEnum(
        enum_cls,
        name=name,
        native_enum=True,
        values_callable=lambda e: [m.value for m in e],
        validate_strings=True,
    )


# ───────────────────────────────────────────────────────────── identity


class User(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255))
    role: Mapped[UserRole] = mapped_column(
        _enum(UserRole, "user_role"), nullable=False, default=UserRole.CUSTOMER
    )
    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, default=DEFAULT_TENANT)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    policy_holder: Mapped[PolicyHolder | None] = relationship(back_populates="user")

    __table_args__ = (UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),)


class PolicyHolder(UUIDPrimaryKey, TimestampMixin, Base):
    """The insured party. Separate from ``User`` because not every holder has a login."""

    __tablename__ = "policy_holders"

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, default=DEFAULT_TENANT)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), unique=True
    )
    external_ref: Mapped[str | None] = mapped_column(String(64))
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(320))
    phone: Mapped[str | None] = mapped_column(String(32))
    date_of_birth: Mapped[date | None] = mapped_column(Date)

    user: Mapped[User | None] = relationship(back_populates="policy_holder")
    policies: Mapped[list[Policy]] = relationship(back_populates="policy_holder")
    claims: Mapped[list[Claim]] = relationship(back_populates="policy_holder")

    __table_args__ = (
        UniqueConstraint("tenant_id", "external_ref", name="uq_policy_holders_tenant_ref"),
        Index("ix_policy_holders_tenant", "tenant_id"),
    )


# ───────────────────────────────────────────────────────── policies & claims


class Policy(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "policies"

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, default=DEFAULT_TENANT)
    policy_number: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_holder_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("policy_holders.id", ondelete="CASCADE"), nullable=False
    )
    insurer: Mapped[str] = mapped_column(String(255), nullable=False)
    product_name: Mapped[str] = mapped_column(String(255), nullable=False)
    uin: Mapped[str | None] = mapped_column(String(64))
    sum_insured: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    inception_date: Mapped[date] = mapped_column(Date, nullable=False)
    expiry_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[PolicyStatus] = mapped_column(
        _enum(PolicyStatus, "policy_status"), nullable=False, default=PolicyStatus.ACTIVE
    )

    policy_holder: Mapped[PolicyHolder] = relationship(back_populates="policies")
    claims: Mapped[list[Claim]] = relationship(back_populates="policy")

    __table_args__ = (
        UniqueConstraint("tenant_id", "policy_number", name="uq_policies_tenant_number"),
        Index("ix_policies_holder", "policy_holder_id"),
        Index("ix_policies_product", "tenant_id", "product_name"),
    )


class Claim(UUIDPrimaryKey, TimestampMixin, Base):
    __tablename__ = "claims"

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, default=DEFAULT_TENANT)
    claim_number: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("policies.id", ondelete="RESTRICT"), nullable=False
    )
    # Denormalized from Policy. See the module docstring - this is the column the
    # authorization guard filters on, and it must not require a join.
    policy_holder_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("policy_holders.id", ondelete="RESTRICT"), nullable=False
    )

    status: Mapped[ClaimStatus] = mapped_column(
        _enum(ClaimStatus, "claim_status"), nullable=False, default=ClaimStatus.SUBMITTED
    )
    claim_type: Mapped[ClaimType] = mapped_column(_enum(ClaimType, "claim_type"), nullable=False)

    # The date of loss decides which policy version applies (ARCHITECTURE 5.3).
    date_of_loss: Mapped[date] = mapped_column(Date, nullable=False)
    reported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    claimed_amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    approved_amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    settled_amount: Mapped[Decimal | None] = mapped_column(Numeric(14, 2))
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")

    # Populated on rejection. `rejection_clause_ref` is what lets the claim lane
    # hand off to the document lane to explain *why* (ARCHITECTURE 1).
    rejection_reason_code: Mapped[str | None] = mapped_column(String(64))
    rejection_clause_ref: Mapped[str | None] = mapped_column(String(255))

    hospital_name: Mapped[str | None] = mapped_column(String(255))
    diagnosis: Mapped[str | None] = mapped_column(Text)
    assigned_to: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    policy: Mapped[Policy] = relationship(back_populates="claims")
    policy_holder: Mapped[PolicyHolder] = relationship(back_populates="claims")
    events: Mapped[list[ClaimEvent]] = relationship(
        back_populates="claim", cascade="all, delete-orphan", order_by="ClaimEvent.occurred_at"
    )

    __table_args__ = (
        UniqueConstraint("tenant_id", "claim_number", name="uq_claims_tenant_number"),
        # The composite index the ownership check hits on every lookup.
        Index("ix_claims_owner_lookup", "tenant_id", "claim_number", "policy_holder_id"),
        Index("ix_claims_holder", "policy_holder_id"),
        Index("ix_claims_status", "tenant_id", "status"),
        CheckConstraint("approved_amount IS NULL OR approved_amount >= 0", name="approved_nonneg"),
    )


class ClaimEvent(UUIDPrimaryKey, Base):
    """Status history. Answers 'what happened to my claim and when'."""

    __tablename__ = "claim_events"

    claim_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("claims.id", ondelete="CASCADE"), nullable=False
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    from_status: Mapped[ClaimStatus | None] = mapped_column(_enum(ClaimStatus, "claim_status"))
    to_status: Mapped[ClaimStatus | None] = mapped_column(_enum(ClaimStatus, "claim_status"))
    note: Mapped[str | None] = mapped_column(Text)
    actor: Mapped[str | None] = mapped_column(String(255))
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    claim: Mapped[Claim] = relationship(back_populates="events")

    __table_args__ = (Index("ix_claim_events_claim", "claim_id", "occurred_at"),)


# ──────────────────────────────────────────────────────── corpus & chunks


class Document(UUIDPrimaryKey, TimestampMixin, Base):
    """A source file plus the metadata that makes retrieval filterable."""

    __tablename__ = "documents"

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, default=DEFAULT_TENANT)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    uploaded_by: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    # Extracted at ingest (ARCHITECTURE 5.2). These become Qdrant payload filters.
    insurer: Mapped[str | None] = mapped_column(String(255))
    product_name: Mapped[str | None] = mapped_column(String(255))
    uin: Mapped[str | None] = mapped_column(String(64))
    doc_type: Mapped[DocType] = mapped_column(
        _enum(DocType, "doc_type"), nullable=False, default=DocType.OTHER
    )
    language: Mapped[str] = mapped_column(String(8), nullable=False, default="en")
    jurisdiction: Mapped[str | None] = mapped_column(String(64))

    # The effective-date window. Retrieval filters on this whenever a date of
    # loss is in play, so an old claim is answered from the wording that applied.
    effective_from: Mapped[date | None] = mapped_column(Date)
    effective_to: Mapped[date | None] = mapped_column(Date)

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentVersion.version_no",
    )

    __table_args__ = (
        Index("ix_documents_tenant_product", "tenant_id", "product_name"),
        Index("ix_documents_effective", "effective_from", "effective_to"),
    )


class DocumentVersion(UUIDPrimaryKey, TimestampMixin, Base):
    """One ingestion of one file. Re-uploading identical bytes is a no-op."""

    __tablename__ = "document_versions"

    document_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(1024), nullable=False)

    status: Mapped[IngestionStatus] = mapped_column(
        _enum(IngestionStatus, "ingestion_status"), nullable=False, default=IngestionStatus.PENDING
    )
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # Superseded versions are marked, never deleted - needed for audit and for
    # claims adjudicated under an older wording (ARCHITECTURE 5.3).
    is_superseded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    parser: Mapped[str | None] = mapped_column(String(64))
    chunk_strategy_version: Mapped[str | None] = mapped_column(String(32))
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    embedding_dim: Mapped[int | None] = mapped_column(Integer)

    page_count: Mapped[int | None] = mapped_column(Integer)
    chunk_count: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)

    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    document: Mapped[Document] = relationship(back_populates="versions")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document_version", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("document_id", "version_no", name="uq_doc_versions_doc_version"),
        # Global dedup key: identical bytes are never ingested twice.
        UniqueConstraint("content_hash", name="uq_doc_versions_content_hash"),
        Index("ix_doc_versions_status", "status"),
    )


class Chunk(UUIDPrimaryKey, Base):
    """Mirror of what lives in Qdrant. ``id`` IS the Qdrant point id."""

    __tablename__ = "chunks"

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, default=DEFAULT_TENANT)
    document_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    # Children point at the parent section that gets expanded into the LLM
    # context at answer time (ARCHITECTURE 6.2).
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE")
    )

    kind: Mapped[ChunkKind] = mapped_column(
        _enum(ChunkKind, "chunk_kind"), nullable=False, default=ChunkKind.CHILD
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[str | None] = mapped_column(String(1024))
    page_no: Mapped[int | None] = mapped_column(Integer)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    token_count: Mapped[int | None] = mapped_column(Integer)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    # The breadcrumb-prefixed text that was actually embedded, kept separately so
    # a chunk's citation shows the clause, not the prefix.
    embedded_text: Mapped[str | None] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    is_superseded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    chunk_strategy_version: Mapped[str | None] = mapped_column(String(32))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    document_version: Mapped[DocumentVersion] = relationship(back_populates="chunks")
    children: Mapped[list[Chunk]] = relationship(
        back_populates="parent", remote_side=lambda: [Chunk.parent_id]
    )
    parent: Mapped[Chunk | None] = relationship(
        back_populates="children", remote_side=lambda: [Chunk.id]
    )

    __table_args__ = (
        Index("ix_chunks_version_index", "document_version_id", "chunk_index"),
        Index("ix_chunks_parent", "parent_id"),
        Index("ix_chunks_active", "tenant_id", "is_superseded"),
        Index("ix_chunks_document", "document_id"),
    )


# ──────────────────────────────────────────────────────────── conversation


class Conversation(UUIDPrimaryKey, TimestampMixin, Base):
    """Our own record of a conversation.

    Distinct from the LangGraph checkpointer's tables, which hold graph state.
    This one exists for audit, analytics and feedback - it must survive a
    checkpoint purge.
    """

    __tablename__ = "conversations"

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, default=DEFAULT_TENANT)
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    title: Mapped[str | None] = mapped_column(String(512))
    # LangGraph thread id, so a conversation can be resumed from its checkpoint.
    thread_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )

    __table_args__ = (Index("ix_conversations_user", "user_id", "created_at"),)


class Message(UUIDPrimaryKey, Base):
    """One turn, with everything needed to reconstruct how it was answered."""

    __tablename__ = "messages"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[MessageRole] = mapped_column(_enum(MessageRole, "message_role"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    intent: Mapped[Intent | None] = mapped_column(_enum(Intent, "intent"))
    model: Mapped[str | None] = mapped_column(String(128))
    prompt_version: Mapped[str | None] = mapped_column(String(32))

    # The audit trail required by ARCHITECTURE 10.4: what was retrieved, what was
    # cited, and whether the system chose to abstain.
    retrieved_chunk_ids: Mapped[list[str] | None] = mapped_column(JSONB)
    citations: Mapped[list[dict] | None] = mapped_column(JSONB)
    confidence: Mapped[float | None] = mapped_column(Float)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    latency_ms: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")
    feedback: Mapped[list[Feedback]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_messages_conversation", "conversation_id", "created_at"),)


class Feedback(UUIDPrimaryKey, Base):
    """Every thumbs-down is a candidate golden-set row (ARCHITECTURE 11.5)."""

    __tablename__ = "feedback"

    message_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    rating: Mapped[FeedbackRating] = mapped_column(
        _enum(FeedbackRating, "feedback_rating"), nullable=False
    )
    comment: Mapped[str | None] = mapped_column(Text)
    promoted_to_golden: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    message: Mapped[Message] = relationship(back_populates="feedback")

    __table_args__ = (UniqueConstraint("message_id", "user_id", name="uq_feedback_message_user"),)


# ────────────────────────────────────────────────────────── eval & audit


class EvalRun(UUIDPrimaryKey, Base):
    """One execution of the golden set. The CI regression gate reads this table."""

    __tablename__ = "eval_runs"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(64), nullable=False)
    git_sha: Mapped[str | None] = mapped_column(String(40))
    is_baseline: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # The full parameter set under test - chunk size, retrieval mode, reranker,
    # top_k - so an ablation arm is reproducible from its row alone.
    config: Mapped[dict] = mapped_column(JSONB, nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSONB)
    per_question: Mapped[list[dict] | None] = mapped_column(JSONB)
    notes: Mapped[str | None] = mapped_column(Text)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_eval_runs_name", "name", "started_at"),)


class AuditLog(UUIDPrimaryKey, Base):
    """Append-only. Never updated, never deleted.

    Both allowed and denied accesses are recorded - a denial pattern is exactly
    what an enumeration attempt looks like (ARCHITECTURE 10.1).
    """

    __tablename__ = "audit_log"

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, default=DEFAULT_TENANT)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    action: Mapped[AuditAction] = mapped_column(_enum(AuditAction, "audit_action"), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_id: Mapped[str | None] = mapped_column(String(128))
    # The policy holder whose data was touched - not necessarily the actor.
    subject_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    allowed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    ip_address: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(512))
    detail: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index("ix_audit_actor", "actor_user_id", "created_at"),
        Index("ix_audit_resource", "resource_type", "resource_id"),
        Index("ix_audit_denied", "allowed", "created_at"),
    )
