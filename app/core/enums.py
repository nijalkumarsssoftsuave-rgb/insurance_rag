"""Shared enumerations. Used by ORM models, API schemas and the graph alike."""

from enum import StrEnum


class UserRole(StrEnum):
    CUSTOMER = "customer"
    AGENT = "agent"
    ADMIN = "admin"


class DocType(StrEnum):
    """Document classes we ingest. Drives retrieval filters and chunking rules."""

    POLICY_WORDING = "policy_wording"
    ENDORSEMENT = "endorsement"
    SOP = "sop"
    CIRCULAR = "circular"
    CLAIM_FORM = "claim_form"
    BROCHURE = "brochure"
    OTHER = "other"


class IngestionStatus(StrEnum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkKind(StrEnum):
    """Parents are returned to the LLM; children are what get embedded (ARCHITECTURE 6.2)."""

    PARENT = "parent"
    CHILD = "child"
    TABLE = "table"
    TABLE_SUMMARY = "table_summary"


class PolicyStatus(StrEnum):
    ACTIVE = "active"
    LAPSED = "lapsed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class ClaimStatus(StrEnum):
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    INFO_REQUIRED = "info_required"
    APPROVED = "approved"
    PARTIALLY_APPROVED = "partially_approved"
    REJECTED = "rejected"
    SETTLED = "settled"
    CLOSED = "closed"
    WITHDRAWN = "withdrawn"


class ClaimType(StrEnum):
    CASHLESS = "cashless"
    REIMBURSEMENT = "reimbursement"
    PRE_AUTHORIZATION = "pre_authorization"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Intent(StrEnum):
    """Router output. Decides which lane handles the turn (ARCHITECTURE 1)."""

    POLICY_QA = "policy_qa"
    CLAIM_STATUS = "claim_status"
    CLAIM_INTAKE = "claim_intake"
    SMALLTALK = "smalltalk"
    OUT_OF_SCOPE = "out_of_scope"


class FeedbackRating(StrEnum):
    UP = "up"
    DOWN = "down"


class AuditAction(StrEnum):
    """Every entry here is reconstructable evidence for a grievance review."""

    CLAIM_VIEWED = "claim_viewed"
    CLAIM_ACCESS_DENIED = "claim_access_denied"
    ANSWER_GENERATED = "answer_generated"
    ANSWER_ABSTAINED = "answer_abstained"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_DELETED = "document_deleted"
    LOGIN = "login"
    LOGIN_FAILED = "login_failed"
