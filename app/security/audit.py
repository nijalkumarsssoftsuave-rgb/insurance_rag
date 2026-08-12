"""Append-only audit trail for every claim access and every answer.

Append-only in practice as well as intent: there is no update or delete helper in
this module. When a customer escalates to a grievance officer or an ombudsman,
you must be able to reconstruct exactly what the assistant said and on what
evidence (ARCHITECTURE 10.4).

Denied accesses are recorded alongside allowed ones. A run of denials from one
account is what claim-number enumeration looks like, and it is invisible if only
successes are logged.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AuditAction
from app.db.models import AuditLog
from app.logging import get_logger
from app.security.authz import AuthSubject

log = get_logger(__name__)


async def record(
    session: AsyncSession,
    *,
    subject: AuthSubject,
    action: AuditAction,
    resource_type: str,
    resource_id: str | None,
    allowed: bool = True,
    detail: dict[str, Any] | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Write one audit entry.

    Never raises. A failed audit write must not take down the request that
    triggered it - but it is logged at error level, because an audit trail with
    silent gaps is worse than no audit trail.
    """
    try:
        session.add(
            AuditLog(
                tenant_id=subject.tenant_id,
                actor_user_id=subject.user_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                subject_id=subject.policy_holder_id,
                allowed=allowed,
                ip_address=ip_address,
                user_agent=user_agent,
                detail=detail,
            )
        )
        await session.flush()
    except Exception as exc:
        log.error(
            "Audit write failed",
            action=action.value,
            resource_type=resource_type,
            resource_id=resource_id,
            error=str(exc),
        )


async def record_answer(
    session: AsyncSession,
    *,
    subject: AuthSubject,
    conversation_id: str,
    abstained: bool,
    chunk_ids: list[str],
    model: str | None,
    prompt_version: str | None,
    confidence: float,
) -> None:
    """Record that an answer was produced, and on what evidence."""
    await record(
        session,
        subject=subject,
        action=AuditAction.ANSWER_ABSTAINED if abstained else AuditAction.ANSWER_GENERATED,
        resource_type="conversation",
        resource_id=conversation_id,
        allowed=True,
        detail={
            "chunk_ids": chunk_ids[:20],
            "chunk_count": len(chunk_ids),
            "model": model,
            "prompt_version": prompt_version,
            "confidence": round(confidence, 3),
        },
    )


def new_conversation_id() -> str:
    return str(uuid.uuid4())
