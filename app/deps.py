"""FastAPI dependency providers: db session, current user, singletons."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import UserRole
from app.db.session import get_db
from app.logging import get_logger
from app.security.authz import AuthSubject

log = get_logger(__name__)

DbSession = Annotated[AsyncSession, Depends(get_db)]

# Stable id so POC conversations group under one subject across restarts.
_DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-0000000000d0")

DEMO_SUBJECT = AuthSubject(
    user_id=_DEMO_USER_ID,
    tenant_id="default",
    role=UserRole.CUSTOMER,
    # Deliberately None. The demo subject owns no policies, so any claim lookup
    # refuses with "this account is not linked to a policy holder" instead of
    # silently returning someone else's claim. The authorization path stays
    # exercised rather than bypassed (ARCHITECTURE 10.1).
    policy_holder_id=None,
)


async def get_subject() -> AuthSubject:
    """The authenticated caller.

    **POC only.** There is no login yet, so every request runs as a fixed demo
    customer. This is the single seam where real authentication lands: verify the
    JWT, load the user, and build the subject from it. Nothing downstream changes,
    because no node ever constructs a subject for itself.

    Document Q&A works fully under this subject. Claim lookups correctly refuse,
    which is the honest behaviour for an unauthenticated caller.
    """
    return DEMO_SUBJECT


Subject = Annotated[AuthSubject, Depends(get_subject)]
