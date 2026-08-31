"""FastAPI dependency providers: db session, current user, singletons."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.enums import UserRole
from app.db.session import get_db
from app.logging import get_logger
from app.security.authz import AuthSubject

log = get_logger(__name__)

DbSession = Annotated[AsyncSession, Depends(get_db)]

# Stable ids so POC conversations and seeded claims group under one subject
# across restarts. Shared with scripts/seed_claims.py, which writes the matching
# policy_holder row.
_DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-0000000000d0")
DEMO_POLICY_HOLDER_ID = uuid.UUID("00000000-0000-0000-0000-0000000000b0")


def _demo_policy_holder_id() -> uuid.UUID | None:
    """Whether the demo subject owns any policies.

    Unlinked by default: with no holder id, a claim lookup refuses instead of
    returning somebody else's claim, so the authorization path is exercised
    rather than bypassed (ARCHITECTURE 10.1).

    Set ``DEMO_POLICY_HOLDER=1`` (what ``scripts/seed_claims.py`` prints) to link
    the subject to the seeded holder so the claim lane can be demonstrated. This
    is a flag rather than a hardcoded id so the refusing path stays the default
    and stays reachable - flipping it off is how you show the guard working.

    Read through ``settings``, not ``os.environ``: values in .env are parsed by
    pydantic-settings and never reach the real environment.
    """
    return DEMO_POLICY_HOLDER_ID if settings.app.demo_policy_holder else None


DEMO_SUBJECT = AuthSubject(
    user_id=_DEMO_USER_ID,
    tenant_id="default",
    role=UserRole.CUSTOMER,
    policy_holder_id=_demo_policy_holder_id(),
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
