"""audit.record() must actually never raise, and must leave the session
usable for whatever the caller does next (Week 11).

Found by running eval/week11/cost_report.py with a subject whose user_id did
not exist in `users` yet: the FK violation was logged (as designed), but the
session was left in SQLAlchemy's post-failed-flush "pending rollback" state,
so claim_lookup_node's own later commit on that same session raised
PendingRollbackError - a crash that looked unrelated to auditing, from code
that never touched audit.py's exception handling.
"""

from __future__ import annotations

import uuid

from sqlalchemy import text

from app.core.enums import AuditAction, UserRole
from app.db.session import async_session_scope
from app.security import audit
from app.security.authz import AuthSubject


async def test_failed_audit_write_does_not_poison_the_session() -> None:
    nonexistent_user = AuthSubject(
        user_id=uuid.uuid4(),  # not in `users` - the FK this constraint checks
        tenant_id="default",
        role=UserRole.CUSTOMER,
        policy_holder_id=None,
    )

    async with async_session_scope() as session:
        # Must not raise - record()'s whole contract is "log it, don't crash
        # the caller" - and the session must still be usable afterward.
        await audit.record(
            session,
            subject=nonexistent_user,
            action=AuditAction.CLAIM_VIEWED,
            resource_type="claim",
            resource_id="CLM-TEST-AUDIT-ROLLBACK",
            allowed=True,
        )
        # Before the fix, this line raised PendingRollbackError - the session
        # was left dirty by the failed flush above, with no rollback issued.
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1
