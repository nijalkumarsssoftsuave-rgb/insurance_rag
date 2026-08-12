"""Row-level ownership enforcement. Identity never comes from the LLM.

The single most important type in this module is ``AuthSubject``. It is built
from a verified JWT and threaded through the graph as opaque state. No node
constructs one, no prompt can influence one, and no tool accepts a subject as a
model-supplied argument.

The attack this prevents is concrete: a user types "show me claim CLM-99999", the
router dutifully extracts the id, and a tool that trusted the model for identity
returns someone else's medical claim (ARCHITECTURE 10.1).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from app.core.enums import UserRole


class AuthorizationError(PermissionError):
    """Raised when a subject is asked to act outside its scope."""


@dataclass(slots=True, frozen=True)
class AuthSubject:
    """A verified caller. Frozen: nothing downstream may widen its own scope."""

    user_id: uuid.UUID
    tenant_id: str
    role: UserRole
    # None for staff logins, who have no policy of their own.
    policy_holder_id: uuid.UUID | None = None

    @property
    def is_staff(self) -> bool:
        return self.role in (UserRole.AGENT, UserRole.ADMIN)

    @property
    def can_read_any_claim(self) -> bool:
        """Staff see claims across holders; customers see only their own.

        Staff access is still written to the audit log - broad is not unlogged.
        """
        return self.is_staff

    def require_policy_holder(self) -> uuid.UUID:
        """The holder id a customer's query must be scoped to.

        Raises rather than returning None: a caller that reaches this without a
        holder id would otherwise be one ``if`` away from an unscoped query.
        """
        if self.policy_holder_id is None:
            raise AuthorizationError(
                "This account is not linked to a policy holder, so it has no claims to read."
            )
        return self.policy_holder_id


@dataclass(slots=True, frozen=True)
class AccessDecision:
    allowed: bool
    reason: str

    @classmethod
    def allow(cls) -> AccessDecision:
        return cls(allowed=True, reason="owner")


def authorize_claim_access(subject: AuthSubject, *, claim_tenant_id: str) -> AccessDecision:
    """Tenant gate, applied before any row is read.

    Ownership itself is enforced in the SQL predicate, not here - see
    ``app.claims.repository``. This function exists to reject cross-tenant access
    early and to give the audit log a reason string.
    """
    if subject.tenant_id != claim_tenant_id:
        return AccessDecision(allowed=False, reason="cross_tenant")
    if subject.can_read_any_claim:
        return AccessDecision(allowed=True, reason="staff")
    if subject.policy_holder_id is None:
        return AccessDecision(allowed=False, reason="no_policy_holder")
    return AccessDecision.allow()
