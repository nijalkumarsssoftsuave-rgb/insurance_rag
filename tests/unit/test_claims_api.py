"""The claim-summary endpoint's access rules.

The summary is written from `ClaimEvent.note` - internal working text where
assessors write things like "recommend rejection under 4.13". That is staff-only,
and the guard is the whole reason this endpoint has a test.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.core.enums import UserRole
from app.deps import get_subject
from app.main import app
from app.security.authz import AuthSubject

URL = "/api/v1/claims/CLM-2026-0004/summary"


def _subject(role: UserRole) -> AuthSubject:
    return AuthSubject(
        user_id=uuid.uuid4(),
        tenant_id="default",
        role=role,
        policy_holder_id=None if role in (UserRole.AGENT, UserRole.ADMIN) else uuid.uuid4(),
    )


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_customer_is_refused(client) -> None:
    """403 rather than 404: the caller is authenticated and simply lacks the role."""
    app.dependency_overrides[get_subject] = lambda: _subject(UserRole.CUSTOMER)
    response = client.get(URL)
    assert response.status_code == 403
    assert "staff" in response.json()["detail"].lower()


def test_customer_is_refused_before_the_database_is_touched(client) -> None:
    """The role check must come first, or a refused caller still learns whether the
    claim exists from how long the refusal took."""
    app.dependency_overrides[get_subject] = lambda: _subject(UserRole.CUSTOMER)
    for claim_number in ("CLM-2026-0004", "CLM-9999-9999", "not-a-claim-number"):
        assert client.get(f"/api/v1/claims/{claim_number}/summary").status_code == 403


def test_the_route_is_mounted(client) -> None:
    """`app/api/v1/claims.py` sat as an empty file that `app/api/router.py` never
    included, so the endpoint existed only in principle.

    Asserted against the OpenAPI schema rather than `app.routes`: this FastAPI
    version defers router expansion, so a mounted route does not appear in
    `app.routes` at all and the obvious check quietly passes on nothing.
    """
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/claims/{claim_number}/summary" in paths, sorted(paths)


def test_the_endpoint_is_wired_to_the_squad_not_the_old_summariser(client, monkeypatch) -> None:
    """Week 12: this endpoint called `summary.summarise()` through Weeks 7, 8
    and 10 without ever being pointed at the validated replacement those weeks
    built - reproduced live against CLM-2026-0005 (Week 8's injection
    fixture), where the old path told the adjuster to approve the claim on a
    fabricated "supervisor correction". No real LLM call here (that
    reproduction lives in eval/week12/), just proof the endpoint actually
    calls `claims_squad.run` and correctly surfaces what it returns -
    wiring, not model behaviour.
    """
    import app.api.v1.claims as claims_module
    from app.agents.claims_squad import SquadResult

    async def fake_run(session, subject, claim_number):  # noqa: ARG001
        return SquadResult(
            answer="stubbed squad answer",
            raw_answer="stubbed squad answer",
            output_overridden=True,
        )

    monkeypatch.setattr(claims_module.claims_squad, "run", fake_run)
    app.dependency_overrides[get_subject] = lambda: _subject(UserRole.AGENT)

    response = client.get("/api/v1/claims/CLM-2026-0004/summary")

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == "stubbed squad answer"
    assert body["output_overridden"] is True
