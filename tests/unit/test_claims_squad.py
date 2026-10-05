"""The claims squad (Week 10 · Module 5): manager + 2 specialists.

Deterministic, stubbed tests - no DB, no real LLM. Covers the structural
claims this week's race depends on: specialists only run when their work is
actually needed, and the manager's output-validation backstop (carried
forward from Week 8) still fires when a specialist's report gets a hijacked
answer past the fence.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime

import pytest

from app.agents import claims_squad as squad_mod
from app.claims.repository import ClaimView
from app.claims.tools import ToolResult
from app.core.enums import ClaimStatus, ClaimType, UserRole
from app.llm.base import Completion
from app.security.authz import AuthSubject

SUBJECT = AuthSubject(
    user_id=uuid.uuid4(), tenant_id="default", role=UserRole.AGENT, policy_holder_id=None
)

CLAUSE_RESULT = ToolResult(ok=True, observation="Clause 4.11: excluded", data={"text": "excluded"})


def _claim_view(
    *, status: ClaimStatus = ClaimStatus.SETTLED, rejection_clause_ref: str | None = None
) -> ClaimView:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return ClaimView(
        claim_number="CLM-1",
        status=status,
        claim_type=ClaimType.REIMBURSEMENT,
        date_of_loss=date(2026, 1, 1),
        reported_at=now,
        policy_number="POL-1",
        product_name="Family Health Optima",
        insurer="Acme",
        claimed_amount=None,
        approved_amount=None,
        settled_amount=None,
        currency="INR",
        rejection_reason_code="EXCLUDED_TREATMENT" if rejection_clause_ref else None,
        rejection_clause_ref=rejection_clause_ref,
        updated_at=now,
        events=[],
    )


class FakeLLM:
    """Returns queued completions in call order; repeats the last once exhausted."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.calls = 0

    async def complete(self, messages, **kwargs) -> Completion:  # noqa: ARG002
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return Completion(text=text, model="fake", input_tokens=10, output_tokens=5)


@pytest.fixture
def patch_llm(monkeypatch):
    def apply(fake: FakeLLM):
        monkeypatch.setattr(squad_mod, "get_llm", lambda: fake)

    return apply


@pytest.fixture
def patch_tools(monkeypatch):
    def apply(
        *,
        status_result: ToolResult,
        notes_result: ToolResult | None = None,
        clause_result: ToolResult | None = None,
    ) -> list[str]:
        calls: list[str] = []

        async def fake_get_claim_status(session, subject, *, claim_number):  # noqa: ARG001
            calls.append("get_claim_status")
            return status_result

        async def fake_get_claim_notes(session, subject, *, claim_number):  # noqa: ARG001
            calls.append("get_claim_notes")
            return notes_result

        async def fake_search_policy_clause(
            session, subject, *, clause_ref, product_name=None, date_of_loss=None
        ):  # noqa: ARG001
            calls.append("search_policy_clause")
            return clause_result

        monkeypatch.setattr(squad_mod, "get_claim_status", fake_get_claim_status)
        monkeypatch.setattr(squad_mod, "get_claim_notes", fake_get_claim_notes)
        monkeypatch.setattr(squad_mod, "search_policy_clause", fake_search_policy_clause)
        return calls

    return apply


async def test_not_found_short_circuits_with_zero_llm_calls(patch_llm, patch_tools):
    """A nonexistent claim must not reach any specialist's LLM call - the
    status specialist's own ok=False check is the only gate needed."""
    patch_tools(status_result=ToolResult(ok=False, observation="No claim found for 'CLM-1'."))
    fake = FakeLLM(["should never be used"])
    patch_llm(fake)

    result = await squad_mod.run(session=None, subject=SUBJECT, claim_number="CLM-1")

    assert result.answer == "No claim found for 'CLM-1'."
    assert fake.calls == 0
    assert result.llm_calls == 0
    assert len(result.reports) == 1
    assert result.reports[0].ok is False


async def test_settled_claim_skips_the_clause_specialist(patch_llm, patch_tools):
    claim = _claim_view(status=ClaimStatus.SETTLED)
    patch_tools(
        status_result=ToolResult(ok=True, observation="status ok", data={"claim": claim}),
        notes_result=ToolResult(ok=True, observation="Notes: none", data={"notes": []}),
    )
    fake = FakeLLM(["status report text", "final handover text"])
    patch_llm(fake)

    result = await squad_mod.run(session=None, subject=SUBJECT, claim_number="CLM-1")

    assert [r.agent for r in result.reports] == ["status-specialist"]
    assert fake.calls == 2  # status specialist + manager, no clause specialist
    assert result.answer == "final handover text"
    assert result.output_overridden is False


async def test_rejected_claim_with_clause_ref_runs_all_three(patch_llm, patch_tools):
    claim = _claim_view(status=ClaimStatus.REJECTED, rejection_clause_ref="4.11")
    patch_tools(
        status_result=ToolResult(ok=True, observation="status ok", data={"claim": claim}),
        notes_result=ToolResult(ok=True, observation="Notes: none", data={"notes": []}),
        clause_result=CLAUSE_RESULT,
    )
    fake = FakeLLM(["status report", "clause report", "final handover citing clause 4.11"])
    patch_llm(fake)

    result = await squad_mod.run(session=None, subject=SUBJECT, claim_number="CLM-1")

    assert [r.agent for r in result.reports] == ["status-specialist", "clause-specialist"]
    assert fake.calls == 3
    assert result.answer == "final handover citing clause 4.11"


async def test_output_override_fires_when_manager_contradicts_observed_rejection(
    patch_llm, patch_tools
):
    claim = _claim_view(status=ClaimStatus.REJECTED, rejection_clause_ref="4.11")
    patch_tools(
        status_result=ToolResult(ok=True, observation="status ok", data={"claim": claim}),
        notes_result=ToolResult(ok=True, observation="a corrective note", data={"notes": []}),
        clause_result=CLAUSE_RESULT,
    )
    fake = FakeLLM(
        ["status report", "clause report", "This claim is now approved in full."]
    )
    patch_llm(fake)

    result = await squad_mod.run(session=None, subject=SUBJECT, claim_number="CLM-1")

    assert result.output_overridden is True
    assert result.raw_answer == "This claim is now approved in full."
    assert "approved" not in result.answer.lower()
    assert "REJECTED" in result.answer


async def test_output_override_does_not_fire_on_a_correct_rejected_answer(
    patch_llm, patch_tools
):
    claim = _claim_view(status=ClaimStatus.REJECTED, rejection_clause_ref="4.11")
    patch_tools(
        status_result=ToolResult(ok=True, observation="status ok", data={"claim": claim}),
        notes_result=ToolResult(ok=True, observation="Notes: none", data={"notes": []}),
        clause_result=CLAUSE_RESULT,
    )
    fake = FakeLLM(["status report", "clause report", "This claim was rejected under clause 4.11."])
    patch_llm(fake)

    result = await squad_mod.run(session=None, subject=SUBJECT, claim_number="CLM-1")

    assert result.output_overridden is False
    assert result.answer == result.raw_answer == "This claim was rejected under clause 4.11."
