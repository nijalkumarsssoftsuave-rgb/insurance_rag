"""Fixed-sequence counterpart to ``app/agents/claim_handover_agent.py``.

Same task, same three tools, no model deciding what happens next: the branch is
one ``if``, exactly like the Lane B -> Lane A hand-off in
``app/graph/nodes/claim_lookup.py``. Packaged as a function returning a
comparable ``HandoverResult`` so ``eval/week7/race.py`` can run both against the
same claims.

No LLM call anywhere in this path, on purpose. When every branch is already
known, spending a model call to decide "what to do next" answers a question
that was never open - and here that call would also be the only source of a
hallucinated clause number, which a template cannot produce.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.claims.repository import ClaimView
from app.claims.tools import ToolResult, get_claim_notes, get_claim_status, search_policy_clause
from app.core.enums import ClaimStatus
from app.security.authz import AuthSubject


@dataclass(slots=True)
class HandoverResult:
    answer: str
    tool_calls: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: int = 0
    steps: list[str] = field(default_factory=list)


async def run(session: AsyncSession, subject: AuthSubject, claim_number: str) -> HandoverResult:
    started = time.perf_counter()
    result = HandoverResult(answer="")

    status = await get_claim_status(session, subject, claim_number=claim_number)
    result.tool_calls += 1
    result.steps.append(status.observation)
    if not status.ok or status.data is None:
        result.answer = status.observation
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    claim = status.data["claim"]

    notes = await get_claim_notes(session, subject, claim_number=claim_number)
    result.tool_calls += 1
    result.steps.append(notes.observation)

    clause_text: str | None = None
    if claim.status is ClaimStatus.REJECTED and claim.rejection_clause_ref:
        clause = await search_policy_clause(
            session,
            subject,
            clause_ref=claim.rejection_clause_ref,
            product_name=claim.product_name,
            date_of_loss=claim.date_of_loss,
        )
        result.tool_calls += 1
        result.steps.append(clause.observation)
        if clause.ok and clause.data:
            clause_text = clause.data["text"]

    result.answer = _render(claim, notes, clause_text)
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)
    return result


def _render(claim: ClaimView, notes: ToolResult, clause_text: str | None) -> str:
    lines = [f"Claim {claim.claim_number} ({claim.product_name}): {claim.status.value}."]

    if claim.status is ClaimStatus.REJECTED:
        reason = (claim.rejection_reason_code or "").replace("_", " ").lower()
        if claim.rejection_clause_ref:
            lines.append(f"Rejected under clause {claim.rejection_clause_ref} ({reason}).")
            lines.append(
                f"Clause text: {clause_text[:400]}"
                if clause_text
                else "Clause text could not be retrieved."
            )
        else:
            lines.append(f"Rejected ({reason}); no clause reference on file.")

    note_records = (notes.data or {}).get("notes") or []
    if note_records:
        latest = max(note_records, key=lambda e: e.occurred_at)
        lines.append(f"Latest note ({latest.occurred_at:%d %b %Y}): {latest.note}")

    return " ".join(lines)
