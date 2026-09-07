"""Tools an agent (app/agents/claim_handover_agent.py) may call.

Same authorization boundary as everywhere else in app/claims: every function
takes ``subject`` as a fixed, code-supplied argument, never something the model
produces. The model only ever supplies a *claim number* or *clause reference* -
exactly what ``app/graph/nodes/route.py`` already lets it extract from free text
today. It never supplies *whose* claim, so a wrong or malicious argument here
degrades to "not found", not to someone else's data (ARCHITECTURE 10.1, 10.2).

``search_policy_clause``'s ``date_of_loss`` gets the same treatment: it decides
*which wording version* is quoted (ARCHITECTURE 5.3), so the agent loop fills it
in from the claim record it already fetched rather than trusting the model to
supply one.

Each tool returns a ``ToolResult``: a short natural-language ``observation`` (fed
back into the agent's transcript) plus ``data`` (kept out of the prompt, used by
the fixed workflow and by eval assertions that need the structured record).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.claims import repository
from app.core.enums import ClaimStatus
from app.retrieval.filters import RetrievalFilter
from app.retrieval.hybrid import get_pipeline
from app.security.authz import AuthSubject


@dataclass(slots=True)
class ToolResult:
    ok: bool
    observation: str
    data: dict[str, Any] | None = None


async def get_claim_status(
    session: AsyncSession, subject: AuthSubject, *, claim_number: str
) -> ToolResult:
    claim = await repository.get_claim(session, subject, claim_number)
    if claim is None:
        return ToolResult(ok=False, observation=f"No claim found for '{claim_number}'.")
    return ToolResult(
        ok=True,
        observation=(
            f"Claim {claim.claim_number} ({claim.product_name}): status={claim.status.value}, "
            f"date_of_loss={claim.date_of_loss.isoformat()}"
            + (
                f", rejection_reason={claim.rejection_reason_code}, "
                f"rejection_clause_ref={claim.rejection_clause_ref}"
                if claim.status is ClaimStatus.REJECTED
                else ""
            )
        ),
        data={"claim": claim},
    )


async def get_claim_notes(
    session: AsyncSession, subject: AuthSubject, *, claim_number: str
) -> ToolResult:
    events = await repository.recent_events(session, subject, claim_number)
    notes = [e for e in events if e.note]
    if not notes:
        return ToolResult(
            ok=True, observation="No adjuster notes on this claim.", data={"notes": []}
        )
    rendered = "; ".join(f"[{e.occurred_at:%Y-%m-%d}] {e.note}" for e in notes)
    return ToolResult(ok=True, observation=f"Notes: {rendered}", data={"notes": notes})


async def search_policy_clause(
    session: AsyncSession,  # noqa: ARG001 - kept for a uniform tool signature
    subject: AuthSubject,
    *,
    clause_ref: str,
    product_name: str | None = None,
    date_of_loss: date | None = None,
) -> ToolResult:
    """Exact clause lookup - filter match, not similarity search.

    ``retrieve_by_clause`` already existed in ``app/retrieval/hybrid.py`` but was
    never called from anywhere; this is that tool wired up for the first time.

    ``date_of_loss`` matters: the corpus carries two non-superseded wordings of
    clause 4.11 a claim apart (tightened 2026-04-01), each valid only in its own
    effective window. Without the date filter the wrong one can come back - see
    the fix in ``retrieve_by_clause`` (ARCHITECTURE 5.3). Callers should pass the
    claim's own date of loss, not let the model guess one.
    """
    if not clause_ref:
        return ToolResult(ok=False, observation="No clause reference given to search for.")

    rf = RetrievalFilter(
        tenant_id=subject.tenant_id, product_name=product_name, date_of_loss=date_of_loss
    )
    hits = await get_pipeline().retrieve_by_clause(clause_ref, rf)
    if not hits:
        return ToolResult(
            ok=False, observation=f"Clause {clause_ref} was not found in the policy wording."
        )
    text = hits[0].text.strip()
    if not text:
        return ToolResult(
            ok=False, observation=f"Clause {clause_ref} matched a document but had no text."
        )
    snippet = text if len(text) <= 600 else text[:600] + "..."
    return ToolResult(ok=True, observation=f"Clause {clause_ref}: {snippet}", data={"text": text})


TOOL_SPECS: dict[str, str] = {
    "get_claim_status": (
        "Look up one claim's status, dates and (if rejected) its reason code and "
        "clause reference. Args: {\"claim_number\": \"CLM-2026-0004\"}."
    ),
    "get_claim_notes": (
        "Fetch the adjuster's working notes for one claim, newest first. "
        "Args: {\"claim_number\": \"CLM-2026-0004\"}."
    ),
    "search_policy_clause": (
        "Fetch the exact wording of one policy clause by its reference number - "
        "use this to explain *why* a claim was rejected. "
        "Args: {\"clause_ref\": \"4.11\", \"product_name\": \"Family Health Optima\"}."
    ),
}

_DISPATCH = {
    "get_claim_status": get_claim_status,
    "get_claim_notes": get_claim_notes,
    "search_policy_clause": search_policy_clause,
}


async def run_tool(
    name: str, args: dict[str, Any], *, session: AsyncSession, subject: AuthSubject
) -> ToolResult:
    fn = _DISPATCH.get(name)
    if fn is None:
        return ToolResult(
            ok=False, observation=f"Unknown tool '{name}'. Available: {list(_DISPATCH)}"
        )
    try:
        return await fn(session, subject, **args)
    except TypeError as exc:
        return ToolResult(ok=False, observation=f"Bad arguments for '{name}': {exc}")
