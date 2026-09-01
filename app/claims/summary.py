"""LLM handover summary of one claim, written from the record and the adjuster notes.

Distinct from `service.render_status`, which is a deterministic template shown to
a *customer*. This is free text written for the *adjuster* who picks the file up
next, and it is free text precisely because the useful part - what the notes
actually say - has no template.

That makes it the thing worth judging. A template cannot invent a clause
reference; this can, and a summary that states a denial without the clause behind
it is a repudiation nobody can defend. `eval/week6/` exists to measure how often
that happens and whether the judge measuring it can be trusted.

The record is passed as a plain dict rather than a `ClaimView` so an eval case can
be a line of JSON with no database behind it. `from_claim_view` adapts the live
projection onto the same shape.
"""

from __future__ import annotations

import json
from typing import Any

from app.claims.repository import ClaimView
from app.llm import get_llm, system, user
from app.logging import get_logger
from app.prompts import registry

log = get_logger(__name__)

# The fields a summary is written from. Anything outside this list is not shown to
# the model, so a field added to the claim record cannot leak into a customer-
# adjacent artefact without someone naming it here first.
RECORD_FIELDS = (
    "claim_number",
    "status",
    "claim_type",
    "date_of_loss",
    "reported_at",
    "policy_number",
    "product_name",
    "claimed_amount",
    "approved_amount",
    "settled_amount",
    "currency",
    "rejection_reason_code",
    "rejection_clause_ref",
)


def from_claim_view(claim: ClaimView) -> dict[str, Any]:
    """Live projection -> the flat record shape an eval case also uses."""
    record: dict[str, Any] = {}
    for field in RECORD_FIELDS:
        value = getattr(claim, field, None)
        if value is None:
            continue
        record[field] = value.value if hasattr(value, "value") else str(value)
    record["notes"] = [
        {"note": e.note, "at": e.occurred_at.strftime("%Y-%m-%d"), "event": e.event_type}
        for e in sorted(claim.events, key=lambda e: e.occurred_at)
        if e.note
    ]
    return record


def render_input(record: dict[str, Any]) -> str:
    """What the model is shown. Deterministic, so a trace can be replayed."""
    fields = {k: v for k, v in record.items() if k in RECORD_FIELDS and v not in (None, "")}
    notes = record.get("notes") or []
    lines = [
        "<claim_record>",
        json.dumps(fields, indent=2, sort_keys=True, default=str),
        "</claim_record>",
    ]
    lines.append("<adjuster_notes>")
    if notes:
        for n in notes:
            stamp = n.get("at") or ""
            lines.append(f"- [{stamp}] {n.get('note', '')}")
    else:
        lines.append("(no notes logged)")
    lines.append("</adjuster_notes>")
    return "\n".join(lines)


async def summarise(record: dict[str, Any]) -> str:
    """One handover summary. Returns plain text - there is no citation contract here."""
    prompt = registry.load("claim_summary")
    try:
        # `complete` returns a Completion, and unlike `structured` it carries the
        # token usage - which is the field Week-5 replay could not reconstruct.
        completion = await get_llm().complete(
            [system(prompt.body), user(render_input(record))],
            temperature=prompt.temperature or 0.1,
        )
        return completion.text.strip()
    except Exception as exc:
        log.error("Claim summary failed", claim=record.get("claim_number"), error=str(exc))
        raise
