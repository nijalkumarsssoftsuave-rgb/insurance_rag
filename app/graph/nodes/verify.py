"""Citation validation and groundedness check before release.

Two layers, cheapest first:

1. **Deterministic** - every cited id must exist in what was actually retrieved,
   and a substantive answer must cite something. This costs nothing and catches
   the most common failure: a fabricated citation that looks authoritative.
2. **Model-based** - an LLM checks whether each claim is supported. This costs a
   round trip, so it runs only when the answer is high-stakes (a coverage
   determination) or the deterministic layer already looks shaky.

A failed verification downgrades the answer to an abstention. Publishing an
uncited coverage claim is a defect, not a stylistic issue (ARCHITECTURE 10.4).
"""

from __future__ import annotations

import re
import time

from pydantic import BaseModel, Field

from app.graph.state import ConversationState
from app.llm import get_llm, system, user
from app.logging import get_logger
from app.prompts import registry

log = get_logger(__name__)

CITATION_PATTERN = re.compile(r"\[([0-9a-fA-F-]{8,})\]")

# Only a *positive* coverage claim needs the exclusions check. An answer that
# says "dental is excluded" IS the exclusion - demanding it also prove it checked
# exclusions rejects correct answers, which is what happened: three of four
# questions abstained with `exclusions_not_checked` while retrieving perfectly.
POSITIVE_COVERAGE = re.compile(
    r"\b(is|are)\s+(covered|payable|eligible|admissible|reimbursable)\b"
    r"|\byou\s+(can|may)\s+claim\b"
    r"|\bwill\s+be\s+(paid|reimbursed|covered)\b",
    re.I,
)
# Any coverage-shaped statement, positive or negative. Used only for the cheaper
# "did it cite anything at all" check.
COVERAGE_ASSERTION = re.compile(
    r"\b(is|are)\s+(covered|excluded|payable|eligible)\b|\byou\s+(can|cannot|may)\s+claim\b",
    re.I,
)

UNVERIFIED_MESSAGE = (
    "I found related policy text but couldn't verify my answer against it well "
    "enough to be confident. I'd rather not guess - shall I connect you to an agent?"
)


class VerificationResult(BaseModel):
    grounded: bool = Field(description="Is every factual claim supported by the context")
    unsupported_claims: list[str] = Field(default_factory=list)
    checked_exclusions: bool = Field(
        default=True, description="For coverage answers: were exclusions considered"
    )


async def verify_node(state: ConversationState) -> dict:
    started = time.perf_counter()

    if state.get("abstained"):
        return {"verified": True, "timings_ms": {"verify": 0}}

    answer = state.get("answer") or ""
    retrieved = set(state.get("retrieved_chunk_ids") or [])
    citations = state.get("citations") or []
    route = state.get("route") or {}
    notes: list[str] = []

    # ── layer 1: deterministic ───────────────────────────────────────────
    inline_ids = set(CITATION_PATTERN.findall(answer))
    fabricated = {cid for cid in inline_ids if cid not in retrieved}
    if fabricated:
        notes.append(f"fabricated_citations:{len(fabricated)}")
        log.warning("Answer cited chunks that were not retrieved", count=len(fabricated))

    asserts_coverage = bool(COVERAGE_ASSERTION.search(answer))
    if asserts_coverage and not citations and not inline_ids:
        notes.append("uncited_coverage_assertion")

    if fabricated or "uncited_coverage_assertion" in notes:
        return _reject(notes, started)

    # ── layer 2: model-based, only where it earns its latency ────────────
    needs_llm_check = bool(route.get("is_coverage_question")) or asserts_coverage
    if not needs_llm_check:
        return {
            "verified": True,
            "verification_notes": notes,
            "timings_ms": {"verify": int((time.perf_counter() - started) * 1000)},
        }

    prompt = registry.load("verify")
    try:
        result = await get_llm().structured(
            [
                system(prompt.body),
                user(
                    f"{state.get('context_text', '')}\n\n"
                    f"QUESTION: {state.get('standalone_question') or state['question']}\n\n"
                    f"ANSWER: {answer}"
                ),
            ],
            VerificationResult,
            temperature=0.0,
        )
    except Exception as exc:
        # A failed check is not a failed answer. The deterministic layer already
        # passed; degrade to that rather than abstaining on an infrastructure error.
        log.warning("Verification call failed, keeping deterministic result", error=str(exc))
        notes.append("llm_verify_unavailable")
        return {
            "verified": True,
            "verification_notes": notes,
            "timings_ms": {"verify": int((time.perf_counter() - started) * 1000)},
        }

    if not result.grounded:
        notes.append(f"unsupported_claims:{len(result.unsupported_claims)}")
        log.warning("Groundedness check failed", claims=result.unsupported_claims[:3])
        return _reject(notes, started)

    # Only gate on exclusions when the answer actually tells the customer that
    # something IS covered. That is the statement with money attached; a
    # negative answer, a sub-limit figure or a waiting period does not need the
    # exclusions section to be defensible.
    if POSITIVE_COVERAGE.search(answer) and not result.checked_exclusions:
        notes.append("exclusions_not_checked")
        return _reject(notes, started)

    return {
        "verified": True,
        "verification_notes": notes,
        "timings_ms": {"verify": int((time.perf_counter() - started) * 1000)},
    }


def _reject(notes: list[str], started: float) -> dict:
    return {
        "answer": UNVERIFIED_MESSAGE,
        "citations": [],
        "verified": False,
        "abstained": True,
        "needs_human": True,
        "confidence": 0.0,
        "verification_notes": notes,
        "timings_ms": {"verify": int((time.perf_counter() - started) * 1000)},
    }
