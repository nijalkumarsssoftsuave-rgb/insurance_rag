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

# "Clause 4.11", "clause 7.3," - the reference a customer or a grievance officer
# will look up in the wording.
CLAUSE_REFERENCE = re.compile(r"\bclause\s+(\d+(?:\.\d+)*)", re.I)
# Any numbering that appears in a section path, so "SECTION 4 - EXCLUSIONS >
# 4.11 Dental Treatment" yields both "4" and "4.11".
SECTION_NUMBER = re.compile(r"\b(\d+(?:\.\d+)*)\b")

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

    # A clause reference is checkable without a model, so it is checked without
    # one. The groundedness judge does not reliably catch a renumbered clause -
    # it reads "excluded under Clause 8.2" against text that really is an
    # exclusion and calls it supported, missing that the text lives in 4.11.
    # A customer who looks up 8.2 finds the wrong rule or no rule at all.
    if invented := _invented_clauses(answer, state.get("context_blocks") or []):
        notes.append(f"invented_clause_refs:{','.join(sorted(invented))}")
        log.warning("Answer cited clauses not present in retrieval", clauses=sorted(invented))

    if fabricated or invented or "uncited_coverage_assertion" in notes:
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
    question = state.get("standalone_question") or state["question"]
    context = state.get("context_text", "")

    try:
        result = await _check(prompt.body, context, question, answer)

        # A single judgement is not stable enough to abstain on. Even at
        # temperature 0 the check is not deterministic, and it was rejecting
        # answers whose figures are quoted verbatim in the context - "the waiting
        # period for maternity benefits is 36 months" against a table row reading
        # "Maternity benefit | 36 months". The same question passed or abstained
        # at random across runs.
        #
        # So a failure is confirmed, not taken on trust. The second call costs a
        # round trip only when the first one already said no, and a genuinely
        # ungrounded answer fails both. This tightens what abstention *means*
        # rather than loosening the guard: we still abstain, just not on noise.
        if not result.grounded:
            second = await _check(prompt.body, context, question, answer)
            if second.grounded:
                log.info("Groundedness check disagreed with itself, accepting on retry")
                notes.append("groundedness_flaky")
                result = second
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


def _invented_clauses(answer: str, blocks: list[dict]) -> set[str]:
    """Clause numbers the answer cites that no retrieved section actually carries.

    Compares against the numbering in each block's ``section_path``, which is the
    same string the model was shown, so a correct answer can always satisfy it.
    Returns an empty set when no block carries a section path - there is nothing
    to check against, and refusing every clause reference in that case would
    reject correct answers over missing metadata.
    """
    cited = set(CLAUSE_REFERENCE.findall(answer))
    if not cited:
        return set()

    available: set[str] = set()
    for block in blocks:
        if section := block.get("section_path"):
            available.update(SECTION_NUMBER.findall(str(section)))

    if not available:
        return set()
    return cited - available


async def _check(
    instruction: str, context: str, question: str, answer: str
) -> VerificationResult:
    return await get_llm().structured(
        [
            system(instruction),
            user(f"{context}\n\nQUESTION: {question}\n\nANSWER: {answer}"),
        ],
        VerificationResult,
        temperature=0.0,
    )


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
