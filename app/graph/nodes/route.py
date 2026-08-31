"""Intent classification + entity extraction in one structured call.

One LLM round trip produces both the lane decision and the metadata filter inputs.
Splitting them would cost an extra 300 ms for no benefit - the model is reading
the same sentence either way (ARCHITECTURE 8, step 3).

The extracted ``claim_number`` decides *what* to look up. It never decides *whose*
claim - that comes from the verified subject in state, and no amount of prompt
manipulation can change it.
"""

from __future__ import annotations

import re
import time
from datetime import date

from pydantic import BaseModel, Field

from app.core.enums import Intent
from app.graph.state import ConversationState
from app.llm import get_llm, system, user
from app.logging import get_logger
from app.prompts import registry

log = get_logger(__name__)

# Cheap pre-extraction. A regex hit is more reliable than a model for a formatted
# identifier, and it gives the router a fallback when the LLM call fails.
# The body may contain internal separators: real numbers are segmented
# (CLM-2026-0004), and an earlier `[A-Z0-9]{4,16}` stopped at the first hyphen,
# so CLM-2026-0004 was looked up as "CLM-2026" and every segmented number came
# back "not found".
#
# The `(?=[A-Z0-9-]*\d)` lookahead requires the body to contain a digit. Without
# it the prefix alternation eats ordinary English - "claim number CLAIM-2026-0001"
# matched the word "claim" and captured "NUMBER" - and "POLICY" matched "POL"
# and captured "ICY". A claim number always has digits; a stray English word
# does not. `_first_group` strips any redundant repeated prefix.
CLAIM_PATTERN = re.compile(
    r"\b(?:CLM|CLAIM)[-/ ]?((?=[A-Z0-9-]*\d)[A-Z0-9]+(?:[-/][A-Z0-9]+){0,4})\b", re.I
)
POLICY_PATTERN = re.compile(
    r"\b(?:POL|POLICY)[-/ ]?((?=[A-Z0-9-]*\d)[A-Z0-9]+(?:[-/][A-Z0-9]+){0,4})\b", re.I
)

COVERAGE_MARKERS = (
    "cover",
    "covered",
    "coverage",
    "exclude",
    "excluded",
    "exclusion",
    "eligible",
    "claimable",
    "payable",
    "waiting period",
    "sub-limit",
    "sublimit",
)


class RouteOutput(BaseModel):
    intent: Intent = Field(description="Which lane should handle this message")
    claim_number: str | None = Field(default=None, description="Only if explicitly stated")
    policy_number: str | None = Field(default=None, description="Only if explicitly stated")
    product_name: str | None = Field(default=None, description="Insurance product, if named")
    date_of_loss: str | None = Field(
        default=None, description="ISO date of the incident, if the user refers to one"
    )
    is_coverage_question: bool = Field(
        default=False, description="True if asking whether something is covered"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


async def route_node(state: ConversationState) -> dict:
    started = time.perf_counter()
    question = state.get("standalone_question") or state["question"]

    regex_claim = _first_group(CLAIM_PATTERN, question)
    regex_policy = _first_group(POLICY_PATTERN, question)

    prompt = registry.load("router")
    try:
        parsed = await get_llm().structured(
            [system(prompt.body), user(question)], RouteOutput, temperature=0.0
        )
    except Exception as exc:
        log.warning("Router failed, falling back to heuristics", error=str(exc))
        parsed = _heuristic_route(question, regex_claim)

    # A regex hit on a formatted identifier beats the model's transcription of it.
    claim_number = regex_claim or parsed.claim_number
    policy_number = regex_policy or parsed.policy_number

    intent = parsed.intent
    # If a claim number is present the user is asking about that claim, whatever
    # the classifier decided.
    if claim_number and intent in (Intent.SMALLTALK, Intent.OUT_OF_SCOPE):
        intent = Intent.CLAIM_STATUS

    route = {
        "intent": intent,
        "claim_number": claim_number,
        "policy_number": policy_number,
        "product_name": parsed.product_name,
        "date_of_loss": _parse_date(parsed.date_of_loss),
        "is_coverage_question": parsed.is_coverage_question or _looks_like_coverage(question),
        "confidence": parsed.confidence,
    }

    log.debug(
        "Routed",
        intent=intent.value,
        has_claim=bool(claim_number),
        coverage=route["is_coverage_question"],
    )

    return {
        "route": route,
        "intent": intent,
        "timings_ms": {"route": int((time.perf_counter() - started) * 1000)},
    }


_REDUNDANT_PREFIX = re.compile(r"^(?:CLM|CLAIM|POL|POLICY)[-/]?")


def _first_group(pattern: re.Pattern[str], text: str) -> str | None:
    """Extract and canonicalise an identifier.

    Normalising here rather than at the call site means the DB lookup matches
    however the customer typed it: "claim clm/2026/0004." and "CLM-2026-0004"
    both resolve to the stored number.
    """
    match = pattern.search(text)
    if not match:
        return None

    body = match.group(1).upper().replace("/", "-").strip("-")
    # "my claim CLM-2026-0004" matches the word "claim" as the prefix and captures
    # "CLM-2026-0004" as the body, which would yield "CLM-CLM-2026-0004".
    body = _REDUNDANT_PREFIX.sub("", body).strip("-")
    if not body or not any(ch.isdigit() for ch in body):
        return None

    return f"CLM-{body}" if "CLM" in pattern.pattern else f"POL-{body}"


def _looks_like_coverage(question: str) -> bool:
    lowered = question.lower()
    return any(marker in lowered for marker in COVERAGE_MARKERS)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _heuristic_route(question: str, claim_number: str | None) -> RouteOutput:
    """Fallback when the LLM is unavailable.

    Deliberately conservative: without a model, route to document Q&A rather than
    guessing at a claim lookup.
    """
    if claim_number:
        return RouteOutput(intent=Intent.CLAIM_STATUS, claim_number=claim_number, confidence=0.4)
    return RouteOutput(
        intent=Intent.POLICY_QA,
        is_coverage_question=_looks_like_coverage(question),
        confidence=0.3,
    )
