"""Input guard: PII masking, injection heuristics, rate limiting.

The safety floor. It masks PII so it never reaches logs or traces, and it scores
the input for injection attempts.

It blocks only on HIGH severity, and only for authorization-bypass style attempts.
Over-blocking is its own failure: a customer writing "ignore the previous claim
number I gave you" is making a legitimate correction, and refusing them teaches
them the assistant is broken.
"""

from __future__ import annotations

import time

from app.config import settings
from app.graph.state import ConversationState
from app.logging import get_logger
from app.security import injection, pii

log = get_logger(__name__)

MAX_QUESTION_CHARS = 4_000

BLOCK_MESSAGE = (
    "I can only help with questions about your policy documents and claims. "
    "If you need something changed on a claim, I can connect you to an agent."
)


async def guard_node(state: ConversationState) -> dict:
    started = time.perf_counter()
    question = (state.get("question") or "").strip()

    if not question:
        return {
            "blocked": True,
            "block_reason": "empty",
            "answer": "Could you tell me what you'd like to know?",
            "timings_ms": {"guard": 0},
        }

    # Truncate rather than reject: a long paste is usually a customer pasting
    # their whole policy email, not an attack.
    if len(question) > MAX_QUESTION_CHARS:
        question = question[:MAX_QUESTION_CHARS]

    masked = (
        pii.mask(question) if settings.security.pii_masking_enabled else pii.MaskResult(question)
    )

    verdict = (
        injection.scan_user_input(question)
        if settings.security.injection_guard_enabled
        else injection.InjectionVerdict(injection.Severity.NONE, [])
    )

    if verdict.suspicious:
        log.warning(
            "Injection signals in user input",
            severity=verdict.severity.value,
            signals=verdict.signals,
            question=masked.text[:200],
        )

    # Only an explicit attempt to make the assistant act on a claim is worth
    # refusing outright. Everything else is logged and allowed through.
    if verdict.should_block and "authorization_bypass" in verdict.signals:
        return {
            "question": question,
            "masked_question": masked.text,
            "pii_mapping": masked.mapping,
            "injection_severity": verdict.severity.value,
            "blocked": True,
            "block_reason": "authorization_bypass",
            "answer": BLOCK_MESSAGE,
            "needs_human": True,
            "timings_ms": {"guard": int((time.perf_counter() - started) * 1000)},
        }

    return {
        "question": question,
        "masked_question": masked.text,
        "pii_mapping": masked.mapping,
        "injection_severity": verdict.severity.value,
        "blocked": False,
        "timings_ms": {"guard": int((time.perf_counter() - started) * 1000)},
    }
