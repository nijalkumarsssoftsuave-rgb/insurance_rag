"""Week 11 support drill, Track D: "the coverage answer that ignored an
exclusion" - found via scripts/plant_drill_fixture.py + a log search derived
from a vague complaint alone (eval/week11/RESULTS.md has the full drill:
8 candidates narrowed to 1, confirmed only afterward via `trace.planted`).

This is the permanent guard: a real production failure (planted, per this
week's brief - "practise finding a planted bad answer") becomes a test that
can never quietly pass again. Checked against the planted text itself before
being trusted as a regression test, the same discipline
`test_clause_reference_with_a_colon_is_accepted` (Week 10) and
`test_failed_audit_write_does_not_poison_the_session` (Week 11) both needed:
a check that has never been shown to fail on the thing it exists to catch is
not yet a check.
"""

from __future__ import annotations

import re

# The planted answer verbatim - scripts/plant_drill_fixture.py writes exactly
# this string to the `messages` table.
PLANTED_BAD_ANSWER = "Yes, dental treatment is covered under your Family Health Optima policy."

# Real answers this session's live API calls actually produced for the same
# question (captured verbatim from eval/week11/ and earlier live checks).
REAL_GOOD_ANSWERS = [
    "Dental treatment, dental surgery and orthodontic procedures of any kind are excluded "
    "from the scope of this Policy. This exclusion shall not apply where such treatment is "
    "necessitated by an accident and requires hospitalisation for at least twenty-four "
    "consecutive hours.",
    "Claim CLM-2026-0004 has been rejected. The date of loss is 2026-05-27, and the rejection "
    "was due to excluded treatment under clause 4.11, which states that dental treatment...",
]

_AFFIRMS_COVERAGE = re.compile(r"\b(is|are)\s+covered\b|\bcovers?\b", re.I)
_MENTIONS_EXCLUSION = re.compile(r"\bexclu\w*\b|\bclause\s*4\.11\b|\baccident\b", re.I)


def dental_coverage_answer_states_the_exclusion(answer: str) -> tuple[bool, str]:
    """A dental-coverage answer that affirms coverage must also name the
    exclusion (or the accident exception to it) - the specific, narrow check
    this week's planted failure earned, not a general exclusion-checker.

    Deliberately text-only, no LLM: the same "a regex decides this for free
    and never has an off day" reasoning `eval/assertions.py` already uses.
    """
    if not _AFFIRMS_COVERAGE.search(answer):
        return True, "does not affirm coverage - nothing to check"
    if _MENTIONS_EXCLUSION.search(answer):
        return True, ""
    return False, "affirms dental coverage with no mention of the exclusion"


def test_check_actually_fails_on_the_planted_answer():
    """Proven, not assumed: this is what makes it a regression test and not
    a check nobody has verified does anything."""
    ok, detail = dental_coverage_answer_states_the_exclusion(PLANTED_BAD_ANSWER)
    assert ok is False
    assert "exclusion" in detail


def test_check_passes_the_real_answers_this_session_actually_produced():
    for answer in REAL_GOOD_ANSWERS:
        ok, _detail = dental_coverage_answer_states_the_exclusion(answer)
        assert ok is True, f"false positive on a real, correct answer: {answer[:80]!r}"


def test_check_does_not_fire_on_an_unrelated_or_negative_answer():
    ok, _ = dental_coverage_answer_states_the_exclusion(
        "Dental treatment is not covered under this policy."
    )
    assert ok is True  # negative claim - no "is covered" to gate on

    ok, _ = dental_coverage_answer_states_the_exclusion(
        "The waiting period for pre-existing diseases is thirty-six months."
    )
    assert ok is True  # unrelated question entirely
