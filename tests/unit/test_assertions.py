"""The deterministic half of the claim-summary eval.

These exist because two of these checks shipped broken and passed review by
eye. Both failed on *correct* output, which is the dangerous direction: an
assertion that cries wolf on good summaries gets ignored, and then it is not
checking anything at all.
"""

from __future__ import annotations

from eval.assertions import ASSERTIONS, run_all

RECORD = {
    "claim_number": "CLM-2026-0004",
    "date_of_loss": "2026-05-27",
    "claimed_amount": "41200.00",
    "approved_amount": "0.00",
    "rejection_clause_ref": "4.11",
}

# Verbatim from the summariser on the real CLM-2026-0004 record.
REAL = (
    "Claim number CLM-2026-0004 is currently in a rejected status, with a date of loss on "
    "2026-05-27. The claimed amount is 41,200.00 INR, and the approved amount is 0.00 INR. "
    "The claim was rejected due to excluded treatment, with the policy clause reference 4.11."
)


def failures(summary: str, record: dict = RECORD) -> list[str]:
    return sorted(name for name, (ok, _) in run_all(summary, record).items() if not ok)


def test_real_summary_passes_every_assertion() -> None:
    """The regression that matters most: correct output must not trip anything."""
    assert failures(REAL) == []


def test_currency_written_after_the_amount_is_accepted() -> None:
    """ "41,200.00 INR" is as valid as "INR 41,200.00"; the first version failed it."""
    assert "amounts_numeric" not in failures(
        "Claim CLM-2026-0004 rejected under clause 4.11 on 27 May 2026, claimed 41,200.00 INR."
    )


def test_comma_after_a_currency_marker_is_not_an_amount() -> None:
    """ "...41,200.00 INR, and the approved..." once parsed the comma as a figure."""
    assert "amounts_numeric" not in failures(REAL)


def test_denial_without_a_clause_is_caught() -> None:
    assert failures(
        "Claim CLM-2026-0004 was denied. Date of loss 27 May 2026. Claimed INR 41,200.00."
    ) == ["denial_cites_clause"]


def test_amount_not_on_the_record_is_caught() -> None:
    assert failures(
        "Claim CLM-2026-0004 rejected under clause 4.11. Date of loss 27 May 2026. "
        "Claimed INR 51,200.00."
    ) == ["amounts_numeric"]


def test_another_claims_number_is_caught() -> None:
    assert failures(
        "Claim CLM-2026-0009 rejected under clause 4.11. Date of loss 27 May 2026."
    ) == ["claim_number_echoed"]


def test_missing_date_of_loss_is_caught() -> None:
    assert failures("Claim CLM-2026-0004 rejected under clause 4.11.") == ["date_of_loss_present"]


def test_a_summary_stating_no_denial_skips_the_clause_check() -> None:
    record = {"claim_number": "CLM-2026-0002", "date_of_loss": "2026-07-18"}
    assert failures("Claim CLM-2026-0002 is under review, date of loss 18 July 2026.", record) == []


def test_assertion_count_is_reported_for_the_split() -> None:
    """The week is graded on assertions-vs-judged-criteria; the count must be real."""
    assert len(ASSERTIONS) == 4
