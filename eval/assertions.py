"""Deterministic checks on a claim summary. No model involved, no API cost, no bad days.

Anything a rule can decide belongs here rather than in the judge. Paying a model to
decide whether an amount is numeric is slower, costs money, and is *less* reliable
than a regex - and every criterion moved out of the judge prompt is one fewer thing
the judge can disagree with a human about.

Each check returns (passed, detail). `detail` is what a failure looks like, so a
failing case can be read without re-running anything.

**Format note.** The task brief specifies claim numbers as `CLM-YYYY-NNNNN`
(five digits). The live data is `CLM-2026-0001` - four. `CLAIM_NUMBER_RE` matches
what the system actually issues; asserting the brief's shape would fail every real
claim and prove nothing. Deviation recorded here rather than silently resolved.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from typing import Any

# Canonical issued form: CLM, four-digit year, four-digit sequence.
CLAIM_NUMBER_RE = re.compile(r"\bCLM-(\d{4})-(\d{4})\b")

# A repudiation stated in prose. Deliberately narrow: "not covered" and "excluded"
# are the phrasings that carry payout consequences, and a summary that merely
# quotes a waiting period is not stating a denial.
DENIAL_RE = re.compile(
    r"\b(rejected|repudiat\w*|denied|declined|not\s+payable|not\s+covered|excluded)\b", re.I
)

# A policy clause reference. The keyword is REQUIRED, because a bare `\d+\.\d+`
# also matches the "0.00" inside "the approved amount is 0.00 INR" - which let a
# denial with no clause reference at all pass this check, the exact failure the
# assertion exists to catch.
CLAUSE_REF_RE = re.compile(
    r"\b(?:clause|section|cl\.)\s*(?:reference|ref|no|number)?\.?\s*(\d{1,2}(?:\.\d{1,2})?)\b",
    re.I,
)


def _iso(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def claim_number_echoed(summary: str, record: dict) -> tuple[bool, str]:
    """The claim number appears, in canonical form, and is the *right* one.

    Two failures hide here, not one: omitting the number, and echoing a
    well-formed number belonging to a different claim.
    """
    expected = str(record.get("claim_number") or "")
    found = CLAIM_NUMBER_RE.findall(summary)
    if not found:
        return False, "no CLM-YYYY-NNNN claim number in the summary"
    rendered = {f"CLM-{y}-{n}" for y, n in found}
    if expected and expected not in rendered:
        return False, f"summary cites {sorted(rendered)}, record is {expected}"
    return True, ""


def date_of_loss_present(summary: str, record: dict) -> tuple[bool, str]:
    """The date of loss is stated and parses to the record's date.

    Accepts any rendering a human would - 27 May 2026, 2026-05-27, 27/05/2026 -
    because the check is on the *fact*, not the format.
    """
    expected = _iso(record.get("date_of_loss"))
    if expected is None:
        return True, "record has no date of loss to echo"
    day, month, year = expected.day, expected.strftime("%b").lower(), expected.year
    text = summary.lower()
    if expected.isoformat() in text:
        return True, ""
    if str(year) in text and month in text and re.search(rf"\b0?{day}\b", text):
        return True, ""
    if re.search(rf"\b0?{day}[/-]0?{expected.month}[/-]{year}\b", text):
        return True, ""
    return False, f"date of loss {expected.isoformat()} not stated"


def amounts_numeric(summary: str, record: dict) -> tuple[bool, str]:
    """Any money the summary states is a parseable figure, not a word.

    Both orders occur in real output - "INR 62,800.00" and "62,800.00 INR" - so
    the check is on the currency marker having a *number* on one side of it, not
    on which side. The first version of this only looked to the right and so
    failed every summary that wrote the currency last: an assertion that fires on
    correct output is worse than no assertion, because it trains you to ignore it.
    """
    # Must start with a digit: `[\d,]+` also matches a bare comma, which made
    # "...41,200.00 INR, and the approved..." parse the comma after INR as an amount.
    money = r"\d[\d,]*(?:\.\d{1,2})?"
    # Word-bounded: an unanchored `Rs\.?` matches the "rs" inside "reimbursement"
    # and the "rs." ending "assessors.", which failed two correct summaries.
    currency = r"(?:\bINR\b|\bRs\.?(?=\s|$)|₹)"
    # An amount written in words is the failure worth catching ("INR forty-one
    # thousand"), not a bare currency mention. "The claim is for reimbursement in
    # INR." states no figure at all and is fine; an earlier version failed it.
    number_word = (
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
        r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|"
        r"sixty|seventy|eighty|ninety|hundred|thousand|lakh|lakhs|crore|crores|million)"
    )
    for match in re.finditer(currency, summary, re.I):
        after = summary[match.end() : match.end() + 30]
        if re.match(rf"\s*{number_word}\b", after, re.I):
            return (
                False,
                f"amount written in words after a currency marker: {match.group()}{after[:26]!r}",
            )

    # Every figure the summary presents *as money* must be one of the record's
    # amounts. Matching on currency adjacency rather than on nearby words is what
    # keeps this honest: an earlier version looked for a figure within 40
    # characters of "claim" and duly read the 2026 out of CLM-2026-0001 as the
    # claimed amount, failing all four real summaries.
    on_record = {
        str(record[f]).split(".")[0].replace(",", "")
        for f in ("claimed_amount", "approved_amount", "settled_amount")
        if record.get(f) not in (None, "")
    }
    # The adjuster notes are an input too, so a figure quoted from them is
    # sourced, not invented. An earlier version failed a summary for reporting
    # "the initial estimate was Rs. 60,000" - which the notes plainly said. The
    # check is against everything the model was shown, not against the record alone.
    for entry in record.get("notes") or []:
        for token in re.findall(money, str(entry.get("note", ""))):
            on_record.add(token.split(".")[0].replace(",", ""))
    if on_record:
        stated = set(re.findall(rf"{currency}\s*({money})", summary, re.I)) | set(
            re.findall(rf"({money})\s*{currency}", summary, re.I)
        )
        for token in stated:
            if token.split(".")[0].replace(",", "") not in on_record:
                return False, f"amount {token} appears in the summary but not on the record"
    return True, ""


def denial_cites_clause(summary: str, record: dict) -> tuple[bool, str]:
    """A stated denial must carry a clause reference.

    The one assertion here with a payout behind it. An adjuster who repudiates
    without a citable clause cannot defend it at the Ombudsman, which is exactly
    the failure `app/graph/nodes/verify.py` was built to catch on the document
    lane and nothing was catching on the claim lane.
    """
    if not DENIAL_RE.search(summary):
        return True, "no denial stated"
    if CLAUSE_REF_RE.search(summary):
        return True, ""
    expected = record.get("rejection_clause_ref")
    hint = f" (record has clause {expected})" if expected else ""
    return False, f"denial stated with no clause reference{hint}"


# Name -> check. The eval reports these separately from the judged criterion, and
# the count of each is the assertion/judge split the week is graded on.
ASSERTIONS: dict[str, Callable[[str, dict], tuple[bool, str]]] = {
    "claim_number_echoed": claim_number_echoed,
    "date_of_loss_present": date_of_loss_present,
    "amounts_numeric": amounts_numeric,
    "denial_cites_clause": denial_cites_clause,
}


def run_all(summary: str, record: dict) -> dict[str, tuple[bool, str]]:
    return {name: check(summary, record) for name, check in ASSERTIONS.items()}
