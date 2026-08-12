"""PII detection, masking, and the reversible masking map.

Scope matters here. This masks what leaves the process - logs, traces, and (when
``PII_MASKING_ENABLED``) the prompt sent to a hosted LLM. It deliberately does
**not** mask policy or claim numbers: those are the identifiers the router must
extract to answer "where is my claim", and stripping them would break the feature
this system exists to provide. They are access-controlled, not secret.

Masking is reversible so an answer can be restored before it reaches the user who
supplied the data in the first place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class PIIKind(StrEnum):
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    PAN = "PAN"
    AADHAAR = "AADHAAR"
    CARD = "CARD"
    IFSC = "IFSC"
    ACCOUNT = "ACCOUNT"


# Order matters: the most specific pattern must win. Aadhaar and card numbers
# both look like digit runs, so card is matched first on its 16-digit shape.
_PATTERNS: tuple[tuple[PIIKind, re.Pattern[str]], ...] = (
    (PIIKind.EMAIL, re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    (PIIKind.PAN, re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    (PIIKind.IFSC, re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")),
    (PIIKind.CARD, re.compile(r"\b(?:\d[ -]?){15,18}\d\b")),
    (PIIKind.AADHAAR, re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    (PIIKind.PHONE, re.compile(r"(?<!\d)(?:\+91[-\s]?|0)?[6-9]\d{9}(?!\d)")),
    (
        PIIKind.ACCOUNT,
        re.compile(r"\b(?:a/c|acc(?:ount)?)\s*(?:no\.?|number)?\s*[:#]?\s*(\d{9,18})\b", re.I),
    ),
)


@dataclass(slots=True, frozen=True)
class PIIMatch:
    kind: PIIKind
    value: str
    start: int
    end: int


@dataclass(slots=True)
class MaskResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.mapping)

    def unmask(self, text: str) -> str:
        """Restore originals. Longest placeholder first, so PII_PHONE_10 is not
        clobbered by a prefix match on PII_PHONE_1."""
        for placeholder in sorted(self.mapping, key=len, reverse=True):
            text = text.replace(placeholder, self.mapping[placeholder])
        return text


def detect(text: str) -> list[PIIMatch]:
    """Non-overlapping matches, most specific pattern first."""
    matches: list[PIIMatch] = []
    claimed: list[tuple[int, int]] = []

    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.span()
            if any(start < c_end and end > c_start for c_start, c_end in claimed):
                continue
            claimed.append((start, end))
            matches.append(PIIMatch(kind=kind, value=m.group(0), start=start, end=end))

    return sorted(matches, key=lambda x: x.start)


def mask(text: str) -> MaskResult:
    """Replace PII with stable placeholders.

    The same value always gets the same placeholder within one call, so a model
    reading the masked text can still tell that two mentions refer to one person.
    """
    matches = detect(text)
    if not matches:
        return MaskResult(text=text)

    mapping: dict[str, str] = {}
    assigned: dict[str, str] = {}
    counters: dict[PIIKind, int] = {}
    out: list[str] = []
    cursor = 0

    for match in matches:
        placeholder = assigned.get(match.value)
        if placeholder is None:
            counters[match.kind] = counters.get(match.kind, 0) + 1
            placeholder = f"[{match.kind}_{counters[match.kind]}]"
            assigned[match.value] = placeholder
            mapping[placeholder] = match.value

        out.append(text[cursor : match.start])
        out.append(placeholder)
        cursor = match.end

    out.append(text[cursor:])
    return MaskResult(text="".join(out), mapping=mapping)


def redact(text: str) -> str:
    """One-way masking, for logs where nothing will ever need restoring."""
    return mask(text).text
