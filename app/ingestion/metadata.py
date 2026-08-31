"""Document-level metadata: insurer, UIN, effective dates, doc type.

Metadata is what makes retrieval *filterable*, and filtering is what makes it
*correct* - the effective-date window decides which wording applies to a claim
(ARCHITECTURE 5.3).

Regex first, LLM second. Identifiers like a UIN have a fixed shape, and a regex
reads them more reliably than a model transcribes them. The LLM is asked only
for the things that genuinely need reading comprehension, and only over the
first couple of pages.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import date, datetime

from pydantic import BaseModel, Field

from app.core.enums import DocType
from app.ingestion.parsers.base import ParsedDocument
from app.logging import get_logger

log = get_logger(__name__)

# How much of the document the LLM sees. Insurer, product and UIN live on the
# cover page; sending forty pages would cost tokens to learn nothing more.
HEAD_CHARS = 6_000

# IRDAI Unique Identification Number, e.g. SHAHLIP23024V072223
UIN_PATTERN = re.compile(r"\b(?:UIN[\s:.-]*)?([A-Z]{3,5}[A-Z]{3,6}\d{2}\d{3}V\d{6})\b")
UIN_LOOSE = re.compile(r"\bUIN\s*[:.-]?\s*([A-Z0-9]{10,25})\b", re.I)

DATE_PATTERNS = (
    re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"),
    re.compile(
        r"\b(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{4})\b",
        re.I,
    ),
)

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}

DOC_TYPE_HINTS: tuple[tuple[DocType, re.Pattern[str]], ...] = (
    (DocType.ENDORSEMENT, re.compile(r"\bendorsement\b", re.I)),
    (DocType.CLAIM_FORM, re.compile(r"\bclaim\s+form\b", re.I)),
    # "IRDAI" used to appear here and was actively harmful: every Indian
    # insurance document prints an IRDAI registration number on its cover by
    # regulation, so it matched policy wordings as often as circulars - and
    # because CIRCULAR is tested before POLICY_WORDING, it won. A misfiled
    # wording then had its product name stripped by the product-neutral rule
    # below and stopped being filterable by product.
    (DocType.CIRCULAR, re.compile(r"\bcircular\b", re.I)),
    (DocType.SOP, re.compile(r"\bstandard\s+operating\b|\bSOP\b", re.I)),
    (DocType.BROCHURE, re.compile(r"\bbrochure\b|\bprospectus\b", re.I)),
    (
        DocType.POLICY_WORDING,
        re.compile(r"\bpolicy\s+wording|\bterms\s+and\s+conditions\b|\bexclusions?\b", re.I),
    ),
)


@dataclass(slots=True)
class DocumentMetadata:
    insurer: str | None = None
    product_name: str | None = None
    uin: str | None = None
    doc_type: DocType | None = None
    language: str = "en"
    effective_from: date | None = None
    effective_to: date | None = None
    confidence: float = 0.0


class _LLMMetadata(BaseModel):
    insurer: str | None = Field(default=None, description="Insurance company name")
    product_name: str | None = Field(default=None, description="Product or plan name")
    uin: str | None = Field(default=None, description="UIN, only if explicitly printed")
    doc_type: DocType = Field(default=DocType.OTHER)
    effective_from: str | None = Field(default=None, description="ISO date if stated")
    effective_to: str | None = Field(default=None, description="ISO date if stated")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


PROMPT = (
    "You extract catalogue metadata from insurance documents.\n\n"
    "Rules:\n"
    "- Extract only what is explicitly printed. Never infer or invent.\n"
    "- Leave a field null if the document does not state it.\n"
    "- Dates must be ISO (YYYY-MM-DD).\n"
    "- `product_name` is the name of the insurance plan or cover, such as\n"
    "  'Family Health Optima' or 'Motor Shield Private Car'.\n"
    "- `product_name` is NEVER a document type. 'Policy Wording', 'Endorsement',\n"
    "  'Circular', 'Prospectus' and 'Standard Operating Procedure' describe what\n"
    "  the document IS, not which product it covers. If the text only gives a\n"
    "  document type, return null for `product_name`.\n"
    "- `product_name` is never the insurer's name either.\n"
    "- An internal document that governs process rather than a single product\n"
    "  (an SOP, a circular) usually has no product name. Return null."
)


def extract_metadata(
    parsed: ParsedDocument, *, filename: str = "", use_llm: bool = True
) -> DocumentMetadata:
    """Regex pass, then optionally an LLM pass to fill the gaps."""
    head = parsed.text[:HEAD_CHARS]
    effective_from, effective_to = _find_effective_window(head)
    meta = DocumentMetadata(
        uin=_find_uin(head),
        doc_type=_guess_doc_type(f"{filename}\n{head}"),
        effective_from=effective_from,
        effective_to=effective_to,
        confidence=0.3,
    )

    if not use_llm:
        return meta

    try:
        llm_meta = asyncio.run(_extract_with_llm(head, filename))
    except Exception as exc:
        # Metadata is an enrichment, not a gate. A document with no product name
        # is still searchable; a failed ingest is not.
        log.warning("LLM metadata extraction failed", error=str(exc))
        return meta

    meta.insurer = _clean(llm_meta.insurer)
    meta.product_name = _clean(llm_meta.product_name)
    meta.uin = meta.uin or _clean(llm_meta.uin)
    # Regex wins on dates. The effective window decides which wording governs a
    # claim, so a transcription slip here answers an old claim from the wrong
    # policy version - the exact failure ARCHITECTURE 5.3 exists to prevent.
    meta.effective_from = meta.effective_from or _parse_iso(llm_meta.effective_from)
    meta.effective_to = meta.effective_to or _parse_iso(llm_meta.effective_to)
    meta.confidence = llm_meta.confidence
    if meta.doc_type in (None, DocType.OTHER):
        meta.doc_type = llm_meta.doc_type

    # Process documents govern how claims are handled, not one product. They
    # frequently mention a product in passing - a claims circular citing the
    # health wording - and the extractor then attributes the whole document to
    # it. Tagging them would hide the settlement-timeline circular from a motor
    # claim question, so they stay product-neutral and remain retrievable for
    # every product.
    if meta.doc_type in (DocType.SOP, DocType.CIRCULAR):
        meta.product_name = None

    log.debug(
        "Extracted metadata",
        insurer=meta.insurer,
        product=meta.product_name,
        uin=meta.uin,
        doc_type=meta.doc_type.value if meta.doc_type else None,
    )
    return meta


async def _extract_with_llm(head: str, filename: str = "") -> _LLMMetadata:
    """Ask for the fields that need reading comprehension.

    The filename is included because policy wordings routinely print their
    document type on the cover ("Document type: Policy Wording") without ever
    repeating the plan name, while the file is called
    ``acme_family_health_optima_wording_v3.2.pdf``. Given only the cover text the
    model returned "Policy Wording" as the product - which would collapse every
    wording in the corpus onto one product name.
    """
    from app.llm import get_llm, system, user

    body = f"Filename: {filename}\n\n{head}" if filename else head
    return await get_llm().structured([system(PROMPT), user(body)], _LLMMetadata, temperature=0.0)


"""Effective window, stated on the cover page of most wordings."""
_EFFECTIVE_WINDOW = re.compile(
    r"(?:effective|valid|in\s+force|period\s+of\s+insurance)\s*"
    r"(?:from|:)?\s*(?P<start>[\d]{1,2}[/-][\d]{1,2}[/-][\d]{4}|[\d]{1,2}\s+\w{3,9}\.?\s+[\d]{4})"
    r"\s*(?:to|-|until|till|through)\s*"
    r"(?P<end>[\d]{1,2}[/-][\d]{1,2}[/-][\d]{4}|[\d]{1,2}\s+\w{3,9}\.?\s+[\d]{4})",
    re.I,
)
_EFFECTIVE_FROM_ONLY = re.compile(
    r"(?:effective|valid|in\s+force)\s*(?:from|:)?\s*"
    r"(?P<start>[\d]{1,2}[/-][\d]{1,2}[/-][\d]{4}|[\d]{1,2}\s+\w{3,9}\.?\s+[\d]{4})",
    re.I,
)


def _find_effective_window(text: str) -> tuple[date | None, date | None]:
    """Read the effective dates without an LLM.

    Deterministic and free, which matters twice over: the dates are the control
    that stops an old claim being answered from a newer wording, and relying on
    a model for them would make that control depend on an API key being present.
    Dates are read as DD/MM/YYYY - these are Indian policy documents.
    """
    if match := _EFFECTIVE_WINDOW.search(text):
        return _parse_loose(match.group("start")), _parse_loose(match.group("end"))
    if match := _EFFECTIVE_FROM_ONLY.search(text):
        return _parse_loose(match.group("start")), None
    return None, None


def _find_uin(text: str) -> str | None:
    for pattern in (UIN_PATTERN, UIN_LOOSE):
        if match := pattern.search(text):
            return match.group(1).upper()
    return None


def _guess_doc_type(text: str) -> DocType | None:
    for doc_type, pattern in DOC_TYPE_HINTS:
        if pattern.search(text):
            return doc_type
    return None


def _parse_iso(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return _parse_loose(value)


def _parse_loose(value: str) -> date | None:
    for pattern in DATE_PATTERNS:
        if match := pattern.search(value):
            groups = match.groups()
            try:
                if groups[1].isdigit():
                    day, month, year = (int(g) for g in groups)
                else:
                    day = int(groups[0])
                    month = _MONTHS[groups[1][:3].lower()]
                    year = int(groups[2])
                return datetime(year, month, day).date()
            except (ValueError, KeyError):
                continue
    return None


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = " ".join(value.split()).strip(" .,-")
    return cleaned[:255] or None
