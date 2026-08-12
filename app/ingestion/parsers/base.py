"""DocumentParser protocol returning a normalized document tree.

Every parser flattens its own output into the same ordered list of ``Block``s.
The chunker in ``app/ingestion/chunking`` never learns which parser produced a
document, so swapping Docling for pypdfium2 (or adding an OCR path later) does
not touch chunking logic.

The tree is deliberately *flat with levels* rather than nested. Policy documents
nest inconsistently - "4.11" may be a heading in one wording and a bold run of
body text in another - so a strict tree forces guesses the flat form doesn't.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

# Shared heading-depth inference. Both parsers use it, for different reasons:
# pypdfium2 has no font data at all, and Docling's layout model labels nearly
# every heading "section_header" (level 2) regardless of nesting. Either way the
# document's own clause numbering is the most reliable depth signal available,
# and depth is what produces "Section 4 > Exclusions > 4.11 Dental" rather than
# a flat list of unrelated headings.
_LEADING_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[.)]?\s+\S")
_TOP_LEVEL_WORD = re.compile(r"^\s*(SECTION|PART|CHAPTER|SCHEDULE|ANNEXURE|APPENDIX)\b", re.I)
_NAMED_SECTION = re.compile(
    r"^\s*(DEFINITIONS?|EXCLUSIONS?|BENEFITS?|CONDITIONS?|WAITING\s+PERIODS?|"
    r"CLAIMS?\s+PROCEDURE|WHAT\s+IS\s+(?:NOT\s+)?COVERED|GENERAL\s+TERMS)\b",
    re.I,
)


def infer_heading_level(text: str) -> int | None:
    """Depth from the heading's own text, or None if it cannot be told.

    "SECTION 4 - EXCLUSIONS" -> 1, "3.2 Room Rent" -> 2, "4.11.3 Scaling" -> 3.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > 120:
        return None
    if _TOP_LEVEL_WORD.match(stripped) or _NAMED_SECTION.match(stripped):
        return 1
    if match := _LEADING_NUMBER.match(stripped):
        # "4" -> 1, "4.11" -> 2, "4.11.3" -> 3. Dots + 1, so a numbered clause
        # nests directly under the "SECTION 4" heading above it.
        return min(match.group(1).count(".") + 1, 6)
    return None


class BlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST_ITEM = "list_item"
    CAPTION = "caption"


@dataclass(slots=True)
class Block:
    """One unit of extracted content, in reading order."""

    text: str
    type: BlockType = BlockType.PARAGRAPH
    page_no: int | None = None
    # Heading depth, 1 = outermost. None for non-headings.
    level: int | None = None

    @property
    def is_heading(self) -> bool:
        return self.type is BlockType.HEADING

    @property
    def is_table(self) -> bool:
        return self.type is BlockType.TABLE


@dataclass(slots=True)
class ParsedDocument:
    """Parser output. Ordered blocks plus what we learned about the file."""

    blocks: list[Block] = field(default_factory=list)
    page_count: int = 0
    title: str | None = None
    parser: str = "unknown"
    # Set when a parser had to fall back or degrade - surfaced in the UI so a
    # badly-extracted document is visible rather than silently poor.
    warnings: list[str] = field(default_factory=list)
    # Raised by a parser that has no table model when it sees row-shaped text it
    # cannot reconstruct. The pipeline reads this to decide whether to re-parse
    # with Docling: a flattened benefit grid gives customers wrong sub-limits,
    # which is worth the extra seconds.
    likely_tables: bool = False

    @property
    def text(self) -> str:
        return "\n\n".join(b.text for b in self.blocks if b.text.strip())

    @property
    def char_count(self) -> int:
        return sum(len(b.text) for b in self.blocks)

    @property
    def heading_count(self) -> int:
        return sum(1 for b in self.blocks if b.is_heading)

    @property
    def table_count(self) -> int:
        return sum(1 for b in self.blocks if b.is_table)


class ParserError(RuntimeError):
    """The file could not be parsed at all."""


@runtime_checkable
class DocumentParser(Protocol):
    """Implementations: ``DoclingParser``, ``PdfiumParser``."""

    @property
    def name(self) -> str: ...

    def supports(self, path: Path) -> bool: ...

    def parse(self, path: Path) -> ParsedDocument: ...
