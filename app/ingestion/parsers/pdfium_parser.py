"""Fast path for single-column text-only PDFs (pypdfium2).

Apache-2.0/BSD-3, no ML models, no download, ~50x faster than Docling. It has no
layout model, so it infers headings from typography-free heuristics: numbering
patterns, capitalisation and line length. That is enough for well-formed policy
wordings whose clauses are numbered, and hopeless for scanned or multi-column
documents - which is why ``pipeline.py`` routes rather than committing to one.

Deliberately not PyMuPDF: excellent library, AGPL-3.0 licence, and the network
clause reaches a hosted product (ARCHITECTURE 15.2).
"""

from __future__ import annotations

import re
from pathlib import Path

from app.ingestion.parsers.base import (
    Block,
    BlockType,
    ParsedDocument,
    ParserError,
    infer_heading_level,
)
from app.logging import get_logger

log = get_logger(__name__)

SUPPORTED = {".pdf"}

# A heading in a policy wording almost always starts with clause numbering or a
# recognised structural word. Matching on that is far more reliable than trying
# to infer emphasis without font data.
CLAUSE_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[.)]?\s+(\S.*)$")
STRUCTURAL_WORD = re.compile(
    r"^\s*(SECTION|PART|CLAUSE|CHAPTER|SCHEDULE|ANNEXURE|APPENDIX|"
    r"DEFINITIONS?|EXCLUSIONS?|BENEFITS?|CONDITIONS?|WAITING\s+PERIODS?|"
    r"WHAT\s+IS\s+(?:NOT\s+)?COVERED|GENERAL\s+TERMS)\b",
    re.I,
)
# Short, mostly-uppercase lines that end without punctuation read as headings.
SHOUTY = re.compile(r"^[A-Z][A-Z0-9 &/,'\-()]{3,60}$")

# Detecting a grid this parser cannot reconstruct.
#
# The signal is line *length*, not numeric density. Prose fills to the right
# margin, so body text arrives as a run of long lines. Table cells wrap inside
# their column, so a grid arrives as a run of short ones ("per day", "Day 31",
# "Rs. 40,000 per eye"). Counting numbers per line looked obvious and was wrong
# twice: ordinary clauses carry numbers too ("within 30 days of 1 April 2024"),
# and a row's numbers often land on a different wrapped line from its label.
#
# This MUST run on raw lines. `_join_wrapped` later concatenates the grid into a
# single paragraph and the structure is gone.
# A cell line is one materially shorter than the document's own prose.
SHORT_LINE_RATIO = 0.55
TABLE_RUN_MIN_LINES = 4
HAS_DIGIT = re.compile(r"\d")


class PdfiumParser:
    """Text + heuristic structure. No models, no network."""

    @property
    def name(self) -> str:
        return "pypdfium2"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED

    def parse(self, path: Path) -> ParsedDocument:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ParserError("pypdfium2 is not installed") from exc

        try:
            pdf = pdfium.PdfDocument(str(path))
        except Exception as exc:
            raise ParserError(f"Could not open {path.name}: {exc}") from exc

        blocks: list[Block] = []
        warnings: list[str] = []
        raw_lines: list[str] = []

        try:
            page_count = len(pdf)
            for page_no in range(page_count):
                page = pdf[page_no]
                try:
                    raw = page.get_textpage().get_text_range()
                finally:
                    page.close()
                raw_lines.extend(raw.splitlines())
                blocks.extend(_blocks_from_page(raw, page_no + 1))
        finally:
            pdf.close()

        likely_tables = _looks_tabular(raw_lines)
        if likely_tables:
            warnings.append("Row-shaped content detected but this parser has no column model.")

        if not blocks:
            # Almost certainly a scanned PDF. Say so rather than indexing nothing.
            warnings.append("No extractable text - this is likely a scanned PDF and needs OCR.")

        return ParsedDocument(
            blocks=blocks,
            page_count=page_count,
            title=_guess_title(blocks),
            parser=self.name,
            warnings=warnings,
            likely_tables=likely_tables,
        )


def _looks_tabular(raw_lines: list[str]) -> bool:
    """Did this document contain a grid we just flattened?

    Measures line length *relative to this document's own median*, not against a
    fixed character count. Margins, page size and font all shift where prose
    wraps, so an absolute threshold that fits an A4 policy wording misfires on
    anything laid out differently - it flagged a prose-only control document as
    tabular. A run of lines far shorter than the document's typical line is the
    signal that survives those differences.
    """
    lengths = [len(line.strip()) for line in raw_lines if line.strip()]
    if len(lengths) < 10:
        return False

    ordered = sorted(lengths)
    median = ordered[len(ordered) // 2]
    if median < 40:
        # The whole document is short lines - a form or a slide, not a grid we
        # can distinguish.
        return False

    cutoff = median * SHORT_LINE_RATIO
    run = digits_in_run = best = best_digits = 0

    for line in raw_lines:
        stripped = line.strip()
        if stripped and len(stripped) <= cutoff:
            run += 1
            if HAS_DIGIT.search(stripped):
                digits_in_run += 1
            if run > best:
                best, best_digits = run, digits_in_run
        else:
            run = digits_in_run = 0

    # A run of short lines alone could be a bullet list. Requiring digits inside
    # it is what distinguishes a benefit grid.
    return best >= TABLE_RUN_MIN_LINES and best_digits >= 2


def _blocks_from_page(raw: str, page_no: int) -> list[Block]:
    if not raw or not raw.strip():
        return []

    blocks: list[Block] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            text = _join_wrapped(buffer)
            if text.strip():
                blocks.append(Block(text=text, type=BlockType.PARAGRAPH, page_no=page_no))
            buffer.clear()

    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            flush()
            continue

        level = _heading_level(stripped)
        if level is not None:
            flush()
            blocks.append(
                Block(
                    text=stripped,
                    type=BlockType.HEADING,
                    page_no=page_no,
                    level=level,
                )
            )
            continue

        buffer.append(stripped)

    flush()
    return blocks


def _heading_level(line: str) -> int | None:
    """Depth from clause numbering, else 1 for structural words. None if body."""
    if len(line) > 120:  # a long line is prose, whatever it starts with
        return None

    if match := CLAUSE_NUMBER.match(line):
        _, remainder = match.groups()
        # "4.11 Dental treatment is excluded unless..." is a numbered *clause*,
        # not a heading. Treat it as a heading only when the remainder is short
        # enough to be a title.
        if len(remainder) <= 80 and not remainder.endswith((".", ";", ",")):
            return infer_heading_level(line) or 2
        return None

    if STRUCTURAL_WORD.match(line):
        return infer_heading_level(line) or 1

    if SHOUTY.match(line) and not line.endswith((".", ",", ";")):
        return 2

    return None


def _join_wrapped(lines: list[str]) -> str:
    """Rejoin PDF hard-wrapping into sentences.

    PDFs break lines at the page margin, so raw extraction yields fragments. Left
    as-is, the chunker would treat each visual line as its own paragraph and
    destroy clause continuity.
    """
    out: list[str] = []
    for line in lines:
        if out and not out[-1].endswith(("-",)) and not _ends_sentence(out[-1]):
            out[-1] = f"{out[-1]} {line}"
        elif out and out[-1].endswith("-"):
            out[-1] = out[-1][:-1] + line  # de-hyphenate across the break
        else:
            out.append(line)
    return "\n".join(out)


def _ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((".", ":", ";", "?", "!"))


def _guess_title(blocks: list[Block]) -> str | None:
    for block in blocks[:10]:
        if block.is_heading and 10 <= len(block.text) <= 120:
            return block.text.strip()
    return None
