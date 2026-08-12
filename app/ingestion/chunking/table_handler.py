"""Keeps tables intact; emits retrievable table summaries.

Benefit grids, sub-limit tables and waiting-period matrices carry the numbers a
customer is actually asking about. Two rules follow:

* **Never split a table across chunks.** Half a grid gives the wrong sub-limit,
  confidently.
* **When a table is too large to keep whole, repeat the header in every part.**
  A row group without its header is a list of numbers with no meaning.

The summary chunk exists because grids retrieve badly. "What are the room rent
sub-limits?" is prose; a markdown table of numbers shares almost no vocabulary
with it. A one-line summary is what the query matches, and it points at the grid.
"""

from __future__ import annotations

import re

from app.ingestion.chunking.base import count_tokens

MARKDOWN_ROW = re.compile(r"^\s*\|.*\|\s*$")
SEPARATOR_ROW = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def is_markdown_table(text: str) -> bool:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return len(lines) >= 2 and sum(bool(MARKDOWN_ROW.match(ln)) for ln in lines) >= 2


def split_table(text: str, *, max_tokens: int = 2000) -> list[str]:
    """Return the table whole, or in header-preserving row groups."""
    if count_tokens(text) <= max_tokens:
        return [text.strip()]

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not is_markdown_table(text):
        # Not a grid we can reason about - fall back to line groups rather than
        # cutting mid-row.
        return _group_lines(lines, max_tokens)

    header: list[str] = []
    body: list[str] = []
    for line in lines:
        if not body and (MARKDOWN_ROW.match(line) and len(header) < 2):
            header.append(line)
            if SEPARATOR_ROW.match(line):
                continue
        else:
            body.append(line)

    if not header:
        return _group_lines(lines, max_tokens)

    header_text = "\n".join(header)
    header_cost = count_tokens(header_text)
    budget = max(max_tokens - header_cost, 200)

    parts: list[str] = []
    current: list[str] = []
    total = 0

    for row in body:
        cost = count_tokens(row)
        if current and total + cost > budget:
            parts.append(f"{header_text}\n" + "\n".join(current))
            current, total = [], 0
        current.append(row)
        total += cost

    if current:
        parts.append(f"{header_text}\n" + "\n".join(current))

    return parts or [text.strip()]


def _group_lines(lines: list[str], max_tokens: int) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    total = 0
    for line in lines:
        cost = count_tokens(line)
        if current and total + cost > max_tokens:
            parts.append("\n".join(current))
            current, total = [], 0
        current.append(line)
        total += cost
    if current:
        parts.append("\n".join(current))
    return parts


def table_summary_text(table: str, section_path: str) -> str | None:
    """A deterministic, retrievable description of what the table contains.

    Deliberately not an LLM call: this runs once per table at ingest, and paying
    a model to describe a grid whose column headers already describe it is waste.
    ``CONTEXTUAL_PREFIX_ENABLED`` covers the LLM-written variant if evaluation
    shows it is worth the tokens.
    """
    lines = [ln.strip() for ln in table.splitlines() if ln.strip()]
    if not lines:
        return None

    headers = _header_cells(lines)
    row_count = sum(1 for ln in lines if MARKDOWN_ROW.match(ln) and not SEPARATOR_ROW.match(ln))
    row_count = max(row_count - 1, 0)  # discount the header row itself

    if not headers:
        return None

    label = section_path or "this section"
    return (
        f"Table in {label} with {row_count} row(s), listing "
        f"{', '.join(headers)}. Contains the specific limits, amounts and "
        f"conditions for {label}."
    )


def _header_cells(lines: list[str]) -> list[str]:
    for line in lines:
        if MARKDOWN_ROW.match(line) and not SEPARATOR_ROW.match(line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            return [c for c in cells if c][:8]
    return []
