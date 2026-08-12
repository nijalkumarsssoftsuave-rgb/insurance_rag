"""Structure-aware chunking: sections, parent/child linkage, tables, breadcrumbs."""

from __future__ import annotations

import pytest

from app.core.enums import ChunkKind
from app.ingestion.chunking import StructureAwareChunker, build_sections
from app.ingestion.chunking.table_handler import (
    is_markdown_table,
    split_table,
    table_summary_text,
)
from app.ingestion.parsers.base import (
    Block,
    BlockType,
    ParsedDocument,
    infer_heading_level,
)

DENTAL = (
    "Dental treatment, dental surgery and orthodontic procedures of any kind are "
    "excluded from the scope of this policy. This exclusion shall not apply where "
    "such treatment is necessitated by an accident and requires hospitalisation for "
    "at least twenty-four consecutive hours. Routine dental check-ups, scaling, "
    "polishing and cosmetic dentistry are never payable under any circumstances. "
    "The Insured Person must submit the treating dentist report within thirty days "
    "of discharge from the hospital or the claim shall be repudiated."
)

TABLE = (
    "| Benefit | Sub-limit | Waiting Period |\n"
    "|---|---|---|\n"
    "| Room rent | 1% of SI per day | 30 days |\n"
    "| ICU charges | 2% of SI per day | 30 days |"
)


def doc(*blocks: Block) -> ParsedDocument:
    return ParsedDocument(blocks=list(blocks), page_count=1, parser="test")


def policy_doc() -> ParsedDocument:
    return doc(
        Block("SECTION 4 - EXCLUSIONS", BlockType.HEADING, 12, 1),
        Block("The Company shall not be liable for the following.", BlockType.PARAGRAPH, 12),
        Block("4.11 Dental Treatment", BlockType.HEADING, 12, 2),
        Block(DENTAL, BlockType.PARAGRAPH, 12),
        Block(TABLE, BlockType.TABLE, 13),
    )


# ─────────────────────────────────────────────────── heading depth


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SECTION 4 - EXCLUSIONS", 1),
        ("PART II", 1),
        ("Definitions", 1),
        ("WAITING PERIODS", 1),
        ("3.2 Room Rent and Boarding", 2),
        ("4.11.3 Scaling", 3),
        ("Some ordinary sentence about nothing", None),
    ],
)
def test_heading_depth_comes_from_clause_numbering(text: str, expected: int | None) -> None:
    """Both parsers rely on this: Docling labels nearly every heading the same
    level, and pypdfium2 has no font data at all."""
    assert infer_heading_level(text) == expected


# ───────────────────────────────────────────────────────── sections


def test_sections_nest_into_a_breadcrumb() -> None:
    sections = build_sections(policy_doc())
    paths = [s.breadcrumb for s in sections]
    assert "SECTION 4 - EXCLUSIONS" in paths
    assert "SECTION 4 - EXCLUSIONS > 4.11 Dental Treatment" in paths


def test_document_without_headings_still_chunks() -> None:
    """A flat document must not silently produce zero chunks."""
    sections = build_sections(doc(Block("Just some prose.", BlockType.PARAGRAPH)))
    assert len(sections) == 1
    assert sections[0].blocks


# ────────────────────────────────────────────── parent/child chunks


def test_parents_and_children_are_linked() -> None:
    out = StructureAwareChunker().chunk(policy_doc(), breadcrumb="Acme | FHO")
    assert out.parents and out.children
    for child in out.children:
        if child.kind is ChunkKind.CHILD:
            assert child.parent_index is not None, "children must point at a parent"
            parent = next(c for c in out.chunks if c.index == child.parent_index)
            assert parent.kind is ChunkKind.PARENT


def test_only_children_are_embedded_parents_are_fetched_by_id() -> None:
    out = StructureAwareChunker().chunk(policy_doc())
    assert all(c.kind is not ChunkKind.PARENT for c in out.children)


def test_breadcrumb_is_embedded_but_not_displayed() -> None:
    """The prefix is a retrieval aid. A citation must quote the clause, not the
    breadcrumb (ARCHITECTURE 6.2 step 3)."""
    out = StructureAwareChunker().chunk(policy_doc(), breadcrumb="Acme General | FHO v3.2")
    child = next(c for c in out.children if c.kind is ChunkKind.CHILD)
    assert child.embedded_text.startswith("[Acme General | FHO v3.2")
    assert not child.text.startswith("[")
    assert child.text in child.embedded_text


def test_child_windows_respect_the_token_budget() -> None:
    out = StructureAwareChunker(child_tokens=60, child_overlap=15).chunk(policy_doc())
    windows = [c for c in out.children if c.kind is ChunkKind.CHILD]
    assert len(windows) > 1, "long clause should split into several windows"
    # Allow slack: a single over-long sentence is emitted whole rather than cut.
    assert all(c.token_count <= 140 for c in windows)


def test_windows_overlap_so_a_clause_is_not_severed() -> None:
    out = StructureAwareChunker(child_tokens=60, child_overlap=25).chunk(policy_doc())
    windows = [c.text for c in out.children if c.kind is ChunkKind.CHILD]
    if len(windows) > 1:
        tail = set(windows[0].split()[-8:])
        assert tail & set(windows[1].split()), "consecutive windows must share text"


def test_parent_carries_the_whole_clause_including_the_carve_out() -> None:
    """The failure this design exists to prevent: retrieving 'dental is excluded'
    without 'unless necessitated by an accident'."""
    out = StructureAwareChunker(child_tokens=40, child_overlap=10).chunk(policy_doc())
    parent = next(c for c in out.parents if "Dental" in c.section_path)
    assert "excluded" in parent.text
    assert "necessitated by an accident" in parent.text


def test_section_path_is_recorded_for_citation() -> None:
    out = StructureAwareChunker().chunk(policy_doc())
    assert any("4.11 Dental Treatment" in c.section_path for c in out.children)


# ────────────────────────────────────────────────────────── tables


def test_table_is_emitted_whole_with_a_summary() -> None:
    out = StructureAwareChunker().chunk(policy_doc())
    tables = [c for c in out.chunks if c.kind is ChunkKind.TABLE]
    summaries = [c for c in out.chunks if c.kind is ChunkKind.TABLE_SUMMARY]
    assert len(tables) == 1
    assert "Room rent" in tables[0].text and "ICU charges" in tables[0].text
    assert summaries, "a grid retrieves badly; the summary is what a prose query matches"


def test_table_summary_points_back_at_the_table() -> None:
    out = StructureAwareChunker().chunk(policy_doc())
    table = next(c for c in out.chunks if c.kind is ChunkKind.TABLE)
    summary = next(c for c in out.chunks if c.kind is ChunkKind.TABLE_SUMMARY)
    assert summary.parent_index == table.index


def test_small_table_is_never_split() -> None:
    assert len(split_table(TABLE, max_tokens=2000)) == 1


def test_large_table_repeats_the_header_in_every_part() -> None:
    """A row group without its header is a list of numbers with no meaning."""
    rows = "\n".join(f"| Benefit {i} | {i}% of SI | {i * 5} days |" for i in range(120))
    parts = split_table(
        f"| Benefit | Sub-limit | Waiting Period |\n|---|---|---|\n{rows}", max_tokens=200
    )
    assert len(parts) > 1
    assert all(p.startswith("| Benefit | Sub-limit | Waiting Period |") for p in parts)


def test_table_detection() -> None:
    assert is_markdown_table(TABLE)
    assert not is_markdown_table("Just a sentence about room rent of 1%.")


def test_table_summary_names_the_columns() -> None:
    summary = table_summary_text(TABLE, "Section 3 > Benefits")
    assert summary and "Sub-limit" in summary and "Section 3 > Benefits" in summary
