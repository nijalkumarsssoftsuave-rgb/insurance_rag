"""Clause-hierarchy splitter producing parent and child chunks.

The method, in five steps:

1. **Group blocks into sections** using the document's own heading hierarchy.
   Policy wordings already declare their boundaries ("4.11 Dental"), so we read
   them rather than inferring them.
2. **Parent = the whole section**, capped at ``parent_max_tokens``. This is what
   the LLM reads, so a clause arrives with its conditions attached.
3. **Children = ~400-token windows** with overlap, split *within* a parent. These
   are what get embedded, because small windows match precisely.
4. **Breadcrumb prefix** on every child before embedding, never in the displayed
   text.
5. **Tables never split** - a benefit grid cut in half yields the wrong sub-limit.

Why this and not the simpler options: fixed-size splits mid-clause every time;
recursive-character is blind to headings and destroys tables; semantic splitting
infers boundaries the document already states, at the cost of an embedding pass
per sentence. See ARCHITECTURE 6.

The result is the failure this prevents: a query matches the child window
containing "dental ... excluded", and the LLM receives the *whole* clause
including "unless necessitated by an accident".
"""

from __future__ import annotations

import re

from app.config import settings
from app.core.enums import ChunkKind
from app.ingestion.chunking.base import (
    ChunkDraft,
    ChunkedDocument,
    Section,
    count_tokens,
)
from app.ingestion.chunking.table_handler import split_table, table_summary_text
from app.ingestion.parsers.base import Block, ParsedDocument
from app.logging import get_logger

log = get_logger(__name__)

# Sections whose content must be retrievable even when a positively-phrased
# question ("is dental covered?") would never match them by similarity.
COMPANION_SECTIONS = re.compile(
    r"exclusion|not\s+covered|definition|waiting\s+period|general\s+condition", re.I
)

# Below this a chunk carries too little context for the cross-encoder to judge,
# so it is merged into its neighbour instead of indexed alone.
MIN_CHUNK_TOKENS = 60

SENTENCE_END = re.compile(r"(?<=[.;:!?])\s+")


class StructureAwareChunker:
    def __init__(
        self,
        *,
        child_tokens: int | None = None,
        child_overlap: int | None = None,
        parent_max_tokens: int | None = None,
    ) -> None:
        cfg = settings.chunking
        self.child_tokens = child_tokens or cfg.child_tokens
        self.child_overlap = child_overlap or cfg.child_overlap
        self.parent_max_tokens = parent_max_tokens or cfg.parent_max_tokens

    @property
    def strategy(self) -> str:
        return "structure_aware"

    def chunk(self, document: ParsedDocument, *, breadcrumb: str = "") -> ChunkedDocument:
        sections = build_sections(document)
        result = ChunkedDocument(
            strategy=self.strategy, strategy_version=settings.chunking.strategy_version
        )
        index = 0

        for section in sections:
            if not section.text.strip():
                continue
            index = self._emit_section(section, result, index, breadcrumb)

        log.debug(
            "Chunked document",
            sections=len(sections),
            parents=len(result.parents),
            children=len(result.children),
            tokens=result.total_tokens,
        )
        return result

    # ─────────────────────────────────────────────────────────── internals

    def _emit_section(
        self, section: Section, result: ChunkedDocument, index: int, breadcrumb: str
    ) -> int:
        prefix = _prefix(breadcrumb, section.breadcrumb)

        # Tables are emitted whole, plus a short summary chunk that a question
        # like "what are the sub-limits?" can match without the grid itself
        # having to score well.
        table_blocks = [b for b in section.blocks if b.is_table]
        prose_blocks = [b for b in section.blocks if not b.is_table]

        prose_text = "\n\n".join(b.text.strip() for b in prose_blocks if b.text.strip())
        parent_index: int | None = None

        if prose_text:
            parent_text = self._cap(prose_text)
            parent_index = index
            result.chunks.append(
                ChunkDraft(
                    index=index,
                    kind=ChunkKind.PARENT,
                    text=parent_text,
                    embedded_text=parent_text,  # parents are never embedded
                    section_path=section.breadcrumb,
                    token_count=count_tokens(parent_text),
                    page_no=section.page_no,
                )
            )
            index += 1

            for window in self._windows(parent_text):
                result.chunks.append(
                    ChunkDraft(
                        index=index,
                        kind=ChunkKind.CHILD,
                        text=window,
                        embedded_text=f"{prefix}\n{window}",
                        section_path=section.breadcrumb,
                        token_count=count_tokens(window),
                        page_no=section.page_no,
                        parent_index=parent_index,
                    )
                )
                index += 1

        for block in table_blocks:
            index = self._emit_table(block, section, result, index, prefix, parent_index)

        return index

    def _emit_table(
        self,
        block: Block,
        section: Section,
        result: ChunkedDocument,
        index: int,
        prefix: str,
        parent_index: int | None,
    ) -> int:
        for part in split_table(block.text, max_tokens=self.parent_max_tokens):
            result.chunks.append(
                ChunkDraft(
                    index=index,
                    kind=ChunkKind.TABLE,
                    text=part,
                    embedded_text=f"{prefix}\n{part}",
                    section_path=section.breadcrumb,
                    token_count=count_tokens(part),
                    page_no=block.page_no,
                    parent_index=parent_index,
                    is_table=True,
                )
            )
            table_index = index
            index += 1

            summary = table_summary_text(part, section.breadcrumb)
            if summary:
                result.chunks.append(
                    ChunkDraft(
                        index=index,
                        kind=ChunkKind.TABLE_SUMMARY,
                        text=summary,
                        embedded_text=f"{prefix}\n{summary}",
                        section_path=section.breadcrumb,
                        token_count=count_tokens(summary),
                        page_no=block.page_no,
                        # Points at the table itself, so matching the summary
                        # returns the grid.
                        parent_index=table_index,
                    )
                )
                index += 1
        return index

    def _cap(self, text: str) -> str:
        """Trim a parent to the token cap at a paragraph boundary."""
        if count_tokens(text) <= self.parent_max_tokens:
            return text
        paragraphs = text.split("\n\n")
        kept: list[str] = []
        total = 0
        for para in paragraphs:
            cost = count_tokens(para)
            if total + cost > self.parent_max_tokens and kept:
                break
            kept.append(para)
            total += cost
        return "\n\n".join(kept) if kept else text

    def _windows(self, text: str) -> list[str]:
        """Sliding windows over sentences, sized in tokens.

        Splitting on sentence boundaries rather than raw token offsets keeps a
        window from starting mid-clause, which is what makes a retrieved chunk
        readable on its own.
        """
        if count_tokens(text) <= self.child_tokens:
            return [text]

        sentences = [s for s in SENTENCE_END.split(text) if s.strip()]
        if not sentences:
            return [text]

        costs = [count_tokens(s) for s in sentences]
        windows: list[str] = []
        start = 0

        while start < len(sentences):
            total = 0
            end = start
            while end < len(sentences) and total + costs[end] <= self.child_tokens:
                total += costs[end]
                end += 1

            # A single sentence longer than the window: emit it whole rather
            # than cutting a clause in half.
            if end == start:
                end = start + 1
                total = costs[start]

            windows.append(" ".join(sentences[start:end]).strip())

            if end >= len(sentences):
                break

            # Step back far enough to cover the overlap budget.
            back = 0
            overlap = 0
            while end - back - 1 > start and overlap < self.child_overlap:
                back += 1
                overlap += costs[end - back]
            start = end - back

        return _merge_stubs(windows)


def _merge_stubs(windows: list[str]) -> list[str]:
    """Fold a too-small trailing window into its predecessor."""
    if len(windows) > 1 and count_tokens(windows[-1]) < MIN_CHUNK_TOKENS:
        windows[-2] = f"{windows[-2]} {windows[-1]}".strip()
        windows.pop()
    return windows


def build_sections(document: ParsedDocument) -> list[Section]:
    """Group blocks under their heading path.

    Maintains a stack of open headings so a level-3 heading inherits the level-1
    and level-2 above it, producing "Section 4 > Exclusions > 4.11 Dental".
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    current = Section(path=[], blocks=[])

    for block in document.blocks:
        if block.is_heading:
            if current.blocks:
                sections.append(current)
            level = block.level or 1
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, block.text.strip()))
            current = Section(path=[h for _, h in stack], blocks=[])
        else:
            current.blocks.append(block)

    if current.blocks:
        sections.append(current)

    # A document with no detected headings still has to be chunked. Treat the
    # whole thing as one section rather than returning nothing.
    if not sections and document.blocks:
        sections = [Section(path=[document.title or "Document"], blocks=document.blocks)]

    return sections


def is_companion_section(section_path: str) -> bool:
    """Exclusions, definitions and waiting periods get force-included at query
    time regardless of similarity (ARCHITECTURE 6.2 step 6)."""
    return bool(COMPANION_SECTIONS.search(section_path))


def _prefix(document_breadcrumb: str, section_breadcrumb: str) -> str:
    """The embedding prefix.

    Free, deterministic, and a large recall win: without it, 400 chunks across a
    corpus that all discuss waiting periods are indistinguishable to a query
    asking about one.
    """
    parts = [p for p in (document_breadcrumb, section_breadcrumb) if p]
    return f"[{' | '.join(parts)}]" if parts else ""
