"""Primary parser: layout-aware, markdown tables, reading order.

Docling runs a layout model, so it gets the two things heuristics cannot:
**reading order** in multi-column policy wordings (where naive extraction
interleaves the columns into gibberish) and **table structure** as markdown.

Configured for clean digital PDFs: OCR off, table structure on. OCR would add
minutes per document on CPU and this corpus does not need it - but the flag is
here for the day a scanned endorsement arrives.

MIT licensed. Slower than pypdfium2 by a large factor, which is why
``pipeline.py`` routes simple documents to the fast path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from app.ingestion.parsers.base import (
    Block,
    BlockType,
    ParsedDocument,
    ParserError,
    infer_heading_level,
)
from app.logging import get_logger

log = get_logger(__name__)

SUPPORTED = {".pdf", ".docx", ".doc", ".pptx", ".html", ".md"}

# Docling's label vocabulary -> our block types. Anything unmapped falls through
# to PARAGRAPH, which is the safe default: worst case a heading is treated as
# body text and the section is slightly coarser.
_LABEL_MAP = {
    "title": (BlockType.HEADING, 1),
    "section_header": (BlockType.HEADING, 2),
    "subtitle-level-1": (BlockType.HEADING, 2),
    "list_item": (BlockType.LIST_ITEM, None),
    "table": (BlockType.TABLE, None),
    "caption": (BlockType.CAPTION, None),
    "text": (BlockType.PARAGRAPH, None),
    "paragraph": (BlockType.PARAGRAPH, None),
}


class DoclingParser:
    """Layout-aware parsing. Lazily constructs the converter."""

    def __init__(self, *, do_ocr: bool = False, do_tables: bool = True) -> None:
        self._do_ocr = do_ocr
        self._do_tables = do_tables
        self._converter: Any = None

    @property
    def name(self) -> str:
        return "docling"

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED

    def _build_converter(self) -> Any:
        if self._converter is not None:
            return self._converter

        # Docling's layout model runs under torch.compile, which shells out to a
        # C++ compiler. On a machine without MSVC (or gcc) that is a hard failure
        # with an opaque "Compiler: cl is not found". Eager mode is slightly
        # slower and always available, and requiring users to install Visual
        # Studio build tools to read a PDF is not a reasonable ask.
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass

        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ParserError("docling is not installed") from exc

        options = PdfPipelineOptions()
        options.do_ocr = self._do_ocr
        options.do_table_structure = self._do_tables
        if self._do_tables:
            options.table_structure_options.do_cell_matching = True

        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        return self._converter

    def parse(self, path: Path) -> ParsedDocument:
        converter = self._build_converter()
        try:
            result = converter.convert(str(path))
        except Exception as exc:
            raise ParserError(f"Docling failed on {path.name}: {exc}") from exc

        doc = result.document
        blocks = _to_blocks(doc)
        warnings: list[str] = []
        if not blocks:
            warnings.append(
                "Docling extracted no content - the file may be scanned (enable OCR) or corrupt."
            )

        return ParsedDocument(
            blocks=blocks,
            page_count=_page_count(doc),
            title=getattr(doc, "name", None) or path.stem,
            parser=self.name,
            warnings=warnings,
        )


def _to_blocks(doc: Any) -> list[Block]:
    """Flatten Docling's document into ordered blocks.

    Uses ``iterate_items`` where available so reading order is preserved, and
    falls back to markdown export otherwise - Docling's API has moved across
    2.x releases and a parser that hard-fails on a minor version bump is worse
    than one that degrades.
    """
    blocks: list[Block] = []

    try:
        for item, _level in doc.iterate_items():
            block = _item_to_block(item)
            if block is not None:
                blocks.append(block)
        if blocks:
            return blocks
    except Exception as exc:
        log.warning("Docling iterate_items unavailable, using markdown", error=str(exc))

    try:
        return _from_markdown(doc.export_to_markdown())
    except Exception as exc:
        log.warning("Docling markdown export failed", error=str(exc))
        return []


def _item_to_block(item: Any) -> Block | None:
    label = str(getattr(item, "label", "") or "").lower()

    if label == "table":
        text = _table_markdown(item)
        return Block(text=text, type=BlockType.TABLE, page_no=_page_of(item)) if text else None

    text = (getattr(item, "text", "") or "").strip()
    if not text:
        return None

    block_type, level = _LABEL_MAP.get(label, (BlockType.PARAGRAPH, None))

    if block_type is BlockType.HEADING:
        # Docling labels almost every heading "section_header", so its own level
        # is nearly constant. Prefer the depth implied by the clause numbering;
        # without this every heading lands at the same level and the breadcrumb
        # collapses to a flat list.
        level = infer_heading_level(text) or level

    return Block(text=text, type=block_type, page_no=_page_of(item), level=level)


def _table_markdown(item: Any) -> str:
    for method in ("export_to_markdown", "export_to_dataframe"):
        fn = getattr(item, method, None)
        if fn is None:
            continue
        try:
            out = fn()
            return out.to_markdown(index=False) if hasattr(out, "to_markdown") else str(out)
        except Exception:
            continue
    return (getattr(item, "text", "") or "").strip()


def _page_of(item: Any) -> int | None:
    try:
        prov = getattr(item, "prov", None)
        if prov:
            return int(prov[0].page_no)
    except Exception:
        pass
    return None


def _page_count(doc: Any) -> int:
    pages = getattr(doc, "pages", None)
    if pages is None:
        return 0
    try:
        return len(pages)
    except TypeError:
        return 0


def _from_markdown(markdown: str) -> list[Block]:
    """Last-resort structure recovery from markdown headings and tables."""
    blocks: list[Block] = []
    buffer: list[str] = []
    table: list[str] = []

    def flush_text() -> None:
        if buffer:
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(Block(text=text, type=BlockType.PARAGRAPH))
            buffer.clear()

    def flush_table() -> None:
        if table:
            blocks.append(Block(text="\n".join(table).strip(), type=BlockType.TABLE))
            table.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            flush_text()
            table.append(stripped)
            continue
        flush_table()

        if stripped.startswith("#"):
            flush_text()
            level = len(stripped) - len(stripped.lstrip("#"))
            blocks.append(
                Block(
                    text=stripped.lstrip("#").strip(),
                    type=BlockType.HEADING,
                    level=min(level, 6),
                )
            )
        elif not stripped:
            flush_text()
        else:
            buffer.append(stripped)

    flush_text()
    flush_table()
    return blocks
