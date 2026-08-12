"""Chunker protocol plus Chunk and ParentChunk dataclasses.

Chunks are produced with *indices*, not database ids. The chunker is a pure
function over a parsed document - it never touches Postgres or Qdrant, so it can
be run a thousand times in an ablation sweep without a database.
``pipeline.py`` assigns UUIDs and resolves ``parent_index`` into ``parent_id``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol, runtime_checkable

from app.core.enums import ChunkKind
from app.ingestion.parsers.base import Block, ParsedDocument

# Fallback ratio when no tokenizer is available. Deliberately conservative: an
# over-estimate produces slightly small chunks, an under-estimate produces
# chunks the embedder silently truncates.
CHARS_PER_TOKEN = 3.6


@lru_cache(maxsize=1)
def _tokenizer():
    """bge-m3's own tokenizer, so chunk sizes mean what they say.

    Counting with a different tokenizer (tiktoken) drifts badly on Indic scripts,
    where XLM-RoBERTa produces far more tokens per character than a GPT BPE does.
    Falls back to a character ratio when the model files are not present, so
    chunking still works on a machine that has never downloaded a model.
    """
    try:
        from app.config import ensure_model_cache, settings

        ensure_model_cache()
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(settings.embedding.model)
    except Exception:
        return None


def count_tokens(text: str) -> int:
    tok = _tokenizer()
    if tok is None:
        return int(len(text) / CHARS_PER_TOKEN) + 1
    return len(tok.encode(text, add_special_tokens=False))


@dataclass(slots=True)
class Section:
    """A contiguous run of blocks under one heading path.

    ``path`` is the breadcrumb - ["Section 4", "Exclusions", "4.11 Dental"] -
    which becomes both the embedding prefix and the citation label.
    """

    path: list[str] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)

    @property
    def heading(self) -> str:
        return self.path[-1] if self.path else ""

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.path)

    @property
    def text(self) -> str:
        return "\n\n".join(b.text.strip() for b in self.blocks if b.text.strip())

    @property
    def page_no(self) -> int | None:
        for block in self.blocks:
            if block.page_no is not None:
                return block.page_no
        return None

    @property
    def has_table(self) -> bool:
        return any(b.is_table for b in self.blocks)


@dataclass(slots=True)
class ChunkDraft:
    """A chunk before it has a database identity."""

    index: int
    kind: ChunkKind
    # What a citation displays and what the LLM reads.
    text: str
    # What actually gets embedded: breadcrumb prefix + text. Kept separate so the
    # prefix improves retrieval without ever appearing in a quoted clause.
    embedded_text: str
    section_path: str
    token_count: int
    page_no: int | None = None
    parent_index: int | None = None
    is_table: bool = False

    @property
    def text_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class ChunkedDocument:
    chunks: list[ChunkDraft] = field(default_factory=list)
    strategy: str = "structure_aware"
    strategy_version: str = "v1"

    @property
    def parents(self) -> list[ChunkDraft]:
        return [c for c in self.chunks if c.kind is ChunkKind.PARENT]

    @property
    def children(self) -> list[ChunkDraft]:
        """What gets embedded. Parents are fetched by id at answer time."""
        return [c for c in self.chunks if c.kind is not ChunkKind.PARENT]

    @property
    def total_tokens(self) -> int:
        return sum(c.token_count for c in self.children)


@runtime_checkable
class Chunker(Protocol):
    """Implementations: ``StructureAwareChunker``."""

    @property
    def strategy(self) -> str: ...

    def chunk(self, document: ParsedDocument, *, breadcrumb: str = "") -> ChunkedDocument: ...
