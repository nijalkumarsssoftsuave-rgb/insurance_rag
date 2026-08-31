"""Parent expansion, overlap dedup, token-budgeted context packing.

Three jobs, in order:

1. **Parent expansion** - swap each matched child for its parent section, so the
   model reads a whole clause rather than a 400-token slice of one.
2. **Dedup** - sibling children of one parent collapse to a single block.
3. **Packing** - fit a token budget, ordering the most relevant *last*.

That last ordering is deliberate and counter-intuitive: attention degrades in the
middle of a long context, and the tail is the strongest position. Putting the best
evidence immediately before the question is worth measurable accuracy
(ARCHITECTURE 8, step 6j).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from app.config import settings
from app.logging import get_logger
from app.retrieval.rerankers.base import RerankedHit
from app.retrieval.vectorstore import Payload, SearchHit, VectorStore

log = get_logger(__name__)

# Approximation used only for budgeting. tiktoken is exact but costs a tokenizer
# pass over every candidate; being 10% conservative is cheaper and safe.
CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass(slots=True)
class ContextBlock:
    """One citable unit of context."""

    chunk_id: str
    text: str
    score: float
    section_path: str | None = None
    doc_id: str | None = None
    page_no: int | None = None
    product_name: str | None = None
    is_companion: bool = False  # force-included, not score-matched

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    def citation(self) -> dict[str, object]:
        """What the UI renders and what the audit trail stores.

        Carries a snippet of the clause: a citation the reader cannot see the
        text of is a reference, not evidence, and the whole point of this system
        is that a person can check the answer against the wording.
        """
        return {
            "chunk_id": self.chunk_id,
            "section_path": self.section_path,
            "page_no": self.page_no,
            "product_name": self.product_name,
            "score": round(self.score, 4),
            "snippet": self.text[:600],
            "is_companion": self.is_companion,
        }


@dataclass(slots=True)
class PackedContext:
    blocks: list[ContextBlock] = field(default_factory=list)
    dropped: int = 0
    total_tokens: int = 0

    @property
    def chunk_ids(self) -> list[str]:
        return [b.chunk_id for b in self.blocks]

    def as_pairs(self) -> list[tuple[str, str | None, str]]:
        """(chunk_id, section_path, text) in final order, for ``wrap_untrusted``.

        The section path is included because the generation prompt requires the
        answer to name the clause ("Clause 4.11, Dental Treatment") while the
        context used to carry only the id and the text. Asked for a clause number
        it had never been shown, the model produced plausible ones: a motor
        exclusions answer cited clauses 8.1-8.4 for text that lives in 7.1-7.4.
        Those inventions are also what the groundedness check intermittently
        rejected, which is why the same question passed or abstained at random.
        """
        return [(b.chunk_id, b.section_path, b.text) for b in self.blocks]


def expand_to_parents(store: VectorStore, ranked: Sequence[RerankedHit]) -> list[ContextBlock]:
    """Replace matched children with their parent sections.

    Children are precise to match against; parents are complete to reason over.
    Several children of one parent collapse into a single block, carrying the best
    score among them.
    """
    parent_ids = {pid for r in ranked if (pid := r.hit.payload.get(Payload.PARENT_ID))}
    parents: dict[str, SearchHit] = {}
    if parent_ids:
        parents = {p.chunk_id: p for p in store.fetch([str(p) for p in parent_ids])}

    blocks: dict[str, ContextBlock] = {}
    for item in ranked:
        payload = item.hit.payload
        parent_id = payload.get(Payload.PARENT_ID)
        source = parents.get(str(parent_id)) if parent_id else None

        # Fall back to the child when the parent is missing - a chunk with no
        # parent is legitimate (a standalone table, a short section).
        key = source.chunk_id if source else item.chunk_id
        text = source.text if source else item.text
        source_payload = source.payload if source else payload

        existing = blocks.get(key)
        if existing is not None:
            existing.score = max(existing.score, item.score)
            continue

        blocks[key] = ContextBlock(
            chunk_id=key,
            text=text,
            score=item.score,
            section_path=source_payload.get(Payload.SECTION_PATH),
            doc_id=source_payload.get(Payload.DOC_ID),
            page_no=source_payload.get(Payload.PAGE_NO),
            product_name=source_payload.get(Payload.PRODUCT_NAME),
        )

    return sorted(blocks.values(), key=lambda b: b.score, reverse=True)


def add_companions(
    blocks: list[ContextBlock], companions: Sequence[SearchHit], *, max_companions: int = 3
) -> list[ContextBlock]:
    """Append force-included exclusions and definitions.

    They are marked and appended rather than merged by score, so they cannot
    displace chunks that actually matched the question - they are insurance
    against a positively-phrased question missing its own exclusion.
    """
    known = {b.chunk_id for b in blocks}
    added = 0
    for hit in companions:
        if added >= max_companions:
            break
        if hit.chunk_id in known or not hit.text.strip():
            continue
        blocks.append(
            ContextBlock(
                chunk_id=hit.chunk_id,
                text=hit.text,
                score=0.0,
                section_path=hit.payload.get(Payload.SECTION_PATH),
                doc_id=hit.payload.get(Payload.DOC_ID),
                page_no=hit.payload.get(Payload.PAGE_NO),
                product_name=hit.payload.get(Payload.PRODUCT_NAME),
                is_companion=True,
            )
        )
        known.add(hit.chunk_id)
        added += 1
    return blocks


def pack(
    blocks: Sequence[ContextBlock],
    *,
    token_budget: int | None = None,
    max_blocks: int | None = None,
) -> PackedContext:
    """Select within budget, then order weakest-first so the best lands last.

    Selection is by score; final ordering is ascending. Companions are always
    considered for inclusion but never crowd out a higher-scoring match.
    """
    budget = token_budget or settings.retrieval.context_token_budget
    ceiling = max_blocks or settings.reranker.top_n

    ordered = sorted(blocks, key=lambda b: (b.is_companion, -b.score))

    selected: list[ContextBlock] = []
    used = 0
    dropped = 0

    for block in ordered:
        if len(selected) >= ceiling:
            dropped += 1
            continue
        cost = block.tokens
        if used + cost > budget:
            dropped += 1
            continue
        selected.append(block)
        used += cost

    # Most relevant last - see the module docstring.
    selected.sort(key=lambda b: b.score)

    if dropped:
        log.debug("Context packing dropped blocks", dropped=dropped, kept=len(selected))

    return PackedContext(blocks=selected, dropped=dropped, total_tokens=used)
