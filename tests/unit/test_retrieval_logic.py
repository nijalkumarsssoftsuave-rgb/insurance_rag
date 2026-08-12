"""Fusion, packing and filter construction - the pure logic, no models."""

from __future__ import annotations

from datetime import date

import pytest
from qdrant_client import models

from app.core.enums import ChunkKind, DocType
from app.retrieval import filters, packing
from app.retrieval.fusion import (
    deduplicate,
    merge_preserving_priority,
    reciprocal_rank_fusion,
)
from app.retrieval.packing import ContextBlock
from app.retrieval.rerankers.base import RerankedHit
from app.retrieval.rerankers.bge_reranker import NoOpReranker
from app.retrieval.vectorstore import Payload, SearchHit


def hit(cid: str, score: float = 0.5, **payload) -> SearchHit:
    base = {Payload.TEXT: f"text for {cid}"}
    base.update(payload)
    return SearchHit(chunk_id=cid, score=score, payload=base)


# ───────────────────────────────────────────────────────────── fusion


def test_rrf_rewards_agreement_across_branches() -> None:
    """A chunk both branches rank highly must beat one only a single branch likes."""
    dense = [hit("a"), hit("b"), hit("c")]
    sparse = [hit("c"), hit("a"), hit("z")]
    fused = reciprocal_rank_fusion([dense, sparse])
    assert fused[0].chunk_id == "a"  # ranks 1 and 2
    assert {h.chunk_id for h in fused} == {"a", "b", "c", "z"}


def test_rrf_deduplicates() -> None:
    fused = reciprocal_rank_fusion([[hit("a")], [hit("a")], [hit("a")]])
    assert len(fused) == 1


def test_rrf_weights_shift_the_ranking() -> None:
    dense = [hit("a"), hit("b")]
    sparse = [hit("b"), hit("a")]
    assert reciprocal_rank_fusion([dense, sparse], weights=[3.0, 1.0])[0].chunk_id == "a"
    assert reciprocal_rank_fusion([dense, sparse], weights=[1.0, 3.0])[0].chunk_id == "b"


def test_rrf_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="weights must match"):
        reciprocal_rank_fusion([[hit("a")]], weights=[1.0, 2.0])


def test_deduplicate_keeps_the_best_score() -> None:
    result = deduplicate([hit("a", 0.2), hit("a", 0.9), hit("b", 0.5)])
    assert len(result) == 2
    assert next(h for h in result if h.chunk_id == "a").score == 0.9


def test_companions_append_without_displacing_matches() -> None:
    """Force-included exclusions must reach context but never outrank real matches."""
    primary = [hit("match1"), hit("match2")]
    merged = merge_preserving_priority(primary, [hit("exclusion"), hit("match1")])
    assert [h.chunk_id for h in merged] == ["match1", "match2", "exclusion"]


# ──────────────────────────────────────────────────────────── packing


def block(cid: str, score: float, text: str = "x" * 360, companion: bool = False) -> ContextBlock:
    return ContextBlock(chunk_id=cid, text=text, score=score, is_companion=companion)


def test_packing_puts_the_best_evidence_last() -> None:
    """Attention degrades in the middle of a context; the tail is the strongest
    position, so the top-scoring block belongs immediately before the question."""
    packed = packing.pack([block("low", 0.2), block("high", 0.9), block("mid", 0.5)])
    assert [b.chunk_id for b in packed.blocks] == ["low", "mid", "high"]


def test_packing_respects_the_token_budget() -> None:
    blocks = [block(f"b{i}", 0.9 - i * 0.01, text="y" * 3600) for i in range(10)]
    packed = packing.pack(blocks, token_budget=3000, max_blocks=10)
    assert packed.total_tokens <= 3000
    assert packed.dropped > 0


def test_packing_respects_the_block_ceiling() -> None:
    blocks = [block(f"b{i}", 0.9) for i in range(20)]
    assert len(packing.pack(blocks, max_blocks=5).blocks) == 5


def test_companions_are_selected_after_scored_matches() -> None:
    blocks = [block("companion", 0.0, companion=True)] + [block(f"m{i}", 0.9) for i in range(3)]
    kept = {b.chunk_id for b in packing.pack(blocks, max_blocks=3).blocks}
    assert "companion" not in kept


def test_empty_input_packs_to_empty() -> None:
    packed = packing.pack([])
    assert packed.blocks == [] and packed.total_tokens == 0


# ──────────────────────────────────────────────────────────── filters


def _keys(f: models.Filter) -> set[str]:
    return {c.key for c in f.must if isinstance(c, models.FieldCondition)}


def test_tenant_and_superseded_are_always_constrained() -> None:
    built = filters.build(filters.RetrievalFilter(tenant_id="acme"))
    assert Payload.TENANT_ID in _keys(built)
    assert Payload.IS_SUPERSEDED in _keys(built)


def test_date_of_loss_adds_both_window_bounds() -> None:
    built = filters.build(filters.RetrievalFilter(date_of_loss=date(2022, 6, 15)))
    assert Payload.EFFECTIVE_FROM_TS in _keys(built)
    assert Payload.EFFECTIVE_TO_TS in _keys(built)


def test_no_date_means_no_window_filter() -> None:
    built = filters.build(filters.RetrievalFilter())
    assert Payload.EFFECTIVE_FROM_TS not in _keys(built)


def test_broadening_drops_soft_filters_but_keeps_tenant() -> None:
    """Tenant isolation is never negotiable, even on the retry path."""
    narrow = filters.RetrievalFilter(
        tenant_id="acme",
        product_name="Family Health Optima",
        insurer="Acme",
        doc_types=[DocType.POLICY_WORDING],
        date_of_loss=date(2024, 1, 1),
    )
    wide = narrow.broadened()
    assert wide.tenant_id == "acme"
    assert wide.product_name is None
    assert wide.insurer is None
    assert wide.date_of_loss is None


def test_parents_are_excluded_from_vector_search_by_default() -> None:
    """Parents are fetched by id during expansion, not matched by embedding."""
    assert ChunkKind.PARENT not in filters.RetrievalFilter().kinds


# ─────────────────────────────────────────────────────────── reranker


def test_noop_reranker_preserves_retrieval_order() -> None:
    """The ablation baseline must not reorder, or it measures the wrong thing."""
    hits = [hit("a", 0.9), hit("b", 0.5), hit("c", 0.1)]
    ranked = NoOpReranker().rerank("q", hits, top_n=3)
    assert [r.chunk_id for r in ranked] == ["a", "b", "c"]
    assert all(isinstance(r, RerankedHit) for r in ranked)
