"""Reciprocal Rank Fusion and candidate deduplication.

Qdrant fuses the prefetch branches server-side in the common path. This module is
for the cases that cross a call boundary: merging force-included companion chunks
with scored results, or merging separate collections during an ablation.

RRF is rank-based on purpose. Dense cosine scores and sparse IDF scores live on
incompatible scales, and normalizing them into comparability requires assumptions
that are wrong often enough to matter. Ranks need no such assumption.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.config import settings
from app.retrieval.vectorstore import SearchHit


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[SearchHit]],
    *,
    k: int | None = None,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[SearchHit]:
    """Fuse ranked lists by ``sum(weight / (k + rank))``.

    Args:
        k: the RRF constant. Larger flattens the contribution of top ranks.
        weights: per-list multipliers, for deliberately trusting one branch more.
        limit: truncate the fused result.
    """
    rrf_k = k if k is not None else settings.retrieval.rrf_k
    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must match the number of result lists")

    scores: dict[str, float] = {}
    best: dict[str, SearchHit] = {}

    for hits, weight in zip(result_lists, weights, strict=True):
        for rank, hit in enumerate(hits):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (rrf_k + rank + 1)
            # Keep the first sighting: payloads are identical, and this makes the
            # output stable across runs.
            best.setdefault(hit.chunk_id, hit)

    fused = [
        SearchHit(chunk_id=cid, score=score, payload=best[cid].payload)
        for cid, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return fused[:limit] if limit else fused


def deduplicate(hits: Sequence[SearchHit]) -> list[SearchHit]:
    """Drop repeated chunk ids, keeping the highest-scoring occurrence."""
    seen: dict[str, SearchHit] = {}
    for hit in hits:
        existing = seen.get(hit.chunk_id)
        if existing is None or hit.score > existing.score:
            seen[hit.chunk_id] = hit
    return sorted(seen.values(), key=lambda h: h.score, reverse=True)


def merge_preserving_priority(
    primary: Sequence[SearchHit], extra: Sequence[SearchHit], *, limit: int | None = None
) -> list[SearchHit]:
    """Append ``extra`` after ``primary``, skipping anything already present.

    Used for force-included exclusions and definitions: they must reach the
    context, but they must not displace the chunks that actually matched the
    question.
    """
    out = list(primary)
    known = {hit.chunk_id for hit in out}
    for hit in extra:
        if hit.chunk_id not in known:
            out.append(hit)
            known.add(hit.chunk_id)
    return out[:limit] if limit else out
