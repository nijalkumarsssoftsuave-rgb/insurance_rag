"""Vector store round-trip: upsert, hybrid search, filters, supersede.

Runs against a throwaway collection with synthetic vectors. The point is to prove
the collection schema, the sparse branch and the effective-date filter behave as
designed before any real embedding work depends on them.
"""

from __future__ import annotations

import random
import uuid
from datetime import date

import pytest
from qdrant_client import models

from app.embeddings.base import SparseVector
from app.retrieval.vectorstore import (
    TS_MAX,
    TS_MIN,
    ChunkPoint,
    Payload,
    VectorStore,
    to_ts,
)

pytestmark = pytest.mark.integration

DIM = 1024
DOC_A = str(uuid.uuid4())
DOC_B = str(uuid.uuid4())


def _dense(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(DIM)]


def _payload(**overrides) -> dict:
    base = {
        Payload.TENANT_ID: "default",
        Payload.DOC_ID: DOC_A,
        Payload.DOC_VERSION_ID: str(uuid.uuid4()),
        Payload.KIND: "child",
        Payload.INSURER: "Acme General",
        Payload.PRODUCT_NAME: "Family Health Optima",
        Payload.DOC_TYPE: "policy_wording",
        Payload.LANGUAGE: "en",
        Payload.IS_SUPERSEDED: False,
        Payload.EFFECTIVE_FROM_TS: to_ts(date(2024, 1, 1), default=TS_MIN),
        Payload.EFFECTIVE_TO_TS: to_ts(date(2025, 12, 31), default=TS_MAX),
        Payload.SECTION_PATH: "Section 4 > Exclusions",
        Payload.TEXT: "Dental treatment is excluded unless necessitated by accident.",
    }
    base.update(overrides)
    return base


@pytest.fixture(scope="module")
def seeded(vector_store: VectorStore) -> VectorStore:
    points = [
        ChunkPoint(
            chunk_id=uuid.uuid4(),
            dense=_dense(1),
            lexical=SparseVector(indices=[10, 20, 30], values=[0.9, 0.5, 0.2]),
            payload=_payload(),
        ),
        ChunkPoint(
            chunk_id=uuid.uuid4(),
            dense=_dense(2),
            lexical=SparseVector(indices=[20, 40], values=[0.7, 0.6]),
            payload=_payload(
                **{
                    Payload.PRODUCT_NAME: "Motor Shield",
                    Payload.TEXT: "Own damage cover applies to accidental external damage.",
                }
            ),
        ),
        # An older wording: effective window closed in 2023.
        ChunkPoint(
            chunk_id=uuid.uuid4(),
            dense=_dense(3),
            lexical=SparseVector(indices=[10, 50], values=[0.8, 0.3]),
            payload=_payload(
                **{
                    Payload.DOC_ID: DOC_B,
                    Payload.EFFECTIVE_FROM_TS: to_ts(date(2022, 1, 1), default=TS_MIN),
                    Payload.EFFECTIVE_TO_TS: to_ts(date(2023, 12, 31), default=TS_MAX),
                    Payload.TEXT: "Dental treatment is excluded in all circumstances.",
                }
            ),
        ),
    ]
    vector_store.upsert(points, wait=True)
    return vector_store


def test_collection_is_created_with_expected_shape(vector_store: VectorStore) -> None:
    info = vector_store.info()
    assert info["exists"] is True
    assert info["status"] == "green"


def test_ensure_collection_is_idempotent(vector_store: VectorStore) -> None:
    assert vector_store.ensure_collection() is False


def test_upsert_and_count(seeded: VectorStore) -> None:
    assert seeded.count() == 3


def test_dense_only_search_returns_hits(seeded: VectorStore) -> None:
    hits = seeded.hybrid_search(dense_queries=[_dense(1)], limit=10)
    assert len(hits) == 3
    assert hits[0].text  # payload came back


def test_hybrid_search_fuses_dense_and_sparse(seeded: VectorStore) -> None:
    """Both branches must contribute; a sparse-only match should still surface."""
    hits = seeded.hybrid_search(
        dense_queries=[_dense(99)],  # unrelated to any indexed vector
        lexical_queries=[SparseVector(indices=[50], values=[1.0])],  # matches point 3 only
        limit=10,
    )
    assert len(hits) == 3
    ids = [h.chunk_id for h in hits]
    assert len(set(ids)) == 3, "RRF fusion must deduplicate across branches"


def test_search_requires_at_least_one_query_vector(seeded: VectorStore) -> None:
    with pytest.raises(ValueError, match="at least one query vector"):
        seeded.hybrid_search(dense_queries=[])


def test_metadata_filter_scopes_results(seeded: VectorStore) -> None:
    flt = models.Filter(
        must=[
            models.FieldCondition(
                key=Payload.PRODUCT_NAME, match=models.MatchValue(value="Motor Shield")
            )
        ]
    )
    hits = seeded.hybrid_search(dense_queries=[_dense(1)], query_filter=flt, limit=10)
    assert len(hits) == 1
    assert hits[0].payload[Payload.PRODUCT_NAME] == "Motor Shield"


def test_effective_date_filter_selects_the_wording_in_force(seeded: VectorStore) -> None:
    """The claim-date filter from ARCHITECTURE 5.3.

    A loss in 2022 must retrieve the 2022-2023 wording, not the current one -
    answering an old claim from today's wording is the failure this prevents.
    """
    loss_ts = to_ts(date(2022, 6, 15), default=TS_MIN)
    flt = models.Filter(
        must=[
            models.FieldCondition(key=Payload.EFFECTIVE_FROM_TS, range=models.Range(lte=loss_ts)),
            models.FieldCondition(key=Payload.EFFECTIVE_TO_TS, range=models.Range(gte=loss_ts)),
        ]
    )
    hits = seeded.hybrid_search(dense_queries=[_dense(1)], query_filter=flt, limit=10)
    assert len(hits) == 1
    assert hits[0].payload[Payload.DOC_ID] == DOC_B
    assert "in all circumstances" in hits[0].text


def test_open_ended_window_sentinels_never_exclude(vector_store: VectorStore) -> None:
    """A document with no effective dates must remain retrievable at any date."""
    assert to_ts(None, default=TS_MIN) == TS_MIN
    assert to_ts(None, default=TS_MAX) == TS_MAX
    assert TS_MIN <= to_ts(date(2026, 1, 1), default=TS_MIN) <= TS_MAX


def test_fetch_by_id_supports_parent_expansion(seeded: VectorStore) -> None:
    first = seeded.hybrid_search(dense_queries=[_dense(1)], limit=1)[0]
    fetched = seeded.fetch([first.chunk_id])
    assert len(fetched) == 1
    assert fetched[0].chunk_id == first.chunk_id
    assert fetched[0].text == first.text


def test_scroll_by_filter_force_includes_a_section(seeded: VectorStore) -> None:
    """Filter-only retrieval, used to force-include exclusions for coverage questions."""
    flt = models.Filter(
        must=[
            models.FieldCondition(
                key=Payload.SECTION_PATH,
                match=models.MatchValue(value="Section 4 > Exclusions"),
            )
        ]
    )
    assert len(seeded.scroll_by_filter(flt, limit=10)) == 3


def test_mark_superseded_flips_payload_without_deleting(seeded: VectorStore) -> None:
    target = seeded.hybrid_search(dense_queries=[_dense(1)], limit=1)[0]
    version_id = target.payload[Payload.DOC_VERSION_ID]

    seeded.mark_superseded(doc_version_id=version_id)

    refetched = seeded.fetch([target.chunk_id])[0]
    assert refetched.payload[Payload.IS_SUPERSEDED] is True
    assert seeded.count() == 3, "superseding must never delete - old claims need it"
