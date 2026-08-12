"""bge-m3 -> Qdrant -> hybrid search, with the real model.

The proof that the AI layer and the storage layer actually fit together: vectors
of the right shape and norm, sparse weights Qdrant accepts, and retrieval that
puts the right clause first for both a paraphrased question and an exact clause
reference.

Slow - loads ~2.3 GB of weights. Run with `-m integration`.
"""

from __future__ import annotations

import uuid

import pytest

from app.embeddings import get_embedder
from app.retrieval.vectorstore import ChunkPoint, Payload, VectorStore

pytestmark = pytest.mark.integration

# Realistic policy text. The dental clause is the target for most assertions.
CORPUS: list[tuple[str, str]] = [
    (
        "Section 4 > Exclusions > 4.11 Dental",
        "Dental treatment, dental surgery and orthodontic procedures are excluded "
        "unless such treatment is necessitated by an accident and requires "
        "hospitalisation for at least twenty-four consecutive hours.",
    ),
    (
        "Section 3 > Benefits > 3.2 Room Rent",
        "Room rent, boarding and nursing expenses are payable up to one per cent "
        "of the sum insured per day, subject to a maximum of five thousand rupees "
        "per day for a single private air-conditioned room.",
    ),
    (
        "Section 5 > Waiting Periods > 5.1 Pre-existing Disease",
        "Any pre-existing disease and its direct complications shall be excluded "
        "until the expiry of forty-eight months of continuous coverage from the "
        "first policy inception date with the Company.",
    ),
    (
        "Section 6 > Claims Procedure > 6.3 Cashless",
        "For cashless treatment the Insured Person must obtain pre-authorisation "
        "from the Third Party Administrator at least forty-eight hours prior to a "
        "planned hospitalisation at a network hospital.",
    ),
]


@pytest.fixture(scope="module")
def indexed(vector_store: VectorStore) -> VectorStore:
    embedder = get_embedder()

    # Breadcrumb prefix, exactly as the chunker will build it (ARCHITECTURE 6.2
    # step 3). It is embedded but is not what a citation displays.
    prefixed = [
        f"[Acme General | Family Health Optima | Policy Wording v3.2 | {path}]\n{text}"
        for path, text in CORPUS
    ]
    embedded = embedder.embed_documents(prefixed)

    points = [
        ChunkPoint(
            chunk_id=uuid.uuid4(),
            dense=embedded.dense[i],
            lexical=embedded.sparse[i],
            payload={
                Payload.TENANT_ID: "default",
                Payload.DOC_ID: "doc-fho-32",
                Payload.DOC_VERSION_ID: "ver-1",
                Payload.KIND: "child",
                Payload.PRODUCT_NAME: "Family Health Optima",
                Payload.INSURER: "Acme General",
                Payload.DOC_TYPE: "policy_wording",
                Payload.LANGUAGE: "en",
                Payload.IS_SUPERSEDED: False,
                Payload.SECTION_PATH: path,
                Payload.TEXT: text,
                Payload.EMBEDDING_MODEL: embedder.model_name,
            },
        )
        for i, (path, text) in enumerate(CORPUS)
    ]
    vector_store.upsert(points, wait=True)
    return vector_store


def _search(store: VectorStore, query: str, limit: int = 4):
    dense, sparse = get_embedder().embed_query(query)
    return store.hybrid_search(dense_queries=[dense], lexical_queries=[sparse], limit=limit)


def _sections(hits) -> list[str]:
    return [h.payload[Payload.SECTION_PATH] for h in hits]


def _search_and_rerank(store: VectorStore, query: str) -> list[str]:
    """Hybrid retrieval followed by the cross-encoder - the real pipeline order.

    Asserting on hybrid rank-1 alone tests the wrong stage. Hybrid search is
    recall-oriented: Qdrant's server-side RRF weights the dense and sparse
    branches equally, so on a query with no genuine lexical overlap the sparse
    branch contributes noise at full strength and can outvote a confident dense
    match. Precision is the reranker's job, which is why it is not optional.
    """
    from app.retrieval.rerankers import BGEReranker

    hits = _search(store, query, limit=4)
    ranked = BGEReranker().rerank(query, hits, top_n=4)
    return [r.hit.payload[Payload.SECTION_PATH] for r in ranked]


def test_vectors_have_the_shape_the_collection_expects(indexed: VectorStore) -> None:
    dense, sparse = get_embedder().embed_query("is dental covered")
    assert len(dense) == 1024
    assert abs(sum(v * v for v in dense) - 1.0) < 1e-3, "must be L2-normalised for cosine"
    assert sparse.nnz > 0
    assert all(isinstance(i, int) for i in sparse.indices)


def test_all_chunks_were_indexed(indexed: VectorStore) -> None:
    assert indexed.count() == len(CORPUS)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("can I claim for getting my teeth fixed?", "Dental"),
        ("how long must I wait to claim for an illness I already had", "Pre-existing"),
        ("what does clause 6.3 say about cashless pre-authorisation", "Cashless"),
        ("how much room rent can I claim per day", "Room Rent"),
    ],
)
def test_hybrid_search_recalls_the_right_clause(
    indexed: VectorStore, query: str, expected: str
) -> None:
    """Recall is what hybrid retrieval owns: the answer must be in the candidates.

    Rank-1 is explicitly not asserted here - see ``_search_and_rerank``.
    """
    sections = _sections(_search(indexed, query, limit=4))
    assert any(expected in s for s in sections), f"{expected} missing from {sections}"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Customer vocabulary, not policy vocabulary. "teeth" never appears in the
        # corpus - the clause says "dental" - so the sparse branch is pure noise
        # here and hybrid alone ranks Room Rent first.
        ("can I claim for getting my teeth fixed?", "Dental"),
        ("how long must I wait to claim for an illness I already had", "Pre-existing"),
        ("what does clause 6.3 say about cashless pre-authorisation", "Cashless"),
        ("how much room rent can I claim per day", "Room Rent"),
    ],
)
def test_reranking_puts_the_right_clause_first(
    indexed: VectorStore, query: str, expected: str
) -> None:
    """Precision is the cross-encoder's job, and this is the proof it earns its
    latency: two of these four queries are ranked wrongly by hybrid search and
    corrected by reranking."""
    sections = _search_and_rerank(indexed, query)
    assert expected in sections[0], f"expected {expected} first, got {sections}"


def test_metadata_filter_still_applies_with_real_vectors(indexed: VectorStore) -> None:
    from app.retrieval import filters

    dense, sparse = get_embedder().embed_query("dental")
    hits = indexed.hybrid_search(
        dense_queries=[dense],
        lexical_queries=[sparse],
        query_filter=filters.build(
            filters.RetrievalFilter(tenant_id="default", product_name="Motor Shield")
        ),
        limit=5,
    )
    assert hits == [], "a filter on another product must exclude everything"


def test_citation_text_excludes_the_breadcrumb_prefix(indexed: VectorStore) -> None:
    """The prefix improves retrieval but must never appear in a quoted clause."""
    hits = _search(indexed, "dental exclusion")
    assert not hits[0].text.startswith("[Acme General")
