"""Qdrant client: collection lifecycle, upsert, hybrid query.

The collection carries one dense vector (bge-m3, 1024d) and one or two sparse
vectors (bge-m3 learned lexical weights, plus optional classic BM25). Multi-branch
prefetch with server-side RRF means every query variant and every branch resolves
in a single round trip (ARCHITECTURE 7.2, 8).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import QdrantSettings, settings
from app.embeddings.base import SparseVector

logger = logging.getLogger(__name__)

# Sentinels for documents with an open-ended or unknown effective window.
# Storing epoch seconds rather than nullable dates means the range filter is a
# plain comparison with no null-handling branch - and a missing date can never
# silently exclude a document from a date-scoped search.
TS_MIN = 0
TS_MAX = 4_102_444_800  # 2100-01-01T00:00:00Z


class Payload:
    """Payload keys, defined once so the writer and the reader cannot drift."""

    TENANT_ID = "tenant_id"
    DOC_ID = "doc_id"
    DOC_VERSION_ID = "doc_version_id"
    CHUNK_ID = "chunk_id"
    PARENT_ID = "parent_id"
    KIND = "kind"

    INSURER = "insurer"
    PRODUCT_NAME = "product_name"
    UIN = "uin"
    DOC_TYPE = "doc_type"
    LANGUAGE = "language"
    JURISDICTION = "jurisdiction"

    EFFECTIVE_FROM_TS = "effective_from_ts"
    EFFECTIVE_TO_TS = "effective_to_ts"
    EFFECTIVE_FROM = "effective_from"  # ISO string, display only
    EFFECTIVE_TO = "effective_to"

    IS_SUPERSEDED = "is_superseded"
    SECTION_PATH = "section_path"
    PAGE_NO = "page_no"
    CHUNK_INDEX = "chunk_index"
    TOKEN_COUNT = "token_count"  # noqa: S105 - LLM tokens, not an auth token
    TEXT = "text"
    EMBEDDING_MODEL = "embedding_model"
    CHUNK_STRATEGY_VERSION = "chunk_strategy_version"


# Indexed fields only. Everything else is stored but not indexed - each index
# costs memory and ingest time, so they are earned by an actual filter in
# `app/retrieval/filters.py`, not added speculatively.
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    Payload.TENANT_ID: models.PayloadSchemaType.KEYWORD,
    Payload.DOC_ID: models.PayloadSchemaType.KEYWORD,
    Payload.DOC_VERSION_ID: models.PayloadSchemaType.KEYWORD,
    Payload.PARENT_ID: models.PayloadSchemaType.KEYWORD,
    Payload.KIND: models.PayloadSchemaType.KEYWORD,
    Payload.INSURER: models.PayloadSchemaType.KEYWORD,
    Payload.PRODUCT_NAME: models.PayloadSchemaType.KEYWORD,
    Payload.DOC_TYPE: models.PayloadSchemaType.KEYWORD,
    Payload.LANGUAGE: models.PayloadSchemaType.KEYWORD,
    Payload.IS_SUPERSEDED: models.PayloadSchemaType.BOOL,
    Payload.EFFECTIVE_FROM_TS: models.PayloadSchemaType.INTEGER,
    Payload.EFFECTIVE_TO_TS: models.PayloadSchemaType.INTEGER,
}


def to_ts(value: date | datetime | None, *, default: int) -> int:
    """Date to epoch seconds, with an explicit sentinel for the open-ended case."""
    if value is None:
        return default
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=UTC)
    else:
        dt = datetime(value.year, value.month, value.day, tzinfo=UTC)
    return int(dt.timestamp())


@dataclass(slots=True)
class ChunkPoint:
    """One point as it is written to Qdrant.

    ``chunk_id`` is the Postgres ``chunks.id``. The two stores share the identifier
    so a retrieved point resolves back to its auditable row without a lookup table.
    """

    chunk_id: uuid.UUID
    dense: list[float]
    payload: dict[str, Any]
    lexical: SparseVector | None = None
    bm25: SparseVector | None = None

    def to_point_struct(self, cfg: QdrantSettings) -> models.PointStruct:
        vector: dict[str, Any] = {cfg.dense_vector_name: self.dense}
        if self.lexical is not None and self.lexical.nnz:
            vector[cfg.lexical_vector_name] = models.SparseVector(
                indices=self.lexical.indices, values=self.lexical.values
            )
        if self.bm25 is not None and self.bm25.nnz:
            vector[cfg.bm25_vector_name] = models.SparseVector(
                indices=self.bm25.indices, values=self.bm25.values
            )
        return models.PointStruct(id=str(self.chunk_id), vector=vector, payload=self.payload)


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return str(self.payload.get(Payload.TEXT, ""))

    @property
    def parent_id(self) -> str | None:
        value = self.payload.get(Payload.PARENT_ID)
        return str(value) if value else None


class VectorStore:
    """Thin, explicit wrapper over ``QdrantClient``.

    Deliberately not a LangChain ``VectorStore``: multi-branch prefetch, named
    sparse vectors and server-side fusion do not fit that interface, and hiding
    them behind it would cost the retrieval quality this design depends on.
    """

    def __init__(self, cfg: QdrantSettings | None = None, client: QdrantClient | None = None):
        self.cfg = cfg or settings.qdrant
        self._client = client

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(
                url=self.cfg.url,
                grpc_port=self.cfg.grpc_port,
                prefer_grpc=self.cfg.prefer_grpc,
                api_key=self.cfg.api_key.get_secret_value() if self.cfg.api_key else None,
                timeout=self.cfg.timeout,
            )
        return self._client

    @property
    def collection(self) -> str:
        return self.cfg.collection

    # ───────────────────────────────────────────────── lifecycle

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self, *, recreate: bool = False) -> bool:
        """Create the collection and its payload indexes. Idempotent.

        Returns True if it created one, False if it was already there.
        """
        if self.exists():
            if not recreate:
                self._ensure_payload_indexes()
                return False
            logger.warning("Dropping existing collection %s", self.collection)
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self.cfg.dense_vector_name: models.VectorParams(
                    size=settings.embedding.dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=self.cfg.hnsw_m, ef_construct=self.cfg.hnsw_ef_construct
                    ),
                )
            },
            sparse_vectors_config=self._sparse_config(),
            quantization_config=self._quantization_config(),
            # Indexing during a large bulk load wastes work. Ingestion raises this
            # threshold, then lowers it to build the graph once at the end.
            optimizers_config=models.OptimizersConfigDiff(default_segment_number=2),
            on_disk_payload=True,
        )
        logger.info(
            "Created collection %s (dim=%d, quantization=%s)",
            self.collection,
            settings.embedding.dim,
            self.cfg.quantization,
        )
        self._ensure_payload_indexes()
        return True

    def _sparse_config(self) -> dict[str, models.SparseVectorParams]:
        # IDF modifier: Qdrant applies inverse-document-frequency weighting
        # server-side, which is what makes the sparse branch behave like a proper
        # lexical retriever rather than raw dot product.
        cfg = {
            self.cfg.lexical_vector_name: models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
        if settings.retrieval.retrieval_bm25_enabled:
            cfg[self.cfg.bm25_vector_name] = models.SparseVectorParams(modifier=models.Modifier.IDF)
        return cfg

    def _quantization_config(self) -> models.ScalarQuantization | None:
        if self.cfg.quantization != "scalar":
            return None
        # int8 cuts the resident vector footprint ~4x (12.3 GB -> 3.1 GB at 3M
        # chunks). Originals stay on disk and are used to rescore the top
        # candidates, so recall loss is negligible (ARCHITECTURE 7.2).
        return models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, quantile=0.99, always_ram=True
            )
        )

    def _ensure_payload_indexes(self) -> None:
        for fname, schema in PAYLOAD_INDEXES.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection, field_name=fname, field_schema=schema
                )
            except Exception as exc:  # already-exists is not an error worth raising
                logger.debug("Payload index %s: %s", fname, exc)

    def drop_collection(self) -> None:
        if self.exists():
            self.client.delete_collection(self.collection)
            logger.warning("Dropped collection %s", self.collection)

    # ───────────────────────────────────────────────── writes

    def upsert(
        self, points: Iterable[ChunkPoint], *, batch_size: int = 128, wait: bool = False
    ) -> int:
        """Upsert in batches. ``wait=False`` by default - bulk ingest should not
        block on each batch reaching the index."""
        batch: list[models.PointStruct] = []
        total = 0
        for point in points:
            batch.append(point.to_point_struct(self.cfg))
            if len(batch) >= batch_size:
                self.client.upsert(collection_name=self.collection, points=batch, wait=wait)
                total += len(batch)
                batch = []
        if batch:
            self.client.upsert(collection_name=self.collection, points=batch, wait=wait)
            total += len(batch)
        return total

    def mark_superseded(self, *, doc_version_id: str) -> None:
        """Supersede rather than delete - older wordings still answer older claims."""
        self.client.set_payload(
            collection_name=self.collection,
            payload={Payload.IS_SUPERSEDED: True},
            points=models.Filter(
                must=[
                    models.FieldCondition(
                        key=Payload.DOC_VERSION_ID, match=models.MatchValue(value=doc_version_id)
                    )
                ]
            ),
        )

    def delete_by_document(self, doc_id: str) -> None:
        """Hard delete. Only for a document removed at the owner's request."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key=Payload.DOC_ID, match=models.MatchValue(value=doc_id)
                        )
                    ]
                )
            ),
        )

    # ───────────────────────────────────────────────── reads

    def hybrid_search(
        self,
        *,
        dense_queries: Sequence[list[float]],
        lexical_queries: Sequence[SparseVector] | None = None,
        bm25_queries: Sequence[SparseVector] | None = None,
        query_filter: models.Filter | None = None,
        limit: int = 60,
        dense_prefetch_limit: int | None = None,
        sparse_prefetch_limit: int | None = None,
    ) -> list[SearchHit]:
        """One round trip: every variant x every branch, fused server-side by RRF.

        Each query variant (paraphrases, HyDE) contributes its own prefetch branch,
        so N variants x 2 branches become 2N branches in a single call rather than
        2N separate searches fused in Python.

        Note: Qdrant's server-side RRF uses its own internal constant. The
        configurable ``rrf_k`` applies to client-side fusion in
        ``app/retrieval/fusion.py``, used when results from separate calls must
        be merged.
        """
        dense_k = dense_prefetch_limit or settings.retrieval.retrieval_dense_top_k
        sparse_k = sparse_prefetch_limit or settings.retrieval.retrieval_sparse_top_k

        prefetch: list[models.Prefetch] = [
            models.Prefetch(
                query=list(vec),
                using=self.cfg.dense_vector_name,
                limit=dense_k,
                filter=query_filter,
                params=self._search_params(),
            )
            for vec in dense_queries
        ]

        for name, queries, klimit in (
            (self.cfg.lexical_vector_name, lexical_queries, sparse_k),
            (self.cfg.bm25_vector_name, bm25_queries, sparse_k),
        ):
            for sparse in queries or []:
                if not sparse.nnz:
                    continue
                prefetch.append(
                    models.Prefetch(
                        query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                        using=name,
                        limit=klimit,
                        filter=query_filter,
                    )
                )

        if not prefetch:
            raise ValueError("hybrid_search requires at least one query vector")

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [
            SearchHit(chunk_id=str(p.id), score=float(p.score or 0.0), payload=p.payload or {})
            for p in response.points
        ]

    def _search_params(self) -> models.SearchParams:
        params = models.SearchParams(hnsw_ef=self.cfg.search_ef)
        if self.cfg.quantization == "scalar":
            # Rescore the shortlist against the full-precision vectors on disk;
            # this is what makes int8 quantization cost almost no recall.
            params.quantization = models.QuantizationSearchParams(rescore=True, oversampling=2.0)
        return params

    def fetch(self, chunk_ids: Sequence[str]) -> list[SearchHit]:
        """Fetch points by id - used for parent expansion."""
        if not chunk_ids:
            return []
        records = self.client.retrieve(
            collection_name=self.collection,
            ids=[str(cid) for cid in chunk_ids],
            with_payload=True,
            with_vectors=False,
        )
        return [SearchHit(chunk_id=str(r.id), score=0.0, payload=r.payload or {}) for r in records]

    def scroll_by_filter(self, query_filter: models.Filter, *, limit: int = 100) -> list[SearchHit]:
        """Filter-only retrieval, no vector. Used to force-include the exclusions
        and definitions sections for coverage questions (ARCHITECTURE 8, step 6i)."""
        records, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [SearchHit(chunk_id=str(r.id), score=0.0, payload=r.payload or {}) for r in records]

    def count(self, query_filter: models.Filter | None = None) -> int:
        return self.client.count(
            collection_name=self.collection, count_filter=query_filter, exact=True
        ).count

    # ───────────────────────────────────────────────── ops

    def health(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant health check failed: %s", exc)
            return False

    def info(self) -> dict[str, Any]:
        if not self.exists():
            return {"exists": False, "collection": self.collection}
        info = self.client.get_collection(self.collection)
        return {
            "exists": True,
            "collection": self.collection,
            "status": str(info.status),
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "segments_count": info.segments_count,
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Process-wide singleton. The Qdrant client is thread-safe and pools its
    own connections, so one instance per process is correct."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
