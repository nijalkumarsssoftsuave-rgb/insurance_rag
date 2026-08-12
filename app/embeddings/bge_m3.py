"""BAAI/bge-m3 - dense 1024d and learned-sparse in a single pass.

One forward pass yields both retrieval branches, which is why this project needs
no separate BM25 model or index. The sparse output is *learned* term weighting,
not corpus statistics, so it behaves like SPLADE: exact-token matching that still
knows which tokens carry meaning (ARCHITECTURE 4.2).

Two operational details that matter more than they look:

* ``max_length`` is 512, not the model's 8192 default. Chunks are ~400 tokens and
  attention cost grows superlinearly, so the default would make ingestion several
  times slower for nothing.
* The model is loaded lazily behind a lock. It is ~2.3 GB resident and both the
  API and the Celery worker import this module; eager loading at import would pay
  that cost in processes that never embed anything.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from app.config import ensure_model_cache, settings
from app.embeddings.base import EmbeddingResult, SparseVector
from app.logging import get_logger

log = get_logger(__name__)

# Sparse vectors are dominated by near-zero weights that cost storage and add
# nothing to the score. Dropping them typically removes over half the terms.
MIN_SPARSE_WEIGHT = 0.01


class BGEM3Embedder:
    """Thread-safe wrapper around ``BGEM3FlagModel``."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        use_fp16: bool | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        cfg = settings.embedding
        self._model_name = model_name or cfg.model
        self._device = device or cfg.device
        self._max_length = max_length or cfg.max_length
        self._batch_size = batch_size or cfg.batch_size
        # fp16 is a CUDA optimization; on CPU it is slower and less accurate.
        self._use_fp16 = (use_fp16 if use_fp16 is not None else cfg.use_fp16) and (
            self._device != "cpu"
        )
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return settings.embedding.dim

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        """Idempotent. First call downloads ~2.3 GB if the cache is cold."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:  # another thread won the race
                return
            ensure_model_cache()
            from FlagEmbedding import BGEM3FlagModel

            started = time.perf_counter()
            log.info(
                "Loading embedder",
                model=self._model_name,
                device=self._device,
                fp16=self._use_fp16,
            )
            self._model = BGEM3FlagModel(
                self._model_name,
                normalize_embeddings=True,  # required: the index uses cosine distance
                use_fp16=self._use_fp16,
                devices=self._device,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,  # ~1.6 MB/chunk - rejected, ARCHITECTURE 4.2
            )
            log.info("Embedder ready", seconds=round(time.perf_counter() - started, 1))

    def _encode(self, texts: list[str], *, batch_size: int | None) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(dense=[], sparse=[])

        self.load()
        # FlagEmbedding's encode is not documented as thread-safe and holds
        # internal batching state, so serialize it. Concurrency belongs at the
        # worker level, not inside one model instance.
        with self._lock:
            output = self._model.encode(
                texts,
                batch_size=batch_size or self._batch_size,
                max_length=self._max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )

        dense = [vec.tolist() for vec in output["dense_vecs"]]
        sparse = [_to_sparse(w) for w in output.get("lexical_weights", [])]
        return EmbeddingResult(dense=dense, sparse=sparse)

    def embed_documents(
        self, texts: list[str], *, batch_size: int | None = None
    ) -> EmbeddingResult:
        """Encode chunk text for indexing. Pass the breadcrumb-prefixed text."""
        return self._encode(texts, batch_size=batch_size)

    def embed_queries(self, texts: list[str]) -> EmbeddingResult:
        """Encode queries.

        Batch every expansion variant into one call. Each variant is a local
        forward pass, not a cheap API hop, so encoding them serially multiplies
        the latency budget (ARCHITECTURE 8).
        """
        return self._encode(texts, batch_size=max(len(texts), 1))

    def embed_query(self, text: str) -> tuple[list[float], SparseVector]:
        result = self._encode([text], batch_size=1)
        sparse = result.sparse[0] if result.sparse else SparseVector([], [])
        return result.dense[0], sparse

    def unload(self) -> None:
        """Release the weights. Used by ablation runs that swap models."""
        with self._lock:
            self._model = None


def _to_sparse(weights: dict[str, float]) -> SparseVector:
    """FlagEmbedding returns ``{token_id_as_string: weight}``.

    Qdrant wants parallel index/value arrays of ints and floats, sorted by index
    for stable comparison.
    """
    items = [
        (int(token_id), float(weight))
        for token_id, weight in weights.items()
        if float(weight) >= MIN_SPARSE_WEIGHT
    ]
    items.sort(key=lambda kv: kv[0])
    return SparseVector(
        indices=[i for i, _ in items],
        values=[v for _, v in items],
    )


_embedder: BGEM3Embedder | None = None


def get_embedder() -> BGEM3Embedder:
    """Process-wide singleton - one 2.3 GB copy of the weights, not one per call."""
    global _embedder
    if _embedder is None:
        _embedder = BGEM3Embedder()
    return _embedder
