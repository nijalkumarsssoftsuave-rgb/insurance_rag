"""Local BAAI/bge-reranker cross-encoder.

Which checkpoint to run is a hardware decision, not a preference:

* ``bge-reranker-base`` (278M) - the CPU-only pick.
* ``bge-reranker-v2-m3`` (568M) - better and multilingual, but several seconds
  per batch on CPU. Use it with a GPU, or export it to ONNX INT8
  (ARCHITECTURE 4.3).

Both are Apache-2.0. The Jina rerankers score comparably but are CC-BY-NC, which
rules them out for a product.

**Implemented directly on transformers, not FlagEmbedding.** FlagEmbedding's
``FlagReranker`` calls ``tokenizer.prepare_for_model``, which transformers 5.x
removed, so it raises on any current install. Scoring a cross-encoder is a
tokenize-forward-sigmoid loop; owning those forty lines is cheaper than pinning
the whole project to transformers 4.x, and it makes the 0..1 score contract
explicit here rather than a flag we hope an upstream library honours.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from app.config import ensure_model_cache, settings
from app.logging import get_logger
from app.retrieval.rerankers.base import RerankedHit, Reranker
from app.retrieval.vectorstore import Payload, SearchHit

log = get_logger(__name__)

# Cross-encoders truncate long inputs anyway; score the child chunk that was
# actually retrieved rather than a parent that would be cut mid-clause.
MAX_PAIR_TOKENS = 512


def pair_text(hit: SearchHit) -> str:
    """What the cross-encoder actually reads: breadcrumb, then chunk text.

    Ingestion prefixes the section breadcrumb onto ``embedded_text`` for the
    bi-encoder, but the cross-encoder - the one model that reads query and chunk
    *together* - was handed the bare ``payload["text"]``. A table chunk is
    pipe-markdown with no prose in it, so without its heading there was nothing
    for the model to match a question against.

    Measured before/after on the golden set: ``eval/runs/RESULTS.md``. Note the
    regression recorded there as well as the gain - this lifts ranking and
    compresses scores, and the abstention gate reads the scores.
    """
    breadcrumb = " | ".join(
        str(value)
        for key in (Payload.PRODUCT_NAME, Payload.SECTION_PATH)
        if (value := hit.payload.get(key))
    )
    return breadcrumb + "\n" + hit.text if breadcrumb else hit.text


class BGEReranker:
    """Cross-encoder reranker with lazy, thread-safe model loading."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self._model_name = model_name or settings.reranker.model
        self._device = device or settings.reranker.device
        self._batch_size = batch_size
        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            ensure_model_cache()
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            started = time.perf_counter()
            log.info("Loading reranker", model=self._model_name, device=self._device)

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
            model.eval()
            if self._device != "cpu":
                model = model.to(self._device).half()
            self._model = model

            torch.set_grad_enabled(False)
            log.info("Reranker ready", seconds=round(time.perf_counter() - started, 1))

    def rerank(
        self, query: str, hits: list[SearchHit], *, top_n: int | None = None
    ) -> list[RerankedHit]:
        if not hits:
            return []

        limit = top_n or settings.reranker.top_n
        self.load()
        import torch

        pairs = [(query, pair_text(hit)) for hit in hits]
        started = time.perf_counter()
        scores: list[float] = []

        with self._lock, torch.no_grad():
            for i in range(0, len(pairs), self._batch_size):
                batch = pairs[i : i + self._batch_size]
                inputs = self._tokenizer(
                    [p[0] for p in batch],
                    [p[1] for p in batch],
                    padding=True,
                    truncation=True,
                    max_length=MAX_PAIR_TOKENS,
                    return_tensors="pt",
                )
                if self._device != "cpu":
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                logits = self._model(**inputs).logits.view(-1).float()
                # bge rerankers emit a single relevance logit. Sigmoid maps it to
                # 0..1, which is what `score_threshold` is calibrated against -
                # comparing a raw logit to 0.30 would disable the abstention gate.
                scores.extend(torch.sigmoid(logits).cpu().tolist())

        ranked = [
            RerankedHit(hit=hit, score=float(score), retrieval_rank=rank, retrieval_score=hit.score)
            for rank, (hit, score) in enumerate(zip(hits, scores, strict=True))
        ]
        ranked.sort(key=lambda r: r.score, reverse=True)

        # How far each chunk moved. The Retrieval Lab renders this, and it is the
        # fastest way to see whether reranking is earning its latency.
        for new_rank, item in enumerate(ranked):
            item.rank_delta = item.retrieval_rank - new_rank

        log.debug(
            "Reranked",
            candidates=len(hits),
            kept=min(limit, len(ranked)),
            ms=round((time.perf_counter() - started) * 1000),
            top_score=round(ranked[0].score, 4) if ranked else None,
        )
        return ranked[:limit]

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._tokenizer = None


class NoOpReranker:
    """Pass-through. Preserves retrieval order.

    Not a stub: the ablation grid needs a run with reranking disabled to measure
    what the cross-encoder actually buys (ARCHITECTURE 11.3).
    """

    @property
    def model_name(self) -> str:
        return "noop"

    def rerank(
        self, query: str, hits: list[SearchHit], *, top_n: int | None = None
    ) -> list[RerankedHit]:
        limit = top_n or settings.reranker.top_n
        return [
            RerankedHit(hit=hit, score=hit.score, retrieval_rank=rank, retrieval_score=hit.score)
            for rank, hit in enumerate(hits[:limit])
        ]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker() if settings.reranker.enabled else NoOpReranker()
    return _reranker
