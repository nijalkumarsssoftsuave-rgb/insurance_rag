"""Multi-branch prefetch: dense plus learned-sparse plus optional BM25.

The full Lane A retrieval path, in one place:

    expand -> encode (one batch) -> hybrid search -> rerank -> gate
           -> companions -> parent expansion -> pack

The embedder and reranker are blocking torch calls. Every one of them is pushed
to a worker thread, because holding the event loop for 400 ms of cross-encoder
inference would stall every other in-flight request on the same process.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from app.config import settings
from app.embeddings.bge_m3 import get_embedder
from app.logging import get_logger
from app.retrieval import expansion, filters, packing
from app.retrieval.packing import PackedContext
from app.retrieval.rerankers import RerankedHit, get_reranker
from app.retrieval.vectorstore import Payload, SearchHit, VectorStore, get_vector_store

log = get_logger(__name__)


@dataclass(slots=True)
class RetrievalOutcome:
    """Everything the graph and the Retrieval Lab need to explain an answer."""

    query: str
    context: PackedContext
    ranked: list[RerankedHit] = field(default_factory=list)
    candidates: list[SearchHit] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    top_score: float = 0.0
    below_threshold: bool = True
    broadened: bool = False
    timings_ms: dict[str, int] = field(default_factory=dict)

    @property
    def has_context(self) -> bool:
        return bool(self.context.blocks)

    @property
    def should_answer(self) -> bool:
        """The confidence gate.

        Abstention is a feature here. A wrong 'yes, that's covered' costs a claim
        payout and a complaint; 'I couldn't find that' costs a support minute
        (ARCHITECTURE 10.4).
        """
        return self.has_context and not self.below_threshold


class RetrievalPipeline:
    def __init__(
        self,
        store: VectorStore | None = None,
        embedder=None,
        reranker=None,
    ) -> None:
        self.store = store or get_vector_store()
        self._embedder = embedder
        self._reranker = reranker

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = get_reranker()
        return self._reranker

    async def retrieve(
        self,
        query: str,
        *,
        retrieval_filter: filters.RetrievalFilter | None = None,
        expand_query: bool = True,
        force_companions: bool = False,
        candidate_limit: int | None = None,
        allow_broadening: bool = True,
    ) -> RetrievalOutcome:
        """Run the pipeline once, retrying broadened if the gate rejects.

        The single retry drops soft filters and widens k. It is bounded at one:
        a second failure means the corpus does not contain the answer, and
        searching harder produces confident noise rather than a better answer.
        """
        rf = retrieval_filter or filters.RetrievalFilter()
        # Cross-encoder latency is linear in the candidate count and dominates the
        # budget on CPU, so this is the main retrieval dial - not an arbitrary
        # constant. See RerankerSettings.candidates for the measurements.
        candidate_limit = candidate_limit or settings.reranker.candidates
        outcome = await self._attempt(
            query,
            rf,
            expand_query=expand_query,
            force_companions=force_companions,
            candidate_limit=candidate_limit,
        )

        if outcome.should_answer or not allow_broadening:
            return outcome

        log.info(
            "Confidence gate rejected - retrying broadened",
            top_score=round(outcome.top_score, 4),
            threshold=settings.reranker.score_threshold,
        )
        retry = await self._attempt(
            query,
            rf.broadened(),
            expand_query=expand_query,
            force_companions=force_companions,
            candidate_limit=candidate_limit * 2,
        )
        retry.broadened = True
        # Keep whichever attempt actually scored better; broadening can dilute.
        return retry if retry.top_score > outcome.top_score else outcome

    async def _attempt(
        self,
        query: str,
        rf: filters.RetrievalFilter,
        *,
        expand_query: bool,
        force_companions: bool,
        candidate_limit: int,
    ) -> RetrievalOutcome:
        timings: dict[str, int] = {}
        clock = time.perf_counter

        # 1. Expansion -------------------------------------------------------
        t0 = clock()
        if expand_query:
            expanded = await expansion.expand(query)
            variants = expanded.all_variants
        else:
            variants = [query]
        timings["expand"] = int((clock() - t0) * 1000)

        # 2. Encode every variant in ONE forward pass ------------------------
        t0 = clock()
        embedded = await asyncio.to_thread(self.embedder.embed_queries, variants)
        timings["encode"] = int((clock() - t0) * 1000)

        # 3. Hybrid search: dense + learned-sparse, fused server-side ---------
        t0 = clock()
        query_filter = filters.build(rf)
        candidates = await asyncio.to_thread(
            self.store.hybrid_search,
            dense_queries=embedded.dense,
            lexical_queries=embedded.sparse or None,
            query_filter=query_filter,
            limit=candidate_limit,
        )
        timings["search"] = int((clock() - t0) * 1000)

        if not candidates:
            return RetrievalOutcome(
                query=query,
                context=PackedContext(),
                variants=variants,
                timings_ms=timings,
            )

        # 4. Rerank + confidence gate ----------------------------------------
        t0 = clock()
        ranked = await asyncio.to_thread(self.reranker.rerank, query, candidates)
        timings["rerank"] = int((clock() - t0) * 1000)

        top_score = ranked[0].score if ranked else 0.0
        below = top_score < settings.reranker.score_threshold

        # 5. Companions, parent expansion, packing ---------------------------
        t0 = clock()
        blocks = packing.expand_to_parents(self.store, ranked)

        if force_companions and blocks:
            doc_ids = [b.doc_id for b in blocks if b.doc_id][:3]
            if doc_ids:
                companions = await asyncio.to_thread(
                    self.store.scroll_by_filter,
                    filters.companion_filter(rf, doc_ids),
                    limit=10,
                )
                blocks = packing.add_companions(blocks, companions)

        context = packing.pack(blocks)
        timings["pack"] = int((clock() - t0) * 1000)

        log.debug(
            "Retrieval complete",
            variants=len(variants),
            candidates=len(candidates),
            blocks=len(context.blocks),
            top_score=round(top_score, 4),
            gated=below,
            timings=timings,
        )

        return RetrievalOutcome(
            query=query,
            context=context,
            ranked=ranked,
            candidates=candidates,
            variants=variants,
            top_score=top_score,
            below_threshold=below,
            timings_ms=timings,
        )

    async def retrieve_by_clause(
        self, clause_ref: str, rf: filters.RetrievalFilter
    ) -> list[SearchHit]:
        """Targeted lookup for the Lane B hand-off.

        A rejection carries a clause reference, so this is a filter hit rather
        than a similarity search - the exact clause is known.
        """
        from qdrant_client import models

        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=Payload.TENANT_ID, match=models.MatchValue(value=rf.tenant_id)
                ),
                models.FieldCondition(
                    key=Payload.SECTION_PATH, match=models.MatchText(text=clause_ref)
                ),
                models.FieldCondition(
                    key=Payload.IS_SUPERSEDED, match=models.MatchValue(value=False)
                ),
            ]
        )
        return await asyncio.to_thread(self.store.scroll_by_filter, query_filter, limit=5)


_pipeline: RetrievalPipeline | None = None


def get_pipeline() -> RetrievalPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RetrievalPipeline()
    return _pipeline
