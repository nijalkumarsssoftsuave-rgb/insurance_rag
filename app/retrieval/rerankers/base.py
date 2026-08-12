"""Reranker protocol - implementations swap by config.

Reranking is the highest-ROI quality lever in the pipeline: a bi-encoder had to
compress each chunk into one vector before it ever saw the query, while a
cross-encoder reads query and chunk together (ARCHITECTURE 4.3).

Scores are **normalized to 0..1** by contract. The abstention threshold
``RERANKER_SCORE_THRESHOLD`` is meaningless against raw logits, and an
implementation that returned them would silently disable the confidence gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.retrieval.vectorstore import SearchHit


@dataclass(slots=True)
class RerankedHit:
    """A hit with its cross-encoder score, keeping the retrieval score for analysis."""

    hit: SearchHit
    score: float
    retrieval_rank: int
    retrieval_score: float
    # How far reranking moved this chunk: positive means it climbed. Set after
    # sorting. A real field rather than an attribute set on the fly, because
    # `slots=True` forbids the latter - and it is what the Retrieval Lab renders
    # to show whether the cross-encoder is earning its latency.
    rank_delta: int = 0

    @property
    def chunk_id(self) -> str:
        return self.hit.chunk_id

    @property
    def text(self) -> str:
        return self.hit.text


@runtime_checkable
class Reranker(Protocol):
    """Implementations: ``BGEReranker``, ``NoOpReranker``."""

    @property
    def model_name(self) -> str: ...

    def rerank(
        self, query: str, hits: list[SearchHit], *, top_n: int | None = None
    ) -> list[RerankedHit]:
        """Score every hit against the query and return the best ``top_n``,
        sorted descending. Scores must be in 0..1."""
        ...
