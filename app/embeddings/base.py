"""Embedder protocol covering dense and sparse representations.

Only the types and the protocol live here. The concrete ``bge-m3`` implementation
lands in the ingestion step - but the vector store needs this contract now, and
defining it first is what keeps the index schema and the encoder from drifting.

bge-m3 emits dense and learned-sparse vectors from a single forward pass, so the
protocol returns both together rather than exposing two independent calls that
callers could accidentally invoke twice (ARCHITECTURE 4.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(slots=True, frozen=True)
class SparseVector:
    """A learned-sparse vector: term ids paired with their weights."""

    indices: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"sparse vector length mismatch: {len(self.indices)} indices, "
                f"{len(self.values)} values"
            )

    @property
    def nnz(self) -> int:
        return len(self.indices)


@dataclass(slots=True)
class EmbeddingResult:
    """One forward pass over a batch of texts."""

    dense: list[list[float]]
    sparse: list[SparseVector] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.dense)


@runtime_checkable
class Embedder(Protocol):
    """Implementations: ``app.embeddings.bge_m3.BGEM3Embedder``."""

    @property
    def model_name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        """Encode chunk text for indexing."""
        ...

    def embed_queries(self, texts: list[str]) -> EmbeddingResult:
        """Encode queries. Batch every expansion variant into one call -
        each variant is a local forward pass, not a cheap API hop."""
        ...
