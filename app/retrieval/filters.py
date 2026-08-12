"""Builds Qdrant metadata filters including effective-date windows.

Filtering is what makes retrieval *correct* in insurance, not merely relevant. The
date-of-loss filter in particular is a correctness control: a claim is adjudicated
under the wording in force when the loss occurred, so answering a 2022 incident
from today's wording produces a confidently wrong, legally exposed answer
(ARCHITECTURE 5.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from qdrant_client import models

from app.core.enums import ChunkKind, DocType
from app.retrieval.vectorstore import TS_MIN, Payload, to_ts


@dataclass(slots=True)
class RetrievalFilter:
    """Everything that can narrow the candidate pool before scoring."""

    tenant_id: str = "default"
    insurer: str | None = None
    product_name: str | None = None
    doc_types: list[DocType] = field(default_factory=list)
    language: str | None = None
    doc_ids: list[str] = field(default_factory=list)
    # When set, only wordings in force on this date are retrieved.
    date_of_loss: date | None = None
    include_superseded: bool = False
    # Children are what carry embeddings; parents are fetched by id afterwards.
    kinds: list[ChunkKind] = field(
        default_factory=lambda: [ChunkKind.CHILD, ChunkKind.TABLE, ChunkKind.TABLE_SUMMARY]
    )

    def broadened(self) -> RetrievalFilter:
        """The retry filter.

        When the confidence gate fires, the most likely cause is an over-narrow
        filter - a product name the user phrased differently, or a date window
        that excluded the right document. Drop the soft constraints, keep tenant
        isolation, which is never negotiable.
        """
        return RetrievalFilter(
            tenant_id=self.tenant_id,
            language=self.language,
            include_superseded=self.include_superseded,
            kinds=self.kinds,
        )


def build(f: RetrievalFilter) -> models.Filter:
    must: list[models.Condition] = [
        models.FieldCondition(key=Payload.TENANT_ID, match=models.MatchValue(value=f.tenant_id))
    ]

    if not f.include_superseded:
        must.append(
            models.FieldCondition(key=Payload.IS_SUPERSEDED, match=models.MatchValue(value=False))
        )

    if f.kinds:
        must.append(
            models.FieldCondition(
                key=Payload.KIND, match=models.MatchAny(any=[k.value for k in f.kinds])
            )
        )

    if f.insurer:
        must.append(
            models.FieldCondition(key=Payload.INSURER, match=models.MatchValue(value=f.insurer))
        )

    if f.product_name:
        must.append(
            models.FieldCondition(
                key=Payload.PRODUCT_NAME, match=models.MatchValue(value=f.product_name)
            )
        )

    if f.doc_types:
        must.append(
            models.FieldCondition(
                key=Payload.DOC_TYPE, match=models.MatchAny(any=[d.value for d in f.doc_types])
            )
        )

    if f.language:
        must.append(
            models.FieldCondition(key=Payload.LANGUAGE, match=models.MatchValue(value=f.language))
        )

    if f.doc_ids:
        must.append(models.FieldCondition(key=Payload.DOC_ID, match=models.MatchAny(any=f.doc_ids)))

    if f.date_of_loss is not None:
        # Sentinels mean open-ended windows compare correctly without a null
        # branch: a document with no dates has from=TS_MIN and to=TS_MAX and so
        # matches every date, which is the safe default.
        loss_ts = to_ts(f.date_of_loss, default=TS_MIN)
        must.append(
            models.FieldCondition(key=Payload.EFFECTIVE_FROM_TS, range=models.Range(lte=loss_ts))
        )
        must.append(
            models.FieldCondition(key=Payload.EFFECTIVE_TO_TS, range=models.Range(gte=loss_ts))
        )

    return models.Filter(must=must)


def companion_filter(f: RetrievalFilter, doc_ids: list[str]) -> models.Filter:
    """Force-include exclusions and definitions from the documents already hit.

    A coverage answer that ignores an exclusion clause is the number-one failure
    mode of insurance RAG. Similarity search will not reliably surface the
    exclusions section when the question is phrased positively ("is dental
    covered?"), so it is fetched by filter rather than by score
    (ARCHITECTURE 6.2, 8 step 6i).
    """
    return models.Filter(
        must=[
            models.FieldCondition(
                key=Payload.TENANT_ID, match=models.MatchValue(value=f.tenant_id)
            ),
            models.FieldCondition(key=Payload.DOC_ID, match=models.MatchAny(any=doc_ids)),
            models.FieldCondition(key=Payload.IS_SUPERSEDED, match=models.MatchValue(value=False)),
        ],
        should=[
            models.FieldCondition(key=Payload.SECTION_PATH, match=models.MatchText(text=term))
            for term in ("Exclusion", "not covered", "Definition", "Waiting")
        ],
    )
