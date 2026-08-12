"""Liveness and readiness probes for every dependency.

Readiness reports each dependency separately rather than a single boolean: when
ingestion stalls, "which of Postgres, Qdrant and the model cache is down" is the
first question, and a bare 503 does not answer it.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.db.session import check_database
from app.retrieval.vectorstore import get_vector_store

router = APIRouter(tags=["health"])


class Health(BaseModel):
    status: str
    database: bool
    qdrant: bool
    collection: str
    points: int | None = None
    embedding_model: str
    llm_model: str


@router.get("/health", response_model=Health)
async def health() -> Health:
    db_ok = await check_database()

    store = get_vector_store()
    qdrant_ok = store.health()
    points = None
    if qdrant_ok:
        try:
            points = store.info().get("points_count")
        except Exception:
            qdrant_ok = False

    return Health(
        status="ok" if (db_ok and qdrant_ok) else "degraded",
        database=db_ok,
        qdrant=qdrant_ok,
        collection=settings.qdrant.collection,
        points=points,
        embedding_model=settings.embedding.model,
        llm_model=settings.llm.llm_model,
    )
