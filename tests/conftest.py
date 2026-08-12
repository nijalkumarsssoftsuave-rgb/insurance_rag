"""Shared fixtures: test client, ephemeral Qdrant, seeded database."""

from __future__ import annotations

import sys
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.db.session import get_sync_engine  # noqa: E402
from app.retrieval.vectorstore import VectorStore  # noqa: E402


@pytest.fixture(scope="module")
def vector_store() -> Generator[VectorStore, None, None]:
    """A throwaway collection per test *module*.

    Module scope, not session: two modules sharing one collection means each
    sees the other's points, and a `count()` assertion that passes alone fails
    in a full run. Creating a collection is milliseconds; the expensive thing
    (model weights) is cached in a process-wide singleton and is unaffected.

    Never points at the configured collection: an integration test must not be
    able to drop the corpus someone spent hours embedding.
    """
    store = VectorStore()
    if not store.health():
        pytest.skip("Qdrant is not reachable - run `docker compose up -d qdrant`")

    store.cfg = settings.qdrant.model_copy(update={"collection": f"test_{uuid.uuid4().hex[:12]}"})
    store.ensure_collection()
    try:
        yield store
    finally:
        store.drop_collection()
        store.close()


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """A session inside an outer transaction that is always rolled back.

    Nothing a test writes can survive, even if the test commits - the enclosing
    transaction is discarded regardless.
    """
    engine = get_sync_engine()
    try:
        connection = engine.connect()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"Postgres is not reachable: {exc}")

    transaction = connection.begin()
    session = Session(bind=connection, expire_on_commit=False)
    try:
        yield session
    finally:
        session.close()
        # A test that provoked an IntegrityError leaves the transaction already
        # deassociated; rolling it back again warns rather than helps.
        if transaction.is_active:
            transaction.rollback()
        connection.close()
