"""SQLAlchemy engine and session factory.

Two engines share one DSN, because the codebase has two execution models:

* **async** - FastAPI request handlers and the LangGraph nodes they drive.
* **sync**  - Celery workers, Alembic and the scripts in ``scripts/``.

psycopg3 backs both, so ``postgresql+psycopg://`` works unchanged for each.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings


def _ensure_compatible_event_loop() -> None:
    """psycopg3 async cannot run on Windows' default ProactorEventLoop.

    Python 3.8+ defaults to Proactor on Windows, and psycopg raises
    ``InterfaceError`` rather than degrading - so without this, every async
    database call fails on a Windows dev machine while the sync path (Celery,
    Alembic, scripts) works fine, which is a confusing way to find out.

    SelectorEventLoop caps out around 512 sockets and cannot spawn subprocesses.
    Neither limit is reachable here, and swapping to a second async driver just
    to avoid one policy call is a worse trade.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


_ensure_compatible_event_loop()


def _async_url(url: str) -> str:
    """psycopg3 speaks async natively; only the sqlite fallback needs rewriting."""
    if url.startswith("sqlite:"):
        return url.replace("sqlite:", "sqlite+aiosqlite:", 1)
    return url


@lru_cache(maxsize=1)
def get_async_engine() -> AsyncEngine:
    return create_async_engine(
        _async_url(settings.storage.database_url),
        echo=settings.storage.db_echo,
        pool_size=settings.storage.db_pool_size,
        max_overflow=settings.storage.db_max_overflow,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_sync_engine() -> Engine:
    return create_engine(
        settings.storage.sync_database_url,
        echo=settings.storage.db_echo,
        pool_size=settings.storage.db_pool_size,
        max_overflow=settings.storage.db_max_overflow,
        pool_pre_ping=True,
    )


@lru_cache(maxsize=1)
def get_async_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=get_async_engine(), expire_on_commit=False, autoflush=False)


@lru_cache(maxsize=1)
def get_sync_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_sync_engine(), expire_on_commit=False, autoflush=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency. Commits on success, rolls back on any exception."""
    async with get_async_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def async_session_scope() -> AsyncGenerator[AsyncSession, None]:
    """Same contract as ``get_db`` for code outside the request cycle."""
    async with get_async_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Sync equivalent for Celery tasks and scripts."""
    session = get_sync_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


async def check_database() -> bool:
    """Readiness probe helper."""
    from sqlalchemy import text

    try:
        async with get_async_session_factory()() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
