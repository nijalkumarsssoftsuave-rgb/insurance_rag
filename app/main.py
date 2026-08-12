"""FastAPI application: lifespan, middleware, router mounting."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import settings
from app.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    log.info(
        "Starting API",
        env=settings.app.app_env,
        embedding_model=settings.embedding.model,
        reranker=settings.reranker.model,
        llm=settings.llm.llm_model,
        collection=settings.qdrant.collection,
    )
    # Models are NOT warmed here on purpose: loading bge-m3 costs ~18 s and 2.3 GB,
    # and an API process that only serves uploads never needs it. The first
    # request that does pays for it, once.
    yield
    log.info("Shutting down API")


app = FastAPI(
    title="Insurance Claims RAG",
    description="Policy Q&A and claim status, retrieval-augmented.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # the Streamlit UI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "insurance-rag", "docs": "/docs", "health": "/api/v1/health"}
