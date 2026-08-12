"""Aggregates all v1 routers under a single APIRouter."""

from fastapi import APIRouter

from app.api.v1 import chat, documents, health

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)

__all__ = ["api_router"]
