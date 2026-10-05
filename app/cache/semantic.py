"""Semantic query cache: embed the question, compare against recently-cached
questions by cosine similarity, serve a hit without touching retrieval,
expansion, generation or verification - the four places Week 11's cost
report found 93% of a Lane A turn's $ cost concentrated (route 11%,
retrieve_expand 7%, generate 43%, verify 39%).

Built on pieces that already existed and were never connected: the embedder
retrieval already loads (one query embedding on a miss, nothing new to run),
Valkey (docker-compose's `valkey` service, already running for Celery), and
`CacheSettings.threshold`/`ttl_seconds` (already configured in app/config.py,
unread by any code until now). This file itself was a docstring-only stub
before Week 11 - the same shape as `app/claims/tools.py` before Week 7 and
`app/security/injection.py::scan_document` before Week 8.

Scoped to Lane A document-QA answers only, never claim status. A cached
entry carries no customer identity - policy wording reads the same for every
policyholder on that product - so it is safe to serve to a different
customer who asks a similar-enough question. Writing a claim answer in here
would mean serving one customer's claim data to whoever asks next, which is
the cross-tenant leak ARCHITECTURE 10.1 exists to prevent. Enforced entirely
by *who calls this module*, not by a filter inside it: `generate_node` only
ever runs on the Lane A (document-QA) path, so nothing claim-shaped ever
reaches `store()`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time

from app.config import settings
from app.logging import get_logger

log = get_logger(__name__)

_INDEX_KEY_PREFIX = "semantic_cache:index"
_ENTRY_KEY_PREFIX = "semantic_cache:entry"
# Linear scan over this many recent entries per lookup. Valkey has no vector
# index (that is a RediSearch module, not present in the plain image this
# project runs) - at the entry count a single-tenant POC corpus produces,
# comparing against a capped, recent window in Python is fast enough that
# building or deploying a vector index for it would be solving a scale
# problem this app does not have.
_MAX_SCAN = 200

_client = None


def _redis():
    global _client
    if _client is None:
        from redis.asyncio import Redis

        _client = Redis.from_url(settings.storage.redis_url, decode_responses=True)
    return _client


def _dot(a: list[float], b: list[float]) -> float:
    """Cosine similarity via a plain dot product - both vectors are already
    unit-normalized by the embedder (`normalize_embeddings=True`), so this
    *is* cosine similarity, not an approximation of it."""
    return sum(x * y for x, y in zip(a, b, strict=True))


async def lookup(question: str, *, tenant_id: str) -> dict | None:
    """The cached turn's result dict on a semantic hit, else None.

    Never raises into the caller: a cache failure must degrade to "run the
    real pipeline", not take the turn down - the same contract
    `app/security/audit.py::record` makes for the audit trail.
    """
    if not settings.cache.enabled:
        return None

    try:
        from app.embeddings.bge_m3 import get_embedder

        r = _redis()
        index_key = f"{_INDEX_KEY_PREFIX}:{tenant_id}"
        entry_keys = await r.lrange(index_key, 0, _MAX_SCAN - 1)
        if not entry_keys:
            return None

        query_vec, _sparse = await asyncio.to_thread(get_embedder().embed_query, question)

        raw_entries = await r.mget(entry_keys)
        best_score, best_result = 0.0, None
        for raw in raw_entries:
            if not raw:
                continue
            entry = json.loads(raw)
            score = _dot(query_vec, entry["embedding"])
            if score > best_score:
                best_score, best_result = score, entry["result"]

        if best_result is not None and best_score >= settings.cache.threshold:
            log.info("Semantic cache hit", score=round(best_score, 4), tenant_id=tenant_id)
            return best_result
    except Exception as exc:
        log.warning("Semantic cache lookup failed, running the real pipeline", error=str(exc))
    return None


async def store(question: str, *, tenant_id: str, result: dict) -> None:
    """Cache one Lane A answer. Call only with a verified, non-abstained
    policy_qa result - the caller's job, not this module's, to check."""
    if not settings.cache.enabled:
        return

    try:
        from app.embeddings.bge_m3 import get_embedder

        r = _redis()
        query_vec, _sparse = await asyncio.to_thread(get_embedder().embed_query, question)

        # Content-addressed, not question-addressed: two near-duplicate
        # questions should both get their own entry (that is the point of a
        # similarity lookup), so the key only needs to be stable enough to
        # dedupe an exact repeat, not to collide semantically-close ones.
        digest = hashlib.sha256(question.strip().casefold().encode()).hexdigest()[:16]
        entry_key = f"{_ENTRY_KEY_PREFIX}:{tenant_id}:{digest}"
        index_key = f"{_INDEX_KEY_PREFIX}:{tenant_id}"

        payload = json.dumps({"embedding": query_vec, "result": result, "cached_at": time.time()})
        ttl = settings.cache.ttl_seconds
        await r.set(entry_key, payload, ex=ttl)
        await r.lrem(index_key, 0, entry_key)
        await r.lpush(index_key, entry_key)
        await r.ltrim(index_key, 0, _MAX_SCAN - 1)
        await r.expire(index_key, ttl)
    except Exception as exc:
        log.warning("Semantic cache store failed, answer still served", error=str(exc))
