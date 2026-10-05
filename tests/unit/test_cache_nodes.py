"""cache_check_node / cache_store_node (Week 11) - stubbed against
app.cache.semantic so these stay fast, deterministic unit tests rather than
depending on a live Valkey instance or the real embedder.
"""

from __future__ import annotations

import uuid

from app.core.enums import UserRole
from app.graph.nodes import cache as cache_nodes
from app.security.authz import AuthSubject

SUBJECT = AuthSubject(
    user_id=uuid.uuid4(), tenant_id="default", role=UserRole.CUSTOMER, policy_holder_id=None
)


def _state(**extra) -> dict:
    return {
        "subject": SUBJECT,
        "question": "Is dental treatment covered?",
        "standalone_question": "Is dental treatment covered?",
        **extra,
    }


async def test_cache_check_returns_the_hit_and_marks_cache_hit(monkeypatch):
    cached = {"answer": "Dental is excluded.", "citations": [], "confidence": 1.0}

    async def fake_lookup(question, *, tenant_id):  # noqa: ARG001
        return cached

    monkeypatch.setattr(cache_nodes.semantic, "lookup", fake_lookup)

    result = await cache_nodes.cache_check_node(_state())

    assert result["cache_hit"] is True
    assert result["answer"] == "Dental is excluded."
    assert "cache_check" in result["timings_ms"]


async def test_cache_check_miss_sets_nothing_but_timing(monkeypatch):
    async def fake_lookup(question, *, tenant_id):  # noqa: ARG001
        return None

    monkeypatch.setattr(cache_nodes.semantic, "lookup", fake_lookup)

    result = await cache_nodes.cache_check_node(_state())

    assert "cache_hit" not in result
    assert "answer" not in result
    assert "cache_check" in result["timings_ms"]


async def test_cache_store_skips_a_cache_hit(monkeypatch):
    calls = []

    async def fake_store(question, *, tenant_id, result):  # noqa: ARG001
        calls.append(question)

    monkeypatch.setattr(cache_nodes.semantic, "store", fake_store)

    await cache_nodes.cache_store_node(
        _state(cache_hit=True, verified=True, abstained=False, answer="x")
    )

    assert calls == []  # already served from cache - do not re-cache it


async def test_cache_store_skips_an_unverified_or_abstained_answer(monkeypatch):
    calls = []

    async def fake_store(question, *, tenant_id, result):  # noqa: ARG001
        calls.append(question)

    monkeypatch.setattr(cache_nodes.semantic, "store", fake_store)

    await cache_nodes.cache_store_node(_state(verified=False, abstained=False, answer="x"))
    await cache_nodes.cache_store_node(_state(verified=True, abstained=True, answer="x"))

    assert calls == []


async def test_cache_store_writes_a_good_answer(monkeypatch):
    calls = []

    async def fake_store(question, *, tenant_id, result):
        calls.append((question, tenant_id, result))

    monkeypatch.setattr(cache_nodes.semantic, "store", fake_store)

    await cache_nodes.cache_store_node(
        _state(
            verified=True,
            abstained=False,
            answer="Dental is excluded unless accident-related.",
            citations=[{"chunk_id": "c1"}],
            confidence=0.9,
        )
    )

    assert len(calls) == 1
    question, tenant_id, result = calls[0]
    assert question == "Is dental treatment covered?"
    assert tenant_id == "default"
    assert result["answer"] == "Dental is excluded unless accident-related."
    assert result["verified"] is True
    assert result["abstained"] is False
