"""The replay bundle written onto every turn.

Two things matter here and nothing else: that the bundle carries what a replay
needs, and that it never carries the identifiers the guard just stripped out.
"""

from __future__ import annotations

from app.api.v1.chat import TRACE_SCHEMA_VERSION, _replay_bundle

SECRET_EMAIL = "asha.rao@example.com"

RESULT = {
    "query_variants": ["is dental covered?", "does the policy pay for dental work?"],
    "top_score": 0.8977,
    "below_threshold": False,
    "broadened": False,
    "context_blocks": [
        {"chunk_id": "c1", "score": 0.8977, "section_path": "SECTION 4 - EXCLUSIONS > 4.11"},
        {"chunk_id": "c2", "score": 0.0031, "section_path": "SECTION 1 - DEFINITIONS > 1.3"},
    ],
    "route": {"product_name": "Family Health Optima", "is_coverage_question": True},
    "injection_severity": "none",
    "pii_mapping": {"[EMAIL_1]": SECRET_EMAIL},
    "verified": True,
    "verification_notes": ["groundedness_flaky"],
    "timings_ms": {"rerank": 2044},
}


def test_bundle_carries_every_retrieved_chunk_with_its_score() -> None:
    """`citations` records what the answer used; this must record what it could
    have used, or a trace cannot separate a retrieval failure from a generation one."""
    bundle = _replay_bundle(RESULT, redacted=True)
    chunks = bundle["retrieval"]["chunks"]
    assert [c["chunk_id"] for c in chunks] == ["c1", "c2"]
    assert all("score" in c for c in chunks)


def test_bundle_carries_the_decoding_parameters_and_filters() -> None:
    bundle = _replay_bundle(RESULT, redacted=True)
    assert bundle["llm"]["temperature"] is not None
    assert bundle["llm"]["model"]
    assert bundle["router"]["product_name"] == "Family Health Optima"
    assert bundle["retrieval"]["variants"] == RESULT["query_variants"]
    assert bundle["schema_version"] == TRACE_SCHEMA_VERSION


def test_bundle_records_placeholders_never_the_original_values() -> None:
    """The whole point of masking before write is that the trace cannot leak it back."""
    bundle = _replay_bundle(RESULT, redacted=True)
    assert bundle["guard"]["pii_placeholders"] == ["[EMAIL_1]"]
    assert SECRET_EMAIL not in str(bundle)


def test_bundle_survives_a_turn_that_never_reached_retrieval() -> None:
    """A blocked or claim-lookup turn has almost none of these keys."""
    bundle = _replay_bundle({}, redacted=False)
    assert bundle["retrieval"]["chunks"] == []
    assert bundle["guard"]["pii_masked_before_write"] is False
    assert bundle["router"]["intent"] is None
