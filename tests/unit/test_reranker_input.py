"""What the cross-encoder is handed.

`pair_text` is small but branchy - two optional payload fields, three degenerate
cases - and it sits on the path every reranked chunk takes. A missing heading
silently returns a table chunk to being unscoreable pipe-markdown, which is the
exact failure this function exists to fix (see `eval/runs/RESULTS.md`).
"""

from __future__ import annotations

from app.retrieval.rerankers.bge_reranker import pair_text
from app.retrieval.vectorstore import Payload, SearchHit

TABLE = "| 3 consecutive claim-free years | 35% |"


def hit(**payload) -> SearchHit:
    return SearchHit(chunk_id="c", score=1.0, payload={Payload.TEXT: TABLE} | payload)


def test_both_fields_present() -> None:
    out = pair_text(hit(**{Payload.PRODUCT_NAME: "Motor Shield", Payload.SECTION_PATH: "SECTION 4 - NCB"}))
    assert out.splitlines() == ["Motor Shield | SECTION 4 - NCB", TABLE]


def test_no_product_name_leaves_no_stray_separator() -> None:
    """SOP and circular payloads carry `product_name: null`."""
    out = pair_text(hit(**{Payload.SECTION_PATH: "SECTION 2 - REJECTION REASON CODES"}))
    assert out.splitlines() == ["SECTION 2 - REJECTION REASON CODES", TABLE]


def test_no_section_path() -> None:
    assert pair_text(hit(**{Payload.PRODUCT_NAME: "Motor Shield"})).splitlines() == ["Motor Shield", TABLE]


def test_neither_field_falls_back_to_bare_text() -> None:
    assert pair_text(hit()) == TABLE
    assert pair_text(hit(**{Payload.PRODUCT_NAME: "", Payload.SECTION_PATH: ""})) == TABLE


def test_chunk_text_is_never_dropped() -> None:
    """Whatever the prefix does, the retrieved text must still be in there."""
    for payload in ({}, {Payload.PRODUCT_NAME: "X"}, {Payload.SECTION_PATH: "Y"},
                    {Payload.PRODUCT_NAME: "X", Payload.SECTION_PATH: "Y"}):
        assert TABLE in pair_text(hit(**payload))
