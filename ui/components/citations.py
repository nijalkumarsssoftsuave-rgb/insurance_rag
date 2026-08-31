"""Expandable citation cards linking back to source clauses.

A citation the reader cannot see the text of is a reference, not evidence. The
whole claim this system makes is that an answer can be checked against the
wording, so the clause text is one click away rather than a chunk id.
"""

from __future__ import annotations

import streamlit as st


def render_citations(citations: list[dict], *, key: str = "") -> None:
    if not citations:
        return

    # Force-included exclusions are marked. They did not match the question by
    # similarity - they were pulled in because a coverage answer that ignores an
    # exclusion is the failure mode that matters most.
    scored = [c for c in citations if not c.get("is_companion")]
    companions = [c for c in citations if c.get("is_companion")]

    label = f"Sources ({len(citations)})"
    with st.expander(label, expanded=False):
        for i, citation in enumerate(scored, start=1):
            _card(i, citation)
        if companions:
            st.caption("Also checked - exclusions and definitions for this policy")
            for i, citation in enumerate(companions, start=len(scored) + 1):
                _card(i, citation, muted=True)


def _card(index: int, citation: dict, *, muted: bool = False) -> None:
    section = citation.get("section_path") or "Unlabelled section"
    product = citation.get("product_name")
    page = citation.get("page_no")
    score = citation.get("score")

    header = f"**{index}. {section}**"
    st.markdown(header)

    bits = []
    if product:
        bits.append(product)
    if page:
        bits.append(f"page {page}")
    if score is not None and not muted:
        bits.append(f"relevance {score:.2f}")
    if bits:
        st.caption(" · ".join(bits))

    if snippet := citation.get("snippet"):
        st.markdown(
            f"<div style='border-left:3px solid #1E4B7A;padding:6px 12px;"
            f"margin:4px 0 10px 0;color:#444;font-size:0.86rem;"
            f"background:#F6F8FB;white-space:pre-wrap'>{_escape(snippet)}</div>",
            unsafe_allow_html=True,
        )


def _escape(text: str) -> str:
    """Retrieved document text is untrusted - it must never render as markup."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").strip()


def render_inspection(turn: dict) -> None:
    """The retrieval inspection view: what was fetched, scored and used.

    ``citations`` are the blocks the model actually cited; ``context_blocks`` is
    everything that reached it. Rendering both side by side is what separates the
    two failure classes without re-running the query: a missing clause here is a
    retrieval failure, a clause that is present but uncited is a generation one.
    """
    blocks = turn.get("context_blocks") or []
    if not blocks:
        return

    cited = {c.get("chunk_id") for c in (turn.get("citations") or [])}

    with st.expander(f"Retrieval detail ({len(blocks)} chunks)", expanded=False):
        flags = []
        if turn.get("broadened"):
            flags.append("retried broadened - the first attempt was gated")
        if turn.get("below_threshold"):
            flags.append("below the confidence threshold")
        if flags:
            st.caption(" · ".join(flags))

        if variants := turn.get("query_variants"):
            st.caption("Queries searched: " + " | ".join(variants))

        st.markdown(
            "| # | used | score | section |\n|---|---|---|---|\n"
            + "\n".join(
                f"| {i} | {'yes' if b.get('chunk_id') in cited else 'no'} "
                f"| {b.get('score', 0):.3f} "
                f"| {_escape(str(b.get('section_path') or 'unlabelled'))}"
                f"{' (companion)' if b.get('is_companion') else ''} |"
                for i, b in enumerate(blocks, start=1)
            )
        )

        if timings := turn.get("timings_ms"):
            st.caption(" · ".join(f"{k} {v}ms" for k, v in timings.items()))
