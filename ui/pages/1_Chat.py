"""Main conversational interface.

Conversation identity lives in `session_state`; the transcript itself is held by
the graph checkpointer and mirrored to Postgres. Streamlit reruns the whole
script on every interaction, so anything kept only in local variables is lost -
the messages list is therefore rebuilt from session state each run
(ARCHITECTURE 9.1).
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.api_client import ApiClient, ApiError  # noqa: E402
from ui.components.citations import render_citations, render_inspection  # noqa: E402

st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


client = get_client()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "turns" not in st.session_state:
    st.session_state.turns = []

SUGGESTIONS = [
    "Is dental treatment covered?",
    "What is the room rent sub-limit?",
    "How long is the waiting period for pre-existing diseases?",
    "How do I file a cashless claim?",
]


# ───────────────────────────────────────────────────────────── sidebar

with st.sidebar:
    st.markdown("### Conversation")
    if st.button("New conversation", width="stretch"):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.turns = []
        st.rerun()

    st.caption(f"ID `{st.session_state.conversation_id[:8]}`")

    try:
        stats = client.stats()
        st.metric("Searchable chunks", f"{stats['embedded_chunks']:,}")
        if stats["embedded_chunks"] == 0:
            st.warning("No documents indexed. Upload one on the Documents page.")
    except ApiError:
        st.error("API unreachable")

    st.divider()
    st.caption(
        "Answers come only from your uploaded documents. If the answer is not in "
        "them, the assistant says so rather than guessing."
    )


# ────────────────────────────────────────────────────────────── header

st.title("Policy Assistant")
st.caption("Ask about coverage, exclusions, waiting periods and claim procedures.")

if not st.session_state.turns:
    st.markdown("**Try one of these**")
    cols = st.columns(len(SUGGESTIONS))
    for col, suggestion in zip(cols, SUGGESTIONS, strict=True):
        if col.button(suggestion, width="stretch"):
            st.session_state.pending = suggestion
            st.rerun()


# ────────────────────────────────────────────────────────── transcript

for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.markdown(turn["question"])

    with st.chat_message("assistant"):
        if turn.get("abstained"):
            st.warning(turn["answer"], icon="🔍")
        else:
            st.markdown(turn["answer"])

        render_citations(turn.get("citations") or [])
        render_inspection(turn)

        meta = []
        if turn.get("intent"):
            meta.append(turn["intent"].replace("_", " "))
        if turn.get("latency_ms"):
            meta.append(f"{turn['latency_ms'] / 1000:.1f}s")
        if turn.get("top_score"):
            meta.append(f"match {turn['top_score']:.2f}")
        if not turn.get("verified", True):
            meta.append("failed verification")
        if meta:
            st.caption(" · ".join(meta))


# ────────────────────────────────────────────────────────────── input

question = st.chat_input("Ask about your policy...")
if not question and "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.empty()
        answer = None
        try:
            # Stage events rather than tokens: generation is constrained decoding,
            # so there is no partial text to stream. Naming the current step keeps
            # a multi-second CPU turn legible.
            for kind, payload in client.chat_stream(question, st.session_state.conversation_id):
                if kind == "stage":
                    status.caption(f"⋯ {payload}")
                else:
                    answer = payload
        except ApiError as exc:
            status.empty()
            st.error(str(exc))
            st.stop()

        status.empty()

        if answer is None:
            st.error("No answer was returned.")
            st.stop()

        st.session_state.conversation_id = answer.conversation_id

        if answer.abstained:
            st.warning(answer.answer, icon="🔍")
        else:
            st.markdown(answer.answer)

        turn = {
            "question": question,
            "answer": answer.answer,
            "citations": answer.citations,
            "intent": answer.intent,
            "abstained": answer.abstained,
            "verified": answer.verified,
            "top_score": answer.top_score,
            "latency_ms": answer.latency_ms,
            "below_threshold": answer.below_threshold,
            "broadened": answer.broadened,
            "query_variants": answer.query_variants,
            "context_blocks": answer.context_blocks,
            "timings_ms": answer.timings_ms,
        }

        render_citations(answer.citations)
        render_inspection(turn)

        meta = []
        if answer.intent:
            meta.append(answer.intent.replace("_", " "))
        meta.append(f"{answer.latency_ms / 1000:.1f}s")
        if answer.top_score:
            meta.append(f"match {answer.top_score:.2f}")
        st.caption(" · ".join(meta))

        st.session_state.turns.append(turn)
