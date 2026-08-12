"""Streamlit entrypoint: auth, session bootstrap, page registration.

A thin HTTP client and nothing more. It loads no models: bge-m3 and the reranker
live in the API and worker processes only. Loading them here would duplicate
~2.5 GB per browser session and reload on every script rerun (ARCHITECTURE 9.1).
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui.api_client import ApiClient  # noqa: E402

st.set_page_config(
    page_title="Insurance Policy Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_client() -> ApiClient:
    """Cached so the connection pool survives Streamlit's script reruns."""
    return ApiClient()


def render_sidebar(client: ApiClient) -> None:
    with st.sidebar:
        st.markdown("### System")
        try:
            health = client.health()
        except Exception as exc:
            st.error("API unreachable")
            st.caption(str(exc)[:200])
            st.code("uvicorn app.main:app --reload --port 8000", language="bash")
            return

        if health.get("status") == "ok":
            st.success("All services up")
        else:
            st.warning("Degraded")

        st.markdown(
            f"- Postgres: {'🟢' if health['database'] else '🔴'}\n"
            f"- Qdrant: {'🟢' if health['qdrant'] else '🔴'}\n"
            f"- Indexed vectors: **{health.get('points') or 0:,}**"
        )
        st.caption(f"Embedder `{health['embedding_model']}`  \nLLM `{health['llm_model']}`")


client = get_client()
render_sidebar(client)

st.title("Insurance Policy Assistant")
st.caption(
    "Upload your policy documents, then ask questions about coverage, exclusions "
    "and waiting periods. Every answer cites the clause it came from."
)

st.info(
    "**Start here →** open **Documents** in the sidebar to upload a policy PDF. "
    "Nothing can be answered until at least one document is indexed.",
    icon="📄",
)

left, right = st.columns(2)
with left:
    st.markdown(
        "#### What it can answer\n"
        "- Is dental treatment covered?\n"
        "- What is the room rent sub-limit?\n"
        "- How long is the pre-existing disease waiting period?\n"
        "- How do I file a cashless claim?"
    )
with right:
    st.markdown(
        "#### How it stays honest\n"
        "- Answers only from your uploaded documents\n"
        "- Cites the clause behind every statement\n"
        "- Checks exclusions before saying something is covered\n"
        "- Says it does not know rather than guessing"
    )
