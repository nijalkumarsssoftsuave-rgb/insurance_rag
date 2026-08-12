"""Corpus management: upload, ingestion status, reindex.

Upload is asynchronous by necessity - extraction and embedding are CPU-bound and
take real time - so the page polls the job rather than pretending the work is
instant. The stage labels mirror the pipeline so a user watching a slow document
can see *which* step is slow.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ui.api_client import ApiClient, ApiError  # noqa: E402

st.set_page_config(page_title="Documents", page_icon="📄", layout="wide")


@st.cache_resource
def get_client() -> ApiClient:
    return ApiClient()


client = get_client()

st.title("Documents")
st.caption("Upload policy wordings, endorsements and claim procedures.")

DOC_TYPES = {
    "Policy wording": "policy_wording",
    "Endorsement": "endorsement",
    "Claims SOP": "sop",
    "Regulatory circular": "circular",
    "Claim form": "claim_form",
    "Brochure / prospectus": "brochure",
    "Other": "other",
}

# Mirrors IngestionStatus. Embedding is called out as the slow step because on
# CPU it is roughly 70% of the wall clock, and users otherwise assume a hang.
STAGES = {
    "pending": (0.05, "Queued"),
    "parsing": (0.25, "Extracting text and tables"),
    "chunking": (0.45, "Splitting into clauses"),
    "embedding": (0.75, "Generating embeddings (the slow step)"),
    "indexing": (0.92, "Writing to the vector index"),
    "completed": (1.0, "Ready"),
    "failed": (1.0, "Failed"),
}

# ─────────────────────────────────────────────────────────────── upload

with st.form("upload", clear_on_submit=True):
    st.markdown("#### Upload a document")
    uploaded = st.file_uploader(
        "PDF or Word document",
        type=["pdf", "docx", "doc", "txt", "md"],
        help="Text-based PDFs only. Scanned documents need OCR, which is disabled.",
    )
    label = st.selectbox("Document type", list(DOC_TYPES), index=0)
    submitted = st.form_submit_button("Upload and index", type="primary")

if submitted:
    if uploaded is None:
        st.warning("Choose a file first.")
    else:
        data = uploaded.getvalue()
        with st.spinner(f"Uploading {uploaded.name} ({len(data) / 1e6:.1f} MB)..."):
            try:
                result = client.upload(uploaded.name, data, doc_type=DOC_TYPES[label])
            except ApiError as exc:
                st.error(str(exc))
                st.stop()

        if result.duplicate:
            st.info(f"**{result.filename}** is already indexed - nothing to do.", icon="✅")
        else:
            st.success(f"**{result.filename}** queued.", icon="⏳")
            progress = st.progress(0.0, text="Queued")
            outcome = st.empty()

            for _ in range(180):  # poll for up to ~6 minutes
                time.sleep(2)
                try:
                    doc = client.document(result.document_id)
                except ApiError:
                    continue  # the pipeline swaps the placeholder row mid-run

                status = doc.get("status", "pending")
                pct, text = STAGES.get(status, (0.1, status))
                progress.progress(pct, text=text)

                if status == "completed":
                    outcome.success(
                        f"Indexed **{doc.get('page_count') or '?'} pages** into "
                        f"**{doc.get('chunk_count') or '?'} chunks** "
                        f"using `{doc.get('parser')}`.",
                        icon="✅",
                    )
                    break
                if status == "failed":
                    outcome.error(doc.get("error") or "Ingestion failed.", icon="❌")
                    break

st.divider()

# ─────────────────────────────────────────────────────────────── corpus

try:
    stats = client.stats()
    documents = client.list_documents()
except ApiError as exc:
    st.error(str(exc))
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Documents", stats["documents"])
c2.metric("Indexed", stats["indexed_documents"])
c3.metric("Searchable chunks", f"{stats['embedded_chunks']:,}")

st.markdown("#### Indexed documents")

if not documents:
    st.info("No documents yet. Upload one above to make the assistant useful.")
else:
    frame = pd.DataFrame(
        [
            {
                "File": d["filename"],
                "Status": d["status"],
                "Type": d["doc_type"].replace("_", " "),
                "Insurer": d.get("insurer") or "—",
                "Product": d.get("product_name") or "—",
                "Effective": " → ".join(
                    x for x in (d.get("effective_from"), d.get("effective_to")) if x
                )
                or "—",
                "Pages": d.get("page_count") or "—",
                "Chunks": d.get("chunk_count") or "—",
                "Parser": d.get("parser") or "—",
            }
            for d in documents
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")

    failed = [d for d in documents if d["status"] == "failed"]
    if failed:
        st.markdown("#### Failed")
        for d in failed:
            with st.expander(f"❌ {d['filename']}"):
                st.code(d.get("error") or "No error recorded.")

    with st.expander("Remove a document from the index"):
        choice = st.selectbox(
            "Document",
            options=[d["document_id"] for d in documents],
            format_func=lambda i: next(d["filename"] for d in documents if d["document_id"] == i),
        )
        st.caption(
            "Deletes the chunks and the vectors. The original upload is kept - "
            "a disputed answer months later needs the source document."
        )
        if st.button("Delete", type="secondary"):
            try:
                client.delete_document(choice)
                st.success("Deleted.")
                st.rerun()
            except ApiError as exc:
                st.error(str(exc))
