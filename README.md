# Insurance Claims RAG Chatbot

A retrieval-augmented chatbot with two answer lanes:

- **Lane A — Policy & coverage Q&A** over uploaded policy wordings, endorsements and SOPs.
- **Lane B — Claim status lookup** against live claim records, behind a hard authorization boundary.

Full design rationale — model choices, chunking strategy, retrieval pipeline, evaluation
harness and licence audit — is in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Stack

| Concern | Choice | Cost |
|---|---|---|
| API | FastAPI | free |
| Orchestration | LangGraph | free |
| Frontend | Streamlit | free |
| Embeddings | `BAAI/bge-m3` — 1024d dense + learned sparse, self-hosted | free |
| Reranker | `BAAI/bge-reranker-v2-m3`, self-hosted | free |
| Vector store | Qdrant (local Docker) | free |
| Database | PostgreSQL | free |
| Cache / broker | Valkey | free |
| Parsing | Docling + pypdfium2 | free |
| Generation | `gpt-4o-mini` | **only paid component** |

A fully zero-cost run is available by setting `LLM_PROVIDER=ollama` — see ARCHITECTURE.md §4.1.

## Quickstart

```bash
# 1. dependencies (install the CPU-only torch wheel first to avoid a ~2.5 GB CUDA download)
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ui,eval,dev]"

# 2. config
cp .env.example .env        # then set OPENAI_API_KEY and JWT_SECRET

# 3. data plane
docker compose up -d qdrant postgres valkey

# 4. one-time setup
python scripts/download_models.py    # ~2.5 GB: bge-m3 + reranker
python scripts/init_qdrant.py        # creates the collection, vectors and payload indexes
alembic upgrade head

# 5. run
python scripts/run_api.py         # API → http://localhost:8010/docs
streamlit run ui/app.py           # UI  → http://localhost:8501
```

Everything in Docker instead: `docker compose --profile full up -d`

## Ingesting documents

Open **Documents** in the Streamlit UI and upload a PDF or DOCX. The page shows
live progress through parse → chunk → embed → index.

Notes on speed (CPU-only, measured):
- A plain-prose PDF takes the fast parser: **~4 s** for a short document.
- A PDF containing tables escalates to Docling for correct grids: **~50 s**.
- Embedding runs at roughly **300 tokens/sec**, so a 40-page wording is ~80 s.

Re-uploading identical bytes is detected by content hash and skipped instantly.

## Evaluation

The eval harness is not optional tooling — it decides the chunking and retrieval
parameters.

```bash
# Deterministic run. Query expansion calls an LLM to paraphrase each question, so
# leaving it on makes the number irreproducible - the paraphrase moves the candidate
# pool underneath the reranker. See eval/RESULTS.md.
QUERY_EXPANSION_ENABLED=false python -m eval.harness -k 3 --label baseline --out eval/runs/baseline.json

python -m eval.harness --only q13 q14      # re-run named questions
python -m eval.harness --questions <path>  # a different question set
```

Each run prints hit-rate@3, hit-rate@1, recall@3, MRR and abstention accuracy, and
labels every question `RETRIEVAL_FAIL`, `RETRIEVED_OK`, `ABSTAIN_OK` or
`OVER_ANSWERED` — the retrieval-versus-generation split. Full per-question output,
including every reranked candidate and its score, goes to the `--out` JSON.

**Latest measured result: [eval/RESULTS.md](eval/RESULTS.md)** — hit-rate@3 0.9524 to
1.0000 from one change to the cross-encoder input, with the failures it did *not* fix.

`eval/ablations.py` and `eval/report.py` are still stubs; there is no parameter sweep
or CI regression gate yet.

## Layout

```
app/         backend — api, graph, ingestion, retrieval, claims, security
ui/          Streamlit frontend
eval/        golden dataset, metrics, harness, measured results
scripts/     one-shot operational scripts
tests/       unit · integration · security (authorization regression suite)
docs/        ARCHITECTURE.md — design rationale and licence audit
docker/      container definitions
pdf/         generated sample corpus (scripts/generate_sample_docs.py)
data/        raw uploads and model weights (gitignored)
```

## Hardware

`bge-m3` is a 568M-parameter model. On CPU, expect roughly 15–25 chunks/second —
fine for a few thousand documents, painful at 50k. See ARCHITECTURE.md §12 for
sizing and the GPU threshold.
