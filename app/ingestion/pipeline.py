"""Orchestrates parse, chunk, enrich, embed, index.

Synchronous by design. It runs inside a Celery worker where there is nothing to
overlap with - the embedder saturates the CPU on its own, so an async pipeline
would add machinery and buy nothing.

The stage order matters for failure handling: the document row and its version
are written *before* the slow work starts, so a crash mid-embed leaves a row in
`embedding` state that can be retried, not an invisible half-ingested document.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.enums import ChunkKind, DocType, IngestionStatus
from app.db.models import Chunk, Document, DocumentVersion
from app.embeddings.bge_m3 import get_embedder
from app.ingestion.chunking import ChunkedDocument, StructureAwareChunker
from app.ingestion.metadata import DocumentMetadata, extract_metadata
from app.ingestion.parsers.base import ParsedDocument, ParserError
from app.logging import get_logger
from app.retrieval.vectorstore import (
    TS_MAX,
    TS_MIN,
    ChunkPoint,
    Payload,
    VectorStore,
    get_vector_store,
    to_ts,
)
from app.storage import content_hash, get_object_store, storage_key

log = get_logger(__name__)

# Below this ratio of headings to blocks, the document has no usable structure
# and the fast parser's heuristics are not worth trusting.
MIN_HEADING_RATIO = 0.02


@dataclass(slots=True)
class IngestionResult:
    document_id: uuid.UUID
    version_id: uuid.UUID
    status: IngestionStatus
    parser: str = ""
    page_count: int = 0
    parent_chunks: int = 0
    child_chunks: int = 0
    tokens_embedded: int = 0
    duplicate: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    timings_s: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is IngestionStatus.COMPLETED


class IngestionPipeline:
    def __init__(
        self,
        *,
        store: VectorStore | None = None,
        embedder=None,
        chunker=None,
        parser=None,
    ) -> None:
        self.store = store or get_vector_store()
        self._embedder = embedder
        self._chunker = chunker or StructureAwareChunker()
        self._parser = parser

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    # ────────────────────────────────────────────────────────── entrypoint

    def ingest_file(
        self,
        session: Session,
        *,
        path: Path,
        filename: str | None = None,
        tenant_id: str = "default",
        uploaded_by: uuid.UUID | None = None,
        doc_type: DocType | None = None,
        extract_llm_metadata: bool = True,
        document_id: uuid.UUID | None = None,
        version_id: uuid.UUID | None = None,
    ) -> IngestionResult:
        """Ingest one file end to end. Idempotent on content hash.

        ``document_id``/``version_id`` adopt rows the API already created so the
        client can poll them. Without this the pipeline would mint its own ids and
        the id handed to the browser would 404 forever - the upload succeeds while
        the progress bar never finishes.
        """
        filename = filename or path.name
        data = path.read_bytes()
        digest = content_hash(data)

        # Only a COMPLETED version counts as a duplicate, and never the row we
        # were handed to fill in. The API writes a placeholder carrying this same
        # hash before queueing us; without both conditions the pipeline finds its
        # own placeholder, calls the document a duplicate, and silently ingests
        # nothing.
        duplicate_query = select(DocumentVersion).where(
            DocumentVersion.content_hash == digest,
            DocumentVersion.status == IngestionStatus.COMPLETED,
        )
        if version_id is not None:
            duplicate_query = duplicate_query.where(DocumentVersion.id != version_id)

        existing = session.scalar(duplicate_query)
        if existing is not None:
            # Identical bytes already indexed. Re-embedding costs minutes of CPU
            # for a guaranteed-identical result.
            log.info("Skipping duplicate", filename=filename, hash=digest[:12])
            return IngestionResult(
                document_id=existing.document_id,
                version_id=existing.id,
                status=IngestionStatus.COMPLETED,
                duplicate=True,
            )

        key = storage_key(digest, filename)
        store = get_object_store()
        if not store.exists(key):
            store.put_file(key, path)

        document = session.get(Document, document_id) if document_id else None
        if document is None:
            document = Document(
                tenant_id=tenant_id,
                filename=filename,
                mime_type=_mime_for(filename),
                size_bytes=len(data),
                uploaded_by=uploaded_by,
                doc_type=doc_type or DocType.OTHER,
            )
            session.add(document)
            session.flush()

        version = session.get(DocumentVersion, version_id) if version_id else None
        if version is None:
            version = DocumentVersion(
                document_id=document.id,
                version_no=1,
                content_hash=digest,
                storage_key=key,
            )
            session.add(version)

        version.content_hash = digest
        version.storage_key = key
        version.status = IngestionStatus.PARSING
        version.started_at = datetime.now(UTC)
        version.chunk_strategy_version = settings.chunking.strategy_version
        version.embedding_model = settings.embedding.model
        version.embedding_dim = settings.embedding.dim
        session.commit()

        result = IngestionResult(
            document_id=document.id, version_id=version.id, status=IngestionStatus.PARSING
        )

        try:
            self._run(
                session,
                document,
                version,
                store.path_for(key),
                result,
                extract_llm_metadata=extract_llm_metadata,
            )
        except Exception as exc:
            log.error("Ingestion failed", filename=filename, error=str(exc))
            version.status = IngestionStatus.FAILED
            version.error_message = str(exc)[:2000]
            result.status = IngestionStatus.FAILED
            result.error = str(exc)
            session.commit()

        return result

    # ───────────────────────────────────────────────────────────── stages

    def _run(
        self,
        session: Session,
        document: Document,
        version: DocumentVersion,
        path: Path,
        result: IngestionResult,
        *,
        extract_llm_metadata: bool,
    ) -> None:
        clock = time.perf_counter

        # 1. Parse ------------------------------------------------------------
        t0 = clock()
        parsed = self._parse(path)
        result.timings_s["parse"] = round(clock() - t0, 2)
        result.parser = parsed.parser
        result.page_count = parsed.page_count
        result.warnings.extend(parsed.warnings)

        if not parsed.blocks:
            raise ParserError(
                "No extractable text. If this is a scanned PDF it needs OCR, "
                "which is disabled for this corpus."
            )

        version.parser = parsed.parser
        version.page_count = parsed.page_count

        # 2. Metadata ---------------------------------------------------------
        t0 = clock()
        meta = extract_metadata(parsed, filename=document.filename, use_llm=extract_llm_metadata)
        _apply_metadata(document, meta)
        result.timings_s["metadata"] = round(clock() - t0, 2)

        # 3. Chunk ------------------------------------------------------------
        version.status = IngestionStatus.CHUNKING
        session.commit()

        t0 = clock()
        chunked = self._chunker.chunk(parsed, breadcrumb=_document_breadcrumb(document))
        result.timings_s["chunk"] = round(clock() - t0, 2)
        result.parent_chunks = len(chunked.parents)
        result.child_chunks = len(chunked.children)

        if not chunked.children:
            raise ParserError("Document produced no indexable chunks.")

        # 4. Persist chunk rows before embedding -----------------------------
        # The mirror has to exist first: it is what resolves a citation back to
        # its source text months later, even after a reindex (ARCHITECTURE 7.3).
        rows = self._persist_chunks(session, document, version, chunked)
        session.commit()

        # 5. Embed ------------------------------------------------------------
        version.status = IngestionStatus.EMBEDDING
        session.commit()

        t0 = clock()
        children = [c for c in chunked.chunks if c.kind is not ChunkKind.PARENT]
        embedded = self.embedder.embed_documents([c.embedded_text for c in children])
        result.timings_s["embed"] = round(clock() - t0, 2)
        result.tokens_embedded = sum(c.token_count for c in children)

        # 6. Index ------------------------------------------------------------
        version.status = IngestionStatus.INDEXING
        session.commit()

        t0 = clock()
        points = [
            ChunkPoint(
                chunk_id=rows[c.index].id,
                dense=embedded.dense[i],
                lexical=embedded.sparse[i] if embedded.sparse else None,
                payload=_payload(document, version, c, rows),
            )
            for i, c in enumerate(children)
        ]
        self.store.upsert(points, wait=True)
        result.timings_s["index"] = round(clock() - t0, 2)

        now = datetime.now(UTC)
        for row in rows.values():
            row.indexed_at = now

        version.status = IngestionStatus.COMPLETED
        version.completed_at = now
        version.chunk_count = len(chunked.chunks)
        version.token_count = result.tokens_embedded
        result.status = IngestionStatus.COMPLETED
        session.commit()

        log.info(
            "Ingested",
            filename=document.filename,
            parser=parsed.parser,
            pages=parsed.page_count,
            parents=result.parent_chunks,
            children=result.child_chunks,
            tokens=result.tokens_embedded,
            timings=result.timings_s,
        )

    def _parse(self, path: Path) -> ParsedDocument:
        """Docling first, pypdfium2 as a fallback.

        Not the other way round. The earlier design tried the fast parser and
        escalated only on a detected table; see IngestionSettings for why that
        heuristic was abandoned. Docling is the one that reconstructs benefit
        grids, and a wrong sub-limit is a worse outcome than a slower ingest.

        pypdfium2 still earns its place: it is the fallback when Docling fails
        (a malformed PDF, a missing model) and the opt-in fast path for a corpus
        known to be prose-only.
        """
        if self._parser is not None:
            return self._parser.parse(path)

        from app.ingestion.parsers.pdfium_parser import PdfiumParser

        if settings.ingestion.prefer_fast_parser and path.suffix.lower() == ".pdf":
            parsed = PdfiumParser().parse(path)
            if parsed.likely_tables:
                log.warning(
                    "Fast parser flagged probable tables it cannot reconstruct",
                    file=path.name,
                )
            return parsed

        from app.ingestion.parsers.docling_parser import DoclingParser

        try:
            return DoclingParser(do_ocr=settings.ingestion.ocr_enabled).parse(path)
        except ParserError as exc:
            if path.suffix.lower() != ".pdf":
                raise
            log.warning("Docling failed, falling back to the fast parser", error=str(exc))
            parsed = PdfiumParser().parse(path)
            parsed.warnings.append(
                "Parsed without a layout model - any tables in this document may be unreliable."
            )
            return parsed

    def _persist_chunks(
        self,
        session: Session,
        document: Document,
        version: DocumentVersion,
        chunked: ChunkedDocument,
    ) -> dict[int, Chunk]:
        """Write chunk rows, resolving parent_index into a real foreign key."""
        rows: dict[int, Chunk] = {}

        # Parents first so children can reference them in the same flush.
        for draft in sorted(chunked.chunks, key=lambda c: c.parent_index is not None):
            row = Chunk(
                tenant_id=document.tenant_id,
                document_id=document.id,
                document_version_id=version.id,
                parent_id=(
                    rows[draft.parent_index].id
                    if draft.parent_index is not None and draft.parent_index in rows
                    else None
                ),
                kind=draft.kind,
                chunk_index=draft.index,
                section_path=draft.section_path[:1024] if draft.section_path else None,
                page_no=draft.page_no,
                token_count=draft.token_count,
                text=draft.text,
                embedded_text=draft.embedded_text,
                text_hash=draft.text_hash,
                embedding_model=settings.embedding.model,
                chunk_strategy_version=chunked.strategy_version,
            )
            session.add(row)
            session.flush()
            rows[draft.index] = row

        return rows


def _document_breadcrumb(document: Document) -> str:
    parts = [document.insurer, document.product_name]
    return " | ".join(p for p in parts if p)


def _apply_metadata(document: Document, meta: DocumentMetadata) -> None:
    document.insurer = meta.insurer or document.insurer
    document.product_name = meta.product_name or document.product_name
    document.uin = meta.uin or document.uin
    document.language = meta.language or document.language
    document.effective_from = meta.effective_from or document.effective_from
    document.effective_to = meta.effective_to or document.effective_to
    if meta.doc_type is not None and document.doc_type is DocType.OTHER:
        document.doc_type = meta.doc_type


def _payload(
    document: Document,
    version: DocumentVersion,
    draft,
    rows: dict[int, Chunk],
) -> dict:
    parent_row = rows.get(draft.parent_index) if draft.parent_index is not None else None
    return {
        Payload.TENANT_ID: document.tenant_id,
        Payload.DOC_ID: str(document.id),
        Payload.DOC_VERSION_ID: str(version.id),
        Payload.CHUNK_ID: str(rows[draft.index].id),
        Payload.PARENT_ID: str(parent_row.id) if parent_row else None,
        Payload.KIND: draft.kind.value,
        Payload.INSURER: document.insurer,
        Payload.PRODUCT_NAME: document.product_name,
        Payload.UIN: document.uin,
        Payload.DOC_TYPE: document.doc_type.value,
        Payload.LANGUAGE: document.language,
        Payload.IS_SUPERSEDED: False,
        # Sentinels, so a document with no stated effective window still matches
        # a date-scoped search instead of silently disappearing from it.
        Payload.EFFECTIVE_FROM_TS: to_ts(document.effective_from, default=TS_MIN),
        Payload.EFFECTIVE_TO_TS: to_ts(document.effective_to, default=TS_MAX),
        Payload.EFFECTIVE_FROM: document.effective_from.isoformat()
        if document.effective_from
        else None,
        Payload.EFFECTIVE_TO: document.effective_to.isoformat() if document.effective_to else None,
        Payload.SECTION_PATH: draft.section_path,
        Payload.PAGE_NO: draft.page_no,
        Payload.CHUNK_INDEX: draft.index,
        Payload.TOKEN_COUNT: draft.token_count,
        Payload.TEXT: draft.text,
        Payload.EMBEDDING_MODEL: settings.embedding.model,
        Payload.CHUNK_STRATEGY_VERSION: version.chunk_strategy_version,
    }


def _mime_for(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".html": "text/html",
    }.get(suffix, "application/octet-stream")
