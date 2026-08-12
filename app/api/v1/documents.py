"""Upload, list, reindex and delete source documents.

Upload returns ``202 Accepted`` with a job id and never blocks. Ingestion is
minutes of CPU on this hardware - holding an HTTP connection open for it would
time out behind any proxy and give the user nothing to look at meanwhile.

The content-hash check happens *before* the job is queued, so re-uploading a file
someone already indexed is instant rather than a repeat of the slowest step in
the system.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, func, select

from app.core.enums import DocType, IngestionStatus
from app.db.models import Chunk, Document, DocumentVersion
from app.deps import DbSession
from app.logging import get_logger
from app.storage import content_hash, get_object_store

log = get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md", ".html"}


class UploadAccepted(BaseModel):
    document_id: uuid.UUID
    version_id: uuid.UUID
    filename: str
    status: IngestionStatus
    duplicate: bool = Field(default=False, description="Identical bytes were already indexed")
    message: str


class DocumentStatus(BaseModel):
    document_id: uuid.UUID
    filename: str
    status: IngestionStatus
    doc_type: DocType
    insurer: str | None = None
    product_name: str | None = None
    uin: str | None = None
    effective_from: str | None = None
    effective_to: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    parser: str | None = None
    error: str | None = None
    uploaded_at: datetime | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in (IngestionStatus.COMPLETED, IngestionStatus.FAILED)


class CorpusStats(BaseModel):
    documents: int
    indexed_documents: int
    chunks: int
    embedded_chunks: int


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=UploadAccepted)
async def upload_document(
    session: DbSession,
    background: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: DocType = Form(default=DocType.OTHER),
    tenant_id: str = Form(default="default"),
) -> UploadAccepted:
    filename = Path(file.filename or "upload.bin").name
    suffix = Path(filename).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"{suffix or 'This file type'} is not supported. "
            f"Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File is {len(data) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES / 1e6:.0f} MB.",
        )

    digest = content_hash(data)

    # Answer the duplicate case before queueing anything. Embedding is the
    # slowest thing this system does, and repeating it for identical bytes
    # produces a guaranteed-identical index.
    #
    # Only a COMPLETED version counts. A row left PENDING or FAILED by an earlier
    # attempt must not block re-ingestion - that would make a single failed
    # upload permanently un-retryable, since content_hash is unique.
    existing = await session.scalar(
        select(DocumentVersion).where(DocumentVersion.content_hash == digest)
    )
    if existing is not None and existing.status is IngestionStatus.COMPLETED:
        return UploadAccepted(
            document_id=existing.document_id,
            version_id=existing.id,
            filename=filename,
            status=existing.status,
            duplicate=True,
            message="This document is already indexed - nothing to do.",
        )

    # Stage to a temp file: the pipeline's parsers need a real path on disk, and
    # the object store is written by the worker once the hash is confirmed.
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        staged = Path(tmp.name)

    if existing is not None:
        # A previous attempt left a row behind. Reuse it rather than inserting a
        # second one - content_hash is unique, so a fresh insert would fail.
        document = await session.get(Document, existing.document_id)
        version = existing
        version.status = IngestionStatus.PENDING
        version.error_message = None
        log.info("Retrying a previously incomplete ingestion", filename=filename)
    else:
        document = Document(
            tenant_id=tenant_id,
            filename=filename,
            mime_type=file.content_type or "application/octet-stream",
            size_bytes=len(data),
            doc_type=doc_type,
        )
        session.add(document)
        await session.flush()

        version = DocumentVersion(
            document_id=document.id,
            version_no=1,
            content_hash=digest,
            storage_key="",  # set by the pipeline once the file is stored
            status=IngestionStatus.PENDING,
        )
        session.add(version)

    await session.commit()

    background.add_task(_ingest, staged, filename, document.id, version.id, tenant_id, doc_type)

    log.info("Upload accepted", filename=filename, bytes=len(data), document_id=str(document.id))
    return UploadAccepted(
        document_id=document.id,
        version_id=version.id,
        filename=filename,
        status=IngestionStatus.PENDING,
        message="Queued. Ingestion takes roughly a minute per 40 pages on CPU.",
    )


def _ingest(
    staged: Path,
    filename: str,
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    tenant_id: str,
    doc_type: DocType,
) -> None:
    """Run the pipeline in the background.

    FastAPI's BackgroundTasks keeps the POC to one process. Swap this for the
    Celery task when you want ingestion to survive an API restart - the pipeline
    call is identical either way.
    """
    from app.db.session import session_scope
    from app.ingestion.pipeline import IngestionPipeline

    try:
        with session_scope() as session:
            # Adopt the rows the endpoint already created. The client is polling
            # `document_id`, so the pipeline must fill these in rather than mint
            # its own - otherwise the upload succeeds while the progress bar
            # waits on an id that no longer exists.
            IngestionPipeline().ingest_file(
                session,
                path=staged,
                filename=filename,
                tenant_id=tenant_id,
                doc_type=doc_type,
                document_id=document_id,
                version_id=version_id,
            )
    except Exception as exc:
        log.error("Background ingestion failed", filename=filename, error=str(exc))
        # Surface the failure on the row the client is polling, or the UI spins
        # until its timeout with no explanation.
        try:
            with session_scope() as session:
                version = session.get(DocumentVersion, version_id)
                if version is not None:
                    version.status = IngestionStatus.FAILED
                    version.error_message = str(exc)[:2000]
        except Exception:
            log.error("Could not record ingestion failure", version_id=str(version_id))
    finally:
        staged.unlink(missing_ok=True)


@router.get("", response_model=list[DocumentStatus])
async def list_documents(
    session: DbSession, tenant_id: str = "default", limit: int = 100
) -> list[DocumentStatus]:
    rows = await session.execute(
        select(Document, DocumentVersion)
        .join(DocumentVersion, DocumentVersion.document_id == Document.id, isouter=True)
        .where(Document.tenant_id == tenant_id)
        .order_by(desc(Document.created_at))
        .limit(min(limit, 500))
    )
    return [_to_status(doc, ver) for doc, ver in rows.all()]


@router.get("/stats", response_model=CorpusStats)
async def corpus_stats(session: DbSession, tenant_id: str = "default") -> CorpusStats:
    documents = await session.scalar(
        select(func.count(Document.id)).where(Document.tenant_id == tenant_id)
    )
    indexed = await session.scalar(
        select(func.count(DocumentVersion.id)).where(
            DocumentVersion.status == IngestionStatus.COMPLETED
        )
    )
    chunks = await session.scalar(select(func.count(Chunk.id)).where(Chunk.tenant_id == tenant_id))
    embedded = await session.scalar(
        select(func.count(Chunk.id)).where(
            Chunk.tenant_id == tenant_id, Chunk.indexed_at.is_not(None)
        )
    )
    return CorpusStats(
        documents=documents or 0,
        indexed_documents=indexed or 0,
        chunks=chunks or 0,
        embedded_chunks=embedded or 0,
    )


@router.get("/{document_id}", response_model=DocumentStatus)
async def get_document(document_id: uuid.UUID, session: DbSession) -> DocumentStatus:
    row = await session.execute(
        select(Document, DocumentVersion)
        .join(DocumentVersion, DocumentVersion.document_id == Document.id, isouter=True)
        .where(Document.id == document_id)
    )
    found = row.first()
    if found is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.")
    return _to_status(*found)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, session: DbSession) -> None:
    """Remove a document from the index and the database.

    The raw file stays in the object store - it is the only copy of what was
    actually uploaded, and a disputed answer months later needs the original.
    """
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such document.")

    from app.retrieval.vectorstore import get_vector_store

    get_vector_store().delete_by_document(str(document_id))
    await session.delete(document)
    await session.commit()
    log.info("Deleted document", document_id=str(document_id))


def _to_status(document: Document, version: DocumentVersion | None) -> DocumentStatus:
    return DocumentStatus(
        document_id=document.id,
        filename=document.filename,
        status=version.status if version else IngestionStatus.PENDING,
        doc_type=document.doc_type,
        insurer=document.insurer,
        product_name=document.product_name,
        uin=document.uin,
        effective_from=document.effective_from.isoformat() if document.effective_from else None,
        effective_to=document.effective_to.isoformat() if document.effective_to else None,
        page_count=version.page_count if version else None,
        chunk_count=version.chunk_count if version else None,
        parser=version.parser if version else None,
        error=version.error_message if version else None,
        uploaded_at=document.created_at,
    )


__all__ = ["router", "get_object_store"]
