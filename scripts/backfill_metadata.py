"""Backfills document catalogue metadata (insurer, product name, UIN).

Needed when a corpus was ingested with ``--no-llm-metadata``: the regex pass reads
the effective dates, but insurer and product name come only from the LLM pass, so
they stay NULL. A NULL product name means the retrieval product filter can never
resolve, and every product-specific question falls back to a broadened second
search (see ``app/retrieval/catalogue.py``).

Reconstructs each document's head from its stored parent chunks rather than
re-parsing the PDF - the text is already in Postgres, and re-running Docling over
the corpus costs minutes for text we have.

Writes to Postgres and to the matching Qdrant payloads, so the filter and the
catalogue agree.

    python scripts/backfill_metadata.py            # only fills NULLs
    python scripts/backfill_metadata.py --overwrite  # re-extract everything
    python scripts/backfill_metadata.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import models  # noqa: E402
from sqlalchemy import select  # noqa: E402

from app.core.enums import DocType  # noqa: E402
from app.db.models import Chunk, Document  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.ingestion.metadata import HEAD_CHARS, _extract_with_llm, _LLMMetadata  # noqa: E402
from app.logging import configure_logging  # noqa: E402
from app.retrieval.vectorstore import Payload, get_vector_store  # noqa: E402


def _head(session, document_id) -> str:
    """Rebuild the document's opening text from its parent chunks."""
    texts = session.scalars(
        select(Chunk.text)
        .where(Chunk.document_id == document_id, Chunk.kind == "parent")
        .order_by(Chunk.chunk_index)
        .limit(12)
    ).all()
    return "\n\n".join(texts)[:HEAD_CHARS]


async def _extract(head: str, filename: str) -> _LLMMetadata:
    return await _extract_with_llm(head, filename)


def _clean(value: str | None) -> str | None:
    if not value:
        return None
    return " ".join(value.split()).strip(" .,-")[:255] or None


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill document catalogue metadata")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--overwrite", action="store_true", help="re-extract even if already set")
    parser.add_argument("--dry-run", action="store_true", help="show what would change")
    args = parser.parse_args()

    configure_logging(json_output=False)
    store = get_vector_store()
    updated = skipped = failed = 0

    with session_scope() as session:
        documents = session.scalars(
            select(Document).where(Document.tenant_id == args.tenant).order_by(Document.filename)
        ).all()

        if not documents:
            print(f"No documents for tenant {args.tenant!r}.")
            return 1

        for doc in documents:
            if doc.product_name and doc.insurer and not args.overwrite:
                print(f"  skip     {doc.filename}  (already has product + insurer)")
                skipped += 1
                continue

            head = _head(session, doc.id)
            if not head.strip():
                print(f"  FAILED   {doc.filename}  (no parent chunk text to read)")
                failed += 1
                continue

            try:
                meta = asyncio.run(_extract(head, doc.filename))
            except Exception as exc:
                print(f"  FAILED   {doc.filename}  ({exc})")
                failed += 1
                continue

            product = _clean(meta.product_name)
            insurer = _clean(meta.insurer)
            uin = _clean(meta.uin)

            # Same rule the ingest path applies: process documents stay
            # product-neutral so they remain retrievable for every product.
            if doc.doc_type in (DocType.SOP, DocType.CIRCULAR):
                product = None

            print(f"  {doc.filename}")
            print(f"      product : {doc.product_name!r} -> {product!r}")
            print(f"      insurer : {doc.insurer!r} -> {insurer!r}")
            print(f"      uin     : {doc.uin!r} -> {uin!r}")

            if args.dry_run:
                continue

            # Only ever fill a gap unless explicitly overwriting: a value read at
            # ingest time saw the real parsed document, this one sees a
            # reconstruction, so the original wins a disagreement.
            if product and (args.overwrite or not doc.product_name):
                doc.product_name = product
            if insurer and (args.overwrite or not doc.insurer):
                doc.insurer = insurer
            if uin and (args.overwrite or not doc.uin):
                doc.uin = uin

            # Keep the vector payloads in step, or the filter would match on a
            # product name the points do not carry.
            try:
                store.client.set_payload(
                    collection_name=store.collection,
                    payload={
                        Payload.PRODUCT_NAME: doc.product_name,
                        Payload.INSURER: doc.insurer,
                        Payload.UIN: doc.uin,
                    },
                    points=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=Payload.DOC_ID, match=models.MatchValue(value=str(doc.id))
                            )
                        ]
                    ),
                    wait=True,
                )
            except Exception as exc:
                print(f"      ! vector payload update failed: {exc}")
                failed += 1
                continue

            updated += 1

    print("\n" + "-" * 68)
    print(f"{updated} updated · {skipped} already complete · {failed} failed")
    if updated and not args.dry_run:
        print("Restart the API so the product catalogue cache is rebuilt.")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
