"""Bulk-ingests a local folder of PDFs and DOCX files.

Runs the pipeline in-process rather than through the API, so it is the right tool
for seeding a corpus and for re-ingesting after a chunking change. Progress is
printed per document because on CPU this takes real time and a silent script that
runs for twenty minutes is indistinguishable from a hung one.

    python scripts/ingest_folder.py --path pdf
    python scripts/ingest_folder.py --path pdf --no-llm-metadata
    python scripts/ingest_folder.py --path pdf --reset
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.enums import DocType  # noqa: E402
from app.db.session import session_scope  # noqa: E402
from app.ingestion.pipeline import IngestionPipeline  # noqa: E402
from app.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

SUFFIXES = {".pdf", ".docx", ".doc", ".txt", ".md", ".html"}

# Filename hints, so a seeded corpus lands with sensible doc types without
# needing the LLM metadata pass.
NAME_HINTS: tuple[tuple[str, DocType], ...] = (
    ("endorsement", DocType.ENDORSEMENT),
    ("circular", DocType.CIRCULAR),
    ("sop", DocType.SOP),
    ("claim_form", DocType.CLAIM_FORM),
    ("brochure", DocType.BROCHURE),
    ("prospectus", DocType.BROCHURE),
    ("wording", DocType.POLICY_WORDING),
    ("policy", DocType.POLICY_WORDING),
)


def guess_doc_type(name: str) -> DocType:
    lowered = name.lower()
    for token, doc_type in NAME_HINTS:
        if token in lowered:
            return doc_type
    return DocType.OTHER


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest a folder of documents")
    parser.add_argument("--path", required=True, help="folder to ingest")
    parser.add_argument("--tenant", default="default")
    parser.add_argument(
        "--no-llm-metadata",
        action="store_true",
        help="skip the LLM metadata pass (no API key needed, no cost)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="DESTRUCTIVE: drop and recreate the vector collection first",
    )
    args = parser.parse_args()

    configure_logging(json_output=False)

    folder = Path(args.path)
    if not folder.is_dir():
        print(f"Not a directory: {folder}", file=sys.stderr)
        return 1

    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in SUFFIXES)
    if not files:
        print(f"No ingestible files in {folder}", file=sys.stderr)
        return 1

    pipeline = IngestionPipeline()

    if args.reset:
        confirm = input(
            f"This drops collection '{pipeline.store.collection}' and re-embeds "
            f"everything. Type 'reset' to confirm: "
        )
        if confirm.strip() != "reset":
            print("Aborted.")
            return 1
        pipeline.store.drop_collection()
        pipeline.store.ensure_collection()

    print(f"\nIngesting {len(files)} file(s) from {folder}/\n")
    started = time.perf_counter()
    ok = failed = duplicate = 0
    total_chunks = 0

    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        t0 = time.perf_counter()

        with session_scope() as session:
            result = pipeline.ingest_file(
                session,
                path=path,
                tenant_id=args.tenant,
                doc_type=guess_doc_type(path.name),
                extract_llm_metadata=not args.no_llm_metadata,
            )

        elapsed = time.perf_counter() - t0

        if result.duplicate:
            duplicate += 1
            print("        already indexed, skipped\n")
            continue
        if not result.ok:
            failed += 1
            print(f"        FAILED: {result.error}\n")
            continue

        ok += 1
        total_chunks += result.child_chunks
        print(
            f"        {result.parser} · {result.page_count}p · "
            f"{result.parent_chunks} parents + {result.child_chunks} children · "
            f"{result.tokens_embedded} tokens · {elapsed:.1f}s"
        )
        if result.warnings:
            for warning in result.warnings:
                print(f"        ! {warning}")
        print()

    total = time.perf_counter() - started
    print("-" * 68)
    print(
        f"{ok} indexed · {duplicate} duplicate · {failed} failed · "
        f"{total_chunks} searchable chunks · {total:.1f}s"
    )
    print(f"Vector store: {pipeline.store.info()}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
