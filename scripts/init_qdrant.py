"""Creates the versioned collection, named vectors and payload indexes.

Idempotent: safe to run on every deploy. Use --recreate only to wipe and rebuild,
which is a full reindex of the corpus.

    python scripts/init_qdrant.py
    python scripts/init_qdrant.py --recreate --yes
    python scripts/init_qdrant.py --info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the script runnable as `python scripts/init_qdrant.py` from the repo root,
# not only as `python -m scripts.init_qdrant`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402
from app.logging import configure_logging, get_logger  # noqa: E402
from app.retrieval.vectorstore import PAYLOAD_INDEXES, VectorStore  # noqa: E402

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialise the Qdrant collection")
    parser.add_argument("--recreate", action="store_true", help="drop and rebuild (destructive)")
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    parser.add_argument("--info", action="store_true", help="print collection info and exit")
    args = parser.parse_args()

    configure_logging(json_output=False)
    store = VectorStore()

    if not store.health():
        log.error(
            "Cannot reach Qdrant", url=settings.qdrant.url, hint="docker compose up -d qdrant"
        )
        return 1

    if args.info:
        print(json.dumps(store.info(), indent=2))
        return 0

    if args.recreate and store.exists():
        points = store.info().get("points_count", 0)
        if not args.yes:
            print(f"\n  This DROPS collection '{store.collection}' and its {points} points.")
            print("  Every document would need re-embedding - hours on CPU.\n")
            if input("  Type the collection name to confirm: ").strip() != store.collection:
                print("Aborted.")
                return 1

    created = store.ensure_collection(recreate=args.recreate)

    log.info(
        "Collection ready",
        collection=store.collection,
        created=created,
        dim=settings.embedding.dim,
        embedding_model=settings.embedding.model,
        quantization=settings.qdrant.quantization,
        sparse_bm25=settings.retrieval.retrieval_bm25_enabled,
        payload_indexes=len(PAYLOAD_INDEXES),
    )
    print(json.dumps(store.info(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
