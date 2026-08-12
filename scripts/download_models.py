"""Pre-downloads bge-m3 and the reranker into data/models.

Run once before first ingest. Roughly 2.3 GB for the embedder and 0.6-2.3 GB for
the reranker depending on which one is configured. Doing it explicitly beats
discovering the download inside a Celery task that then times out.

    python scripts/download_models.py
    python scripts/download_models.py --embedder-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_ROOT, settings  # noqa: E402
from app.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)


def _set_cache_dir() -> Path:
    """Keep weights inside the repo's data dir rather than the user profile."""
    cache = Path(os.environ.get("HF_HOME") or PROJECT_ROOT / "data" / "models")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    return cache


def main() -> int:
    parser = argparse.ArgumentParser(description="Download local models")
    parser.add_argument("--embedder-only", action="store_true")
    parser.add_argument("--reranker-only", action="store_true")
    args = parser.parse_args()

    configure_logging(json_output=False)
    cache = _set_cache_dir()
    log.info("Model cache", path=str(cache))

    from huggingface_hub import snapshot_download

    targets: list[tuple[str, str]] = []
    if not args.reranker_only:
        targets.append(("embedder", settings.embedding.model))
    if not args.embedder_only and settings.reranker.enabled:
        targets.append(("reranker", settings.reranker.model))

    for role, repo_id in targets:
        started = time.perf_counter()
        log.info("Downloading", role=role, repo=repo_id)
        try:
            # Deliberately no ignore_patterns. FlagEmbedding calls
            # snapshot_download itself with no filters, so excluding the ONNX
            # weights here does not save the disk - it just moves a multi-minute
            # download into the first Celery task, which is the worst place for it.
            # A pre-download has to be complete to be worth anything.
            path = snapshot_download(repo_id=repo_id, cache_dir=str(cache))
        except Exception as exc:
            log.error("Download failed", role=role, repo=repo_id, error=str(exc))
            return 1
        log.info(
            "Downloaded",
            role=role,
            repo=repo_id,
            seconds=round(time.perf_counter() - started, 1),
            path=path,
        )

    print("\nModels ready. Next: python scripts/init_qdrant.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
