"""Local filesystem object store. POC default, zero cost.

Raw uploads are kept forever, not deleted after parsing. Two reasons, both
operational: you *will* change the chunking strategy after the ablation grid, and
re-chunking from stored source is a background job while asking users to
re-upload is a project; and a disputed answer months later needs the original
document, not a reconstruction of it.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from app.config import settings
from app.logging import get_logger
from app.storage.base import ObjectStore

log = get_logger(__name__)


class LocalFileStore(ObjectStore):
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or settings.storage.object_store_path)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        # Reject traversal before it touches the filesystem: `key` can originate
        # from an uploaded filename.
        target = (self.root / key).resolve()
        if not str(target).startswith(str(self.root.resolve())):
            raise ValueError(f"Refusing key outside the store root: {key}")
        return target

    def put(self, key: str, data: bytes) -> str:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        log.debug("Stored object", key=key, bytes=len(data))
        return key

    def put_file(self, key: str, source: Path) -> str:
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return key

    def get(self, key: str) -> bytes:
        return self._resolve(key).read_bytes()

    def path_for(self, key: str) -> Path:
        return self._resolve(key)

    def exists(self, key: str) -> bool:
        return self._resolve(key).exists()

    def delete(self, key: str) -> None:
        target = self._resolve(key)
        if target.exists():
            target.unlink()


def content_hash(data: bytes) -> str:
    """Dedup key. Identical bytes are never ingested twice - which matters more
    now that embedding is self-hosted and costs minutes, not cents."""
    return hashlib.sha256(data).hexdigest()


def storage_key(hash_hex: str, filename: str) -> str:
    """Content-addressed, sharded two levels so no directory grows unbounded."""
    suffix = Path(filename).suffix.lower() or ".bin"
    return f"{hash_hex[:2]}/{hash_hex[2:4]}/{hash_hex}{suffix}"


_store: LocalFileStore | None = None


def get_object_store() -> LocalFileStore:
    global _store
    if _store is None:
        _store = LocalFileStore()
    return _store
