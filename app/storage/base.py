"""ObjectStore protocol - local filesystem now, S3 later.

Deliberately not MinIO: it is AGPL-3.0 and the network clause reaches a hosted
product (ARCHITECTURE 15.2). The local filesystem costs nothing, needs no
service, and this protocol makes S3/R2 a one-class swap if you ever want it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ObjectStore(Protocol):
    """Implementations: ``LocalFileStore``."""

    def put(self, key: str, data: bytes) -> str:
        """Store bytes, return the storage key."""
        ...

    def get(self, key: str) -> bytes: ...

    def path_for(self, key: str) -> Path:
        """Local path for a parser that needs a real file on disk."""
        ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None: ...
