"""Background indexer — page queue for re-indexing."""

from __future__ import annotations

from .queue import IndexQueue
from .worker import worker

__all__ = ["IndexQueue", "worker"]
