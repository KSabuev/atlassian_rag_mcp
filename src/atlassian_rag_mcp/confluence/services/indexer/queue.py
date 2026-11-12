"""IndexQueue — persistent background indexing queue (async-safe)."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any

from atlassian_rag_mcp.confluence.models import QueueStatus
from atlassian_rag_mcp.log.logger import logger


class IndexQueue:
    """Persistent queue for page re-indexing with periodic flush to disk.

    Async-safe: uses asyncio.Lock since all callers run in async context.
    """

    def __init__(
        self,
        working_dir: str,
        *,
        persist_interval: float = 5.0,
        max_stored_errors: int = 100,
        max_retries: int = 3,
        max_sweeps: int = 3,
    ) -> None:
        self._queue: dict[str, int | None] = {}
        self._pending: dict[str, int | None] = {}
        self._lock = asyncio.Lock()
        self._stats: dict[str, Any] = {
            "indexed_today": 0,
            "empty_pages": 0,
            "errors": [],
        }
        self._retry_counts: dict[str, int] = {}
        self._dead_letters: dict[str, int] = {}
        self._requeue_after: dict[str, bool] = {}
        self._max_retries = max_retries
        self._max_sweeps = max_sweeps
        self._dirty: bool = False
        self._store_path = os.path.join(working_dir, "queue_store.json")
        self._persist_interval = persist_interval
        self._max_stored_errors = max_stored_errors
        self._load_state()

    async def enqueue(self, page_id: str, version: int | None = None) -> None:
        """Add page_id to re-index queue (idempotent).

        ``version`` is the target version to index (may be None when unknown
        at enqueue time; the worker learns it at fetch time).

        If the page is currently being indexed (in work), the request is not
        dropped: a requeue-after flag is set so the page is put back in the
        queue once the in-work phase ends (consumed by
        :meth:`consume_requeue_after`).
        """
        async with self._lock:
            if page_id in self._pending:
                self._requeue_after[page_id] = True
                self._dirty = True
                logger.info(
                    f"Re-index requested for in-work page={page_id} — "
                    f"will re-queue when done"
                )
                return
            self._queue[page_id] = version
            self._dirty = True
        logger.info(
            f"Enqueued for re-index: page={page_id} (queue={len(self._queue)})"
        )

    async def get_status(self) -> dict[str, Any]:
        async with self._lock:
            return QueueStatus(
                queue_size=len(self._queue),
                currently_indexing=sorted(self._pending),
                total_indexed_today=self._stats["indexed_today"],
                empty_pages=self._stats.get("empty_pages", 0),
                last_errors=self._stats["errors"][-20:],
                in_work_versions={
                    pid: ver
                    for pid, ver in self._pending.items()
                    if ver is not None
                },
                dead_letters=dict(self._dead_letters),
            ).model_dump()

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def queue_size(self) -> int:
        return len(self._queue)

    async def set_in_work(self, page_id: str, version: int | None = None) -> None:
        """Record that ``page_id`` is being indexed at ``version``.

        The version is considered "in work" — search should hide chunks
        belonging to it until it is cleared (i.e. once fully written).
        """
        async with self._lock:
            self._pending[page_id] = version
            self._dirty = True

    async def clear_in_work(self, page_id: str) -> None:
        """Remove ``page_id`` from the in-work set (indexing finished)."""
        async with self._lock:
            self._pending.pop(page_id, None)
            self._dirty = True

    async def requeue(self, page_id: str, version: int | None = None) -> None:
        """Atomically move page_id from in-work back to the queue (shutdown-safe).

        Used on cancellation so the in-flight page is not lost: the pending
        marker is cleared and the page re-added under a single lock.
        """
        async with self._lock:
            self._pending.pop(page_id, None)
            self._queue[page_id] = version
            self._dirty = True
        logger.info(f"Requeued in-flight page={page_id} (queue={len(self._queue)})")

    async def consume_requeue_after(self, page_id: str) -> bool:
        """Consume the requeue-after flag set by :meth:`enqueue`.

        Called by the worker once the page's in-work phase has ended (after
        ``mark_done``). If a re-index was requested while the page was being
        indexed, the page is put back in the queue so the request is not
        lost. If the page is still in work at that moment (a new indexing
        started), the flag is kept and consumed on the next completion.
        Returns True when the page was re-queued.
        """
        async with self._lock:
            if not self._requeue_after.pop(page_id, None):
                return False
            if page_id in self._pending:
                self._requeue_after[page_id] = True
                self._dirty = True
                return False
            self._queue[page_id] = None
            self._dirty = True
        logger.info(f"Re-queued page={page_id} (re-index requested while in work)")
        return True

    async def get_in_work_versions(self) -> dict[str, int]:
        """Return dict[page_id -> version] of pages currently being indexed.

        Only pages with a concrete version are included (None versions cannot
        be meaningfully hidden and are skipped).
        """
        async with self._lock:
            return {
                pid: ver
                for pid, ver in self._pending.items()
                if ver is not None
            }

    async def periodic_persist(self, interval: float | None = None) -> None:
        """Flush dirty state to disk periodically."""
        interval = interval if interval is not None else self._persist_interval
        while True:
            await asyncio.sleep(interval)
            if self._dirty:
                async with self._lock:
                    self._flush_to_disk()

    async def flush_to_disk(self) -> None:
        """Write state to disk under lock (safe for concurrent access)."""
        async with self._lock:
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        """Write current in-memory state to JSON (atomic write).

        On failure the state stays dirty so the next flush retries; a
        persistence outage must never silently drop queue/retry/dead-letter
        state.
        """
        data = {
            "queue": dict(self._queue),
            "pending": dict(self._pending),
            "stats": self._stats,
            "retry_counts": self._retry_counts,
            "dead_letters": dict(self._dead_letters),
            "requeue_after": dict(self._requeue_after),
        }
        try:
            os.makedirs(os.path.dirname(self._store_path) or ".", exist_ok=True)
            tmp = self._store_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self._store_path)
        except Exception as e:
            logger.error(
                f"Queue persistence to {self._store_path} failed: {e!r} — "
                f"state stays dirty, will retry on next flush",
                exc_info=True,
            )
            self._dirty = True
            return
        self._dirty = False

    def _load_state(self) -> None:
        """Load queue state from JSON on startup."""
        try:
            with open(self._store_path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return

        raw_queue = data.get("queue", [])
        self._queue = (
            dict(raw_queue)
            if isinstance(raw_queue, dict)
            else {pid: None for pid in raw_queue}
        )
        self._pending = {}
        raw_pending = data.get("pending", [])
        orphaned = (
            dict(raw_pending)
            if isinstance(raw_pending, dict)
            else {pid: None for pid in raw_pending}
        )
        if orphaned:
            self._queue.update(orphaned)
        self._stats = {
            "indexed_today": 0,
            "empty_pages": data.get("stats", {}).get("empty_pages", 0),
            "errors": data.get("stats", {}).get("errors", []),
        }
        self._retry_counts = data.get("retry_counts", {})
        self._dead_letters = data.get("dead_letters", {})
        self._requeue_after = {
            pid: True for pid in (data.get("requeue_after") or {})
        }
        if self._queue:
            logger.info(
                f"Restored {len(self._queue)} items from persistent queue"
            )

    async def dequeue(self) -> str | None:
        """Pop the OLDEST page_id from the queue under lock (FIFO).

        Returns None if the queue is empty.
        """
        async with self._lock:
            if not self._queue:
                return None
            page_id = next(iter(self._queue))
            version = self._queue.pop(page_id)
            self._pending[page_id] = version
            self._dirty = True
            return page_id

    async def mark_done(self, page_id: str) -> None:
        async with self._lock:
            self._pending.pop(page_id, None)
            self._dirty = True

    async def record_success(self) -> None:
        async with self._lock:
            self._stats["indexed_today"] += 1
            self._dirty = True

    async def record_empty(self) -> None:
        """Count a page that produced 0 chunks while stale data still existed.

        A page legitimately empty produces no chunks and is normal; a page
        that previously had chunks and now yields nothing is suspicious
        (e.g. a chunker regression) — this counter surfaces that signal.
        """
        async with self._lock:
            self._stats["empty_pages"] = self._stats.get("empty_pages", 0) + 1
            self._dirty = True

    async def record_error(self, page_id: str, error: str) -> None:
        async with self._lock:
            self._stats["errors"].append({
                "page_id": page_id,
                "error": error,
                "at": datetime.now(timezone.utc).isoformat(),
            })
            self._stats["errors"] = self._stats["errors"][-self._max_stored_errors:]
            self._dirty = True

    async def should_retry(self, page_id: str) -> bool:
        """Check if page_id is under max_retries. Increments counter."""
        async with self._lock:
            current = self._retry_counts.get(page_id, 0)
            if current >= self._max_retries:
                return False
            self._retry_counts[page_id] = current + 1
            self._dirty = True
            return True

    async def retry_delay(self, page_id: str) -> float:
        """Backoff before the next retry: 1s, 2s, 4s for retries 1/2/3."""
        async with self._lock:
            count = self._retry_counts.get(page_id, 0)
            return 2 ** max(0, count - 1)

    async def record_dead_letter(self, page_id: str) -> None:
        """Mark page_id as dead-lettered (retries exhausted)."""
        async with self._lock:
            if page_id not in self._dead_letters:
                self._dead_letters[page_id] = 0
            self._dirty = True

    async def sweep_dead_letters(self) -> list[str]:
        """Re-enqueue dead-lettered pages with a fresh retry budget.

        Pages that already received ``max_sweeps`` sweeps stay dead
        (permanent dead letters — e.g. a page permanently gone from
        Confluence would otherwise cause a re-index storm).
        Returns the re-enqueued page ids.
        """
        swept: list[str] = []
        async with self._lock:
            for pid, sweeps in list(self._dead_letters.items()):
                if pid in self._queue or pid in self._pending:
                    continue
                if sweeps >= self._max_sweeps:
                    continue
                self._dead_letters[pid] = sweeps + 1
                self._retry_counts.pop(pid, None)
                self._queue[pid] = None
                swept.append(pid)
            if swept:
                self._dirty = True
        if swept:
            logger.info(f"Dead-letter sweep re-enqueued {len(swept)} page(s)")
        return swept

    async def dead_letter_sweep_loop(self, interval: float = 300.0) -> None:
        """Periodically re-enqueue dead-lettered pages (background task)."""
        while True:
            await asyncio.sleep(interval)
            await self.sweep_dead_letters()

    async def clear_retry(self, page_id: str) -> None:
        """Remove page_id from retry tracking and error records (on success)."""
        async with self._lock:
            self._retry_counts.pop(page_id, None)
            self._dead_letters.pop(page_id, None)
            self._stats["errors"] = [
                e for e in self._stats["errors"] if e.get("page_id") != page_id
            ]
            self._dirty = True
