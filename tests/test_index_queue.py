"""IndexQueue: isolated async unit tests with file I/O mocks.

All methods are async after the asyncio.Lock migration.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from atlassian_rag_mcp.confluence.services.indexer.queue import IndexQueue


@pytest.fixture
def mock_fs():
    """Mocks all filesystem operations so IndexQueue never touches real disk."""
    makedirs = MagicMock()
    path_exists = MagicMock(return_value=False)
    os_replace = MagicMock()
    json_dump = MagicMock()
    mock_file = mock_open()

    with (
        patch("os.makedirs", makedirs),
        patch("os.path.exists", path_exists),
        patch("os.replace", os_replace),
        patch("json.dump", json_dump),
        patch("builtins.open", mock_file),
    ):
        yield {
            "makedirs": makedirs,
            "path_exists": path_exists,
            "os_replace": os_replace,
            "json_dump": json_dump,
            "mock_file": mock_file,
        }


@pytest.fixture
async def queue(mock_fs):
    q = IndexQueue(working_dir="/tmp/test_queue")
    return q


class TestInit:
    async def test_creates_empty_queue(self, queue):
        assert len(queue._queue) == 0
        assert len(queue._pending) == 0
        assert queue._stats["indexed_today"] == 0
        assert queue._dirty is False

    async def test_loads_existing_state(self, mock_fs):
        mock_fs["path_exists"].return_value = True
        mock_fs["mock_file"].return_value.read.return_value = json.dumps({
            "queue": ["p1", "p2"],
            "pending": ["p3"],
            "stats": {"indexed_today": 5, "errors": []},
        })
        q = IndexQueue(working_dir="/tmp/test_queue")
        assert "p1" in q._queue
        assert "p2" in q._queue
        assert "p3" in q._queue
        assert q._stats["indexed_today"] == 0

    async def test_load_does_not_fail_on_missing_file(self, mock_fs):
        mock_fs["path_exists"].return_value = False
        q = IndexQueue(working_dir="/tmp/test_queue")
        assert len(q._queue) == 0
        assert len(q._pending) == 0

    async def test_loads_empty_pages_stat(self, mock_fs):
        mock_fs["path_exists"].return_value = True
        mock_fs["mock_file"].return_value.read.return_value = json.dumps({
            "queue": [],
            "pending": [],
            "stats": {"indexed_today": 5, "empty_pages": 7, "errors": []},
        })
        q = IndexQueue(working_dir="/tmp/test_queue")
        assert q._stats["empty_pages"] == 7
        assert q._stats["indexed_today"] == 0


class TestEnqueue:
    async def test_adds_page_id(self, queue):
        await queue.enqueue("p1")
        assert "p1" in queue._queue
        assert queue._dirty is True

    async def test_no_duplicate(self, queue):
        await queue.enqueue("p1")
        await queue.enqueue("p1")
        assert len(queue._queue) == 1

    async def test_dirty_flag_set(self, queue):
        await queue.enqueue("p1")
        assert queue._dirty is True


class TestDequeue:
    async def test_returns_page_id(self, queue):
        await queue.enqueue("p1")
        pid = await queue.dequeue()
        assert pid == "p1"
        assert "p1" not in queue._queue
        assert "p1" in queue._pending

    async def test_empty_queue_returns_none(self, queue):
        pid = await queue.dequeue()
        assert pid is None

    async def test_multiple_dequeues_fifo(self, queue):
        await queue.enqueue("p1")
        await queue.enqueue("p2")
        p1 = await queue.dequeue()
        p2 = await queue.dequeue()
        assert p1 == "p1"
        assert p2 == "p2"


class TestMarkDone:
    async def test_removes_from_pending(self, queue):
        await queue.enqueue("p1")
        await queue.dequeue()
        await queue.mark_done("p1")
        assert "p1" not in queue._pending

    async def test_increments_success_count_via_record_success(self, queue):
        await queue.record_success()
        assert queue._stats["indexed_today"] == 1

    async def test_record_empty_increments_stat(self, queue):
        await queue.record_empty()
        assert queue._stats["empty_pages"] == 1
        status = await queue.get_status()
        assert status["empty_pages"] == 1


class TestRecordError:
    async def test_adds_error_to_stats(self, queue):
        await queue.record_error("p1", "Something went wrong")
        assert len(queue._stats["errors"]) == 1
        assert queue._stats["errors"][0]["page_id"] == "p1"
        assert "error" in queue._stats["errors"][0]
        assert "at" in queue._stats["errors"][0]

    async def test_limits_errors_to_100(self, queue):
        for i in range(110):
            await queue.record_error(f"p{i}", f"err{i}")
        assert len(queue._stats["errors"]) == 100


class TestGetStatus:
    async def test_returns_status_dict(self, queue):
        await queue.enqueue("p1")
        status = await queue.get_status()
        assert status["queue_size"] == 1
        assert status["currently_indexing"] == []
        assert status["total_indexed_today"] == 0
        assert status["empty_pages"] == 0
        assert status["last_errors"] == []

    async def test_shows_currently_indexing(self, queue):
        await queue.enqueue("p1")
        await queue.dequeue()
        status = await queue.get_status()
        assert "p1" in status["currently_indexing"]

    async def test_shows_last_errors(self, queue):
        await queue.record_error("p1", "err")
        status = await queue.get_status()
        assert len(status["last_errors"]) == 1


class TestFlushToDisk:
    async def test_writes_atomically(self, queue, mock_fs):
        await queue.enqueue("p1")
        await queue.flush_to_disk()
        assert mock_fs["json_dump"].called
        assert mock_fs["os_replace"].called
        assert queue._dirty is False

    async def test_failure_keeps_state_dirty(self, queue, mock_fs):
        """A write/OS failure must not be silent: state stays dirty for retry."""
        mock_fs["json_dump"].side_effect = OSError("disk full")
        await queue.enqueue("p1")
        await queue.flush_to_disk()
        assert queue._dirty is True
        assert "p1" in queue._queue
        assert not mock_fs["os_replace"].called


class TestInWorkVersions:
    async def test_set_in_work_records_version(self, queue):
        await queue.set_in_work("p1", 5)
        assert await queue.get_in_work_versions() == {"p1": 5}

    async def test_clear_in_work_removes(self, queue):
        await queue.set_in_work("p1", 5)
        await queue.clear_in_work("p1")
        assert await queue.get_in_work_versions() == {}

    async def test_none_version_not_hidden(self, queue):
        await queue.set_in_work("p1", None)
        assert await queue.get_in_work_versions() == {}

    async def test_dequeue_to_pending_clears_from_queue(self, queue):
        await queue.enqueue("p1", 3)
        pid = await queue.dequeue()
        assert pid == "p1"
        status = await queue.get_status()
        assert "p1" in status["currently_indexing"]

    async def test_loads_versioned_state(self, mock_fs):
        mock_fs["path_exists"].return_value = True
        mock_fs["mock_file"].return_value.read.return_value = json.dumps({
            "queue": {"p1": 2, "p2": None},
            "pending": {"p3": 4},
            "stats": {"indexed_today": 5, "errors": []},
        })
        q = IndexQueue(working_dir="/tmp/test_queue")
        assert q._queue.get("p1") == 2
        assert q._queue.get("p2") is None
        assert q._queue.get("p3") == 4


class TestRetryBackoff:
    async def test_retry_delay_backoff_1_2_4(self, queue):
        assert await queue.retry_delay("p1") == 1
        assert await queue.should_retry("p1")
        assert await queue.retry_delay("p1") == 1
        assert await queue.should_retry("p1")
        assert await queue.retry_delay("p1") == 2
        assert await queue.should_retry("p1")
        assert await queue.retry_delay("p1") == 4
        assert not await queue.should_retry("p1")

    async def test_clear_retry_removes_page_errors(self, queue):
        await queue.record_error("p1", "boom")
        await queue.record_error("p2", "other")
        await queue.clear_retry("p1")
        errors = (await queue.get_status())["last_errors"]
        assert all(e["page_id"] != "p1" for e in errors)
        assert any(e["page_id"] == "p2" for e in errors)

    async def test_clear_retry_resets_dead_letter(self, queue):
        await queue.record_dead_letter("p1")
        await queue.clear_retry("p1")
        assert (await queue.get_status())["dead_letters"] == {}


class TestDeadLetterSweep:
    async def test_record_and_sweep_fresh_budget(self, queue):
        for _ in range(3):
            await queue.should_retry("p1")
        assert not await queue.should_retry("p1")
        await queue.record_dead_letter("p1")
        assert (await queue.get_status())["dead_letters"] == {"p1": 0}
        swept = await queue.sweep_dead_letters()
        assert swept == ["p1"]
        assert queue.queue_size == 1
        assert await queue.should_retry("p1")
        await queue.clear_retry("p1")
        assert (await queue.get_status())["dead_letters"] == {}

    async def test_sweep_respects_max_sweeps(self, queue):
        await queue.record_dead_letter("p1")
        for _ in range(3):
            await queue.sweep_dead_letters()
            await queue.dequeue()
            for _ in range(3):
                await queue.should_retry("p1")
            await queue.mark_done("p1")
            await queue.record_dead_letter("p1")
        await queue.dequeue()
        await queue.mark_done("p1")
        status = await queue.get_status()
        assert queue.queue_size == 0
        assert status["dead_letters"] == {"p1": 3}
        assert await queue.sweep_dead_letters() == []

    async def test_sweep_skips_pages_already_in_queue(self, queue):
        await queue.record_dead_letter("p1")
        await queue.enqueue("p1")
        assert await queue.sweep_dead_letters() == []
        assert (await queue.get_status())["dead_letters"] == {"p1": 0}

    async def test_dead_letters_persist_and_load(self, mock_fs):
        mock_fs["path_exists"].return_value = True
        mock_fs["mock_file"].return_value.read.return_value = json.dumps({
            "queue": {},
            "pending": {},
            "stats": {"indexed_today": 0, "errors": []},
            "dead_letters": {"p1": 1},
        })
        q = IndexQueue(working_dir="/tmp/test_queue")
        assert q._dead_letters == {"p1": 1}


class TestRequeue:
    async def test_requeue_moves_pending_to_queue(self, queue):
        await queue.enqueue("p1")
        await queue.dequeue()
        assert queue.pending_count == 1
        await queue.requeue("p1")
        assert queue.pending_count == 0
        assert queue.queue_size == 1
        assert "p1" in queue._queue
        assert queue._dirty is True

    async def test_requeue_adds_when_not_pending(self, queue):
        await queue.requeue("p1")
        assert queue.queue_size == 1

    async def test_requeue_clears_in_work_version(self, queue):
        await queue.enqueue("p1", 5)
        await queue.dequeue()
        await queue.set_in_work("p1", 7)
        await queue.requeue("p1")
        assert await queue.get_in_work_versions() == {}
        assert queue.queue_size == 1


class TestRequeueAfter:
    """Enqueue while a page is in work must not be lost (requeue-after flag)."""

    async def test_enqueue_while_in_work_sets_flag(self, queue):
        await queue.set_in_work("p1", 5)
        await queue.enqueue("p1")
        assert queue._requeue_after == {"p1": True}
        assert "p1" not in queue._queue
        assert queue._dirty is True

    async def test_enqueue_not_in_work_no_flag(self, queue):
        await queue.enqueue("p1")
        assert "p1" in queue._queue
        assert queue._requeue_after == {}

    async def test_flag_idempotent(self, queue):
        await queue.set_in_work("p1", 5)
        await queue.enqueue("p1")
        await queue.enqueue("p1")
        assert queue._requeue_after == {"p1": True}

    async def test_consume_requeues_when_flag_set(self, queue):
        await queue.set_in_work("p1", 5)
        await queue.enqueue("p1")
        await queue.mark_done("p1")
        assert await queue.consume_requeue_after("p1") is True
        assert "p1" in queue._queue
        assert queue._requeue_after == {}

    async def test_consume_no_flag_noop(self, queue):
        assert await queue.consume_requeue_after("p1") is False
        assert queue.queue_size == 0
        assert queue._requeue_after == {}

    async def test_consume_keeps_flag_if_still_in_work(self, queue):
        await queue.set_in_work("p1", 5)
        await queue.enqueue("p1")
        assert await queue.consume_requeue_after("p1") is False
        assert queue._requeue_after == {"p1": True}
        await queue.mark_done("p1")
        assert await queue.consume_requeue_after("p1") is True
        assert "p1" in queue._queue

    async def test_flush_includes_flag(self, queue, mock_fs):
        await queue.set_in_work("p1", 5)
        await queue.enqueue("p1")
        await queue.flush_to_disk()
        data = mock_fs["json_dump"].call_args[0][0]
        assert data["requeue_after"] == {"p1": True}

    async def test_load_restores_flag(self, mock_fs):
        mock_fs["path_exists"].return_value = True
        mock_fs["mock_file"].return_value.read.return_value = json.dumps({
            "queue": {},
            "pending": {},
            "stats": {"indexed_today": 0, "errors": []},
            "requeue_after": {"p1": True},
        })
        q = IndexQueue(working_dir="/tmp/test_queue")
        assert q._requeue_after == {"p1": True}
