"""Tests for indexer worker — _index_page and worker loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlassian_rag_mcp.confluence.models import Page, PageContent
from atlassian_rag_mcp.confluence.services.indexer.worker import _index_page, worker


@pytest.fixture
def confluence_client():
    conf = MagicMock()
    conf.get_page = AsyncMock(return_value=PageContent(
        page_title="Test Page",
        html="<p>hello</p>",
        version=2,
        last_updated="2024-01-01",
        parent_id="parent1",
    ))
    return conf


@pytest.fixture
def vector_service():
    vdb = MagicMock()
    vdb.delete_by_filter = AsyncMock(return_value=True)

    fts = MagicMock()
    fts.delete_by_filter = AsyncMock(return_value=True)

    svc = MagicMock()
    svc._ensure_collection = AsyncMock()
    svc.index_page = AsyncMock(return_value=True)
    svc.get_chunks_by_filter = AsyncMock(return_value=[])
    svc.vector_db = vdb
    svc.fts_index = fts
    return svc


@pytest.fixture
def queue():
    q = MagicMock()
    q.dequeue = AsyncMock(side_effect=["p1", None])
    q.record_success = AsyncMock()
    q.record_error = AsyncMock()
    q.mark_done = AsyncMock()
    q.clear_retry = AsyncMock()
    q.should_retry = AsyncMock(return_value=True)
    q.retry_delay = AsyncMock(return_value=1.0)
    q.enqueue = AsyncMock()
    q.set_in_work = AsyncMock()
    q.clear_in_work = AsyncMock()
    q.requeue = AsyncMock()
    q.consume_requeue_after = AsyncMock(return_value=False)
    q.record_dead_letter = AsyncMock()
    q._retry_counts = {}
    q._max_retries = 3
    q._max_sweeps = 3
    q.pending_count = 0
    q.queue_size = 0
    return q


def _stop_worker(idle: float = 5.0):
    """Patch asyncio.sleep so that idle-sized sleeps raise CancelledError.

    Backoff sleeps (< idle) pass through normally; only the empty-queue
    idle sleep triggers cancellation to stop the worker loop.
    """

    async def _fake_sleep(t: float, *args, **kwargs):
        if t >= idle:
            raise asyncio.CancelledError

    return patch("asyncio.sleep", new=_fake_sleep)


class TestIndexPage:
    async def test_index_page_success_delegates_to_index_page(self, confluence_client, vector_service):
        await _index_page(
            page_id="p1",
            confluence_client=confluence_client,
            vector_service=vector_service,
        )
        confluence_client.get_page.assert_awaited_once_with(
            "p1", expand="body.view,version,ancestors"
        )
        vector_service._ensure_collection.assert_not_called()
        vector_service.index_page.assert_awaited_once()
        vector_service.vector_db.delete_by_filter.assert_not_called()
        vector_service.fts_index.delete_by_filter.assert_not_called()

    async def test_index_page_vdb_failure_still_delegates(self, confluence_client, vector_service):
        confluence_client.get_page = AsyncMock(side_effect=RuntimeError("VDB down"))
        with pytest.raises(RuntimeError, match="VDB down"):
            await _index_page(
                page_id="p1",
                confluence_client=confluence_client,
                vector_service=vector_service,
            )

    async def test_index_page_delegates_indexing_to_service(self, confluence_client, vector_service):
        await _index_page(
            page_id="p1",
            confluence_client=confluence_client,
            vector_service=vector_service,
        )
        vector_service.index_page.assert_awaited_once()

    async def test_index_page_raises_on_failure(self, confluence_client, vector_service):
        vector_service.index_page = AsyncMock(side_effect=RuntimeError("index failed"))
        with pytest.raises(RuntimeError, match="index failed"):
            await _index_page(
                page_id="p1",
                confluence_client=confluence_client,
                vector_service=vector_service,
            )

    async def test_index_page_no_ancestors(self, confluence_client, vector_service):
        confluence_client.get_page = AsyncMock(return_value=PageContent(
            page_title="Root",
            html="<p>root</p>",
            version=1,
            last_updated="2024-01-01",
        ))
        await _index_page(
            page_id="p1",
            confluence_client=confluence_client,
            vector_service=vector_service,
        )
        vector_service.index_page.assert_awaited_once()

    async def test_index_page_page_error_raises(self, confluence_client, vector_service):
        confluence_client.get_page = AsyncMock(return_value=Page(error="Not found"))
        with pytest.raises(RuntimeError, match="Failed to fetch page p1: Not found"):
            await _index_page(
                page_id="p1",
                confluence_client=confluence_client,
                vector_service=vector_service,
            )

    @staticmethod
    def _worker_module():
        import importlib
        return importlib.import_module("atlassian_rag_mcp.confluence.services.indexer.worker")

    async def test_index_page_calls_pipeline(self, confluence_client, vector_service):
        with patch.object(self._worker_module(), "chunking_pipeline") as mock_pipe:
            mock_pipe.return_value = []
            await _index_page(
                page_id="p1",
                confluence_client=confluence_client,
                vector_service=vector_service,
            )
        mock_pipe.assert_called_once()
        assert "max_token_size" not in mock_pipe.call_args.kwargs

    async def test_index_page_empty_chunks_leaves_storage_untouched(
        self, confluence_client, vector_service,
    ):
        """0 chunks → no write, no delete; old data is kept intact."""
        with patch.object(self._worker_module(), "chunking_pipeline") as mock_pipe:
            mock_pipe.return_value = []
            await _index_page(
                page_id="p1",
                confluence_client=confluence_client,
                vector_service=vector_service,
            )
        mock_pipe.assert_called_once()
        vector_service.index_page.assert_not_called()
        vector_service.vector_db.delete_by_filter.assert_not_called()
        vector_service.fts_index.delete_by_filter.assert_not_called()

    async def test_index_page_empty_chunks_with_stale_data_records_empty(self):
        """0 chunks + existing chunks in storage → empty_pages signal, no write."""
        conf = MagicMock()
        conf.get_page = AsyncMock(return_value=PageContent(
            page_title="T", html="", version=3, last_updated="2024-01-01",
        ))
        vsvc = MagicMock()
        vsvc._ensure_collection = AsyncMock()
        vsvc.index_page = AsyncMock(return_value=True)
        vsvc.get_chunks_by_filter = AsyncMock(return_value=[{"id": "pt1"}])
        q = MagicMock()
        q.record_empty = AsyncMock()
        q.set_in_work = AsyncMock()
        with patch.object(self._worker_module(), "chunking_pipeline") as mock_pipe:
            mock_pipe.return_value = []
            await _index_page(
                page_id="p1", confluence_client=conf,
                vector_service=vsvc, index_queue=q,
            )
        vsvc.get_chunks_by_filter.assert_awaited_once_with(page_id="p1", limit=1)
        q.record_empty.assert_awaited_once_with()
        vsvc.index_page.assert_not_called()
        vector_db_mock = vsvc.vector_db
        vector_db_mock.delete_by_filter.assert_not_called()

    async def test_index_page_empty_chunks_without_stale_data_is_quiet(self):
        """0 chunks + nothing in storage → INFO only, no empty_pages bump."""
        conf = MagicMock()
        conf.get_page = AsyncMock(return_value=PageContent(
            page_title="T", html="", version=1, last_updated="2024-01-01",
        ))
        vsvc = MagicMock()
        vsvc._ensure_collection = AsyncMock()
        vsvc.index_page = AsyncMock(return_value=True)
        vsvc.get_chunks_by_filter = AsyncMock(return_value=[])
        q = MagicMock()
        q.record_empty = AsyncMock()
        q.set_in_work = AsyncMock()
        with patch.object(self._worker_module(), "chunking_pipeline") as mock_pipe:
            mock_pipe.return_value = []
            await _index_page(
                page_id="p1", confluence_client=conf,
                vector_service=vsvc, index_queue=q,
            )
        q.record_empty.assert_not_called()
        vsvc.index_page.assert_not_called()

    async def test_index_page_sets_space_key_on_chunks(self):
        conf = MagicMock()
        conf.get_page = AsyncMock(return_value=PageContent(
            page_title="Test Page",
            html="<p>hello</p>",
            version=2,
            last_updated="2024-01-01",
            space_key="AAA",
        ))
        from atlassian_rag_mcp.confluence.models import ChunkPayload
        chunk = ChunkPayload(
            chunk_id="c1", chunk_type="section", text="hello",
            page_id="p1", page_title="Test Page",
        )
        vsvc = MagicMock()
        vsvc._ensure_collection = AsyncMock()
        vsvc.index_page = AsyncMock(return_value=True)
        with patch.object(self._worker_module(), "chunking_pipeline") as mock_pipe:
            mock_pipe.return_value = [chunk]
            await _index_page(
                page_id="p1", confluence_client=conf, vector_service=vsvc,
            )
        assert chunk.space_key == "AAA"
        passed_chunks = vsvc.index_page.call_args.kwargs["chunks"]
        assert all(c.space_key == "AAA" for c in passed_chunks)


class TestWorker:
    async def test_drains_queue_and_records_success(self, queue, confluence_client, vector_service):
        with _stop_worker():
            with pytest.raises(asyncio.CancelledError):
                await worker(
                    queue=queue,
                    confluence_client=confluence_client,
                    vector_service=vector_service,
                    worker_id=0,
                )
        queue.record_success.assert_awaited_once()
        queue.clear_retry.assert_awaited_once_with("p1")
        queue.clear_in_work.assert_awaited_once()
        queue.mark_done.assert_awaited_once()
        queue.enqueue.assert_not_called()
        queue.requeue.assert_not_called()

    async def test_records_error_and_retries_then_dead_letters(
        self, queue, confluence_client, vector_service,
    ):
        """Two attempts: first retries, second exhausts → dead-lettered."""
        vector_service.index_page = AsyncMock(side_effect=RuntimeError("index failed"))
        queue.should_retry = AsyncMock(side_effect=[True, False])
        with _stop_worker():
            with pytest.raises(asyncio.CancelledError):
                await worker(
                    queue=queue,
                    confluence_client=confluence_client,
                    vector_service=vector_service,
                    worker_id=0,
                )
        assert queue.record_error.await_count == 2
        assert queue.should_retry.await_count == 2
        queue.retry_delay.assert_awaited_once_with("p1")
        queue.record_dead_letter.assert_awaited_once_with("p1")
        queue.clear_in_work.assert_awaited_once()
        queue.mark_done.assert_awaited_once()
        queue.enqueue.assert_not_called()
        queue.requeue.assert_not_called()

    async def test_dead_letter_when_retries_exhausted(self, queue, confluence_client, vector_service):
        vector_service.index_page = AsyncMock(side_effect=RuntimeError("index failed"))
        queue.should_retry = AsyncMock(return_value=False)
        with _stop_worker():
            with pytest.raises(asyncio.CancelledError):
                await worker(
                    queue=queue,
                    confluence_client=confluence_client,
                    vector_service=vector_service,
                    worker_id=0,
                )
        queue.record_error.assert_awaited_once()
        queue.should_retry.assert_awaited_once_with("p1")
        queue.record_dead_letter.assert_awaited_once_with("p1")
        queue.enqueue.assert_not_called()
        queue.requeue.assert_not_called()
        queue.clear_in_work.assert_awaited_once()
        queue.mark_done.assert_awaited_once()

    async def test_cancel_during_backoff_requeues(self, queue, confluence_client, vector_service):
        """Cancellation during backoff sleep requeues the page atomically."""
        vector_service.index_page = AsyncMock(side_effect=RuntimeError("fail"))
        queue.should_retry = AsyncMock(return_value=True)
        queue.retry_delay = AsyncMock(return_value=10.0)
        with _stop_worker():
            with pytest.raises(asyncio.CancelledError):
                await worker(
                    queue=queue,
                    confluence_client=confluence_client,
                    vector_service=vector_service,
                    worker_id=0,
                )
        queue.requeue.assert_awaited_once_with("p1")
        queue.enqueue.assert_not_called()

    async def test_empty_queue_sleeps(self, queue, confluence_client, vector_service):
        queue.dequeue = AsyncMock(return_value=None)
        with _stop_worker():
            with pytest.raises(asyncio.CancelledError):
                await worker(
                    queue=queue,
                    confluence_client=confluence_client,
                    vector_service=vector_service,
                    worker_id=0,
                )
        queue.dequeue.assert_awaited()

    async def test_ensure_collection_once_at_start(
        self, queue, confluence_client, vector_service,
    ):
        """Collection is ensured once per worker, not once per indexed page."""
        queue.dequeue = AsyncMock(side_effect=["p1", "p2", None])
        with _stop_worker():
            with pytest.raises(asyncio.CancelledError):
                await worker(
                    queue=queue,
                    confluence_client=confluence_client,
                    vector_service=vector_service,
                    worker_id=0,
                )
        assert vector_service._ensure_collection.await_count == 1
        assert queue.record_success.await_count == 2

    async def test_ensure_collection_retries_until_ready(
        self, queue, confluence_client, vector_service,
    ):
        """A transient VectorDB/embedding outage at startup must not kill the worker."""
        vector_service._ensure_collection = AsyncMock(
            side_effect=[RuntimeError("vdb down"), None],
        )
        queue.dequeue = AsyncMock(side_effect=["p1", None])

        async def _fake_sleep(t: float, *args, **kwargs):
            if t >= 7.0:
                raise asyncio.CancelledError

        with patch("asyncio.sleep", new=_fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await worker(
                    queue=queue,
                    confluence_client=confluence_client,
                    vector_service=vector_service,
                    worker_id=0,
                    idle_sleep=7.0,
                )
        assert vector_service._ensure_collection.await_count == 2
        queue.record_success.assert_awaited_once()


class TestWorkerRealQueue:
    """Worker against a real IndexQueue — catches pending/requeue regressions."""

    @pytest.fixture
    def real_queue(self, tmp_path):
        from atlassian_rag_mcp.confluence.services.indexer.queue import IndexQueue
        return IndexQueue(
            str(tmp_path),
            persist_interval=999.0,
            max_retries=3,
            max_sweeps=1,
        )

    async def test_fetch_failure_not_lost_through_retries(
        self, real_queue, confluence_client, vector_service,
    ):
        """Pre-try failure (get_page raises) must retry in-work, not be dropped."""
        confluence_client.get_page = AsyncMock(side_effect=RuntimeError("net down"))
        await real_queue.enqueue("p1")
        with _stop_worker():
            with pytest.raises(asyncio.CancelledError):
                await worker(
                    queue=real_queue,
                    confluence_client=confluence_client,
                    vector_service=vector_service,
                    worker_id=0,
                )
        status = await real_queue.get_status()
        assert status["dead_letters"] == {"p1": 0}
        assert real_queue._retry_counts.get("p1") == 3
        assert real_queue.queue_size == 0
        assert real_queue.pending_count == 0

    async def test_transient_fetch_failure_recovers(
        self, real_queue, confluence_client, vector_service,
    ):
        """Failure on first attempt, success on retry → indexed, no dead letter."""
        confluence_client.get_page = AsyncMock(side_effect=[
            RuntimeError("net down"),
            PageContent(
                page_title="Test Page",
                html="<p>hello</p>",
                version=2,
                last_updated="2024-01-01",
                parent_id="parent1",
            ),
        ])
        await real_queue.enqueue("p1")
        with _stop_worker():
            with pytest.raises(asyncio.CancelledError):
                await worker(
                    queue=real_queue,
                    confluence_client=confluence_client,
                    vector_service=vector_service,
                    worker_id=0,
                )
        status = await real_queue.get_status()
        assert status["total_indexed_today"] == 1
        assert status["dead_letters"] == {}
        assert real_queue._retry_counts == {}
        assert real_queue.queue_size == 0
        assert real_queue.pending_count == 0

    async def test_empty_chunks_removes_page_without_writing(
        self, real_queue, confluence_client, vector_service,
    ):
        """0 chunks → page dropped from queue, storage untouched, no dead-letter."""
        import importlib
        wm = importlib.import_module(
            "atlassian_rag_mcp.confluence.services.indexer.worker"
        )
        with patch.object(wm, "chunking_pipeline", return_value=[]):
            await real_queue.enqueue("p1")
            with _stop_worker():
                with pytest.raises(asyncio.CancelledError):
                    await worker(
                        queue=real_queue,
                        confluence_client=confluence_client,
                        vector_service=vector_service,
                        worker_id=0,
                    )
        status = await real_queue.get_status()
        assert status["total_indexed_today"] == 1
        assert status["dead_letters"] == {}
        assert real_queue._retry_counts == {}
        assert real_queue.queue_size == 0
        assert real_queue.pending_count == 0
        vector_service.index_page.assert_not_called()
        vector_service.vector_db.delete_by_filter.assert_not_called()
        vector_service.fts_index.delete_by_filter.assert_not_called()

    async def test_enqueue_during_indexing_is_reprocessed(
        self, real_queue, confluence_client, vector_service,
    ):
        """An enqueue that arrives while a page is in work is not lost:
        the page is re-queued once the in-work phase ends."""
        calls = {"n": 0}

        async def _index(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                await real_queue.enqueue("p1")
        vector_service.index_page = AsyncMock(side_effect=_index)
        await real_queue.enqueue("p1")
        with _stop_worker():
            with pytest.raises(asyncio.CancelledError):
                await worker(
                    queue=real_queue,
                    confluence_client=confluence_client,
                    vector_service=vector_service,
                    worker_id=0,
                )
        assert vector_service.index_page.await_count == 2
        status = await real_queue.get_status()
        assert status["total_indexed_today"] == 2
        assert real_queue.queue_size == 0
        assert real_queue.pending_count == 0
        assert real_queue._requeue_after == {}

    async def test_cancel_during_backoff_returns_page_to_queue(
        self, real_queue, confluence_client, vector_service,
    ):
        """Cancellation during the retry backoff must requeue the page (persist)."""
        confluence_client.get_page = AsyncMock(side_effect=RuntimeError("net down"))
        await real_queue.enqueue("p1")
        async def _fake_sleep(t, *a, **k):
            raise asyncio.CancelledError
        with patch("asyncio.sleep", new=_fake_sleep):
            with pytest.raises(asyncio.CancelledError):
                await worker(
                    queue=real_queue,
                    confluence_client=confluence_client,
                    vector_service=vector_service,
                    worker_id=0,
                )
        assert real_queue.queue_size == 1
        assert real_queue.pending_count == 0
        status = await real_queue.get_status()
        assert status["dead_letters"] == {}
