"""Tests for check_relevant — page version relevance detection."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atlassian_rag_mcp.confluence.services.check_relevant import check_relevant


@pytest.fixture
def confluence():
    c = MagicMock()
    c.get_page = AsyncMock()
    return c


@pytest.fixture
def index_queue():
    q = MagicMock()
    q.enqueue = AsyncMock()
    return q


def _make_result(page_id: str, version: int = 1) -> dict:
    return {"page_id": page_id, "version": version, "text": "foo"}


class TestEmptyInput:
    async def test_empty_list_returns_empty(self, confluence):
        assert await check_relevant([], confluence) == []

    async def test_none_list_handled_gracefully(self, confluence):
        assert await check_relevant([], confluence) == []


class TestVersionMatch:
    async def test_all_fresh_keeps_all(self, confluence, index_queue):
        results = [_make_result("p1", 2), _make_result("p2", 3)]
        confluence.get_page.side_effect = [
            MagicMock(version=2),
            MagicMock(version=3),
        ]
        out = await check_relevant(results, confluence, index_queue)
        assert len(out) == 2
        index_queue.enqueue.assert_not_called()

    async def test_single_mixed_versions(self, confluence, index_queue):
        results = [_make_result("p1", 1,), _make_result("p2", 3)]
        confluence.get_page.side_effect = [
            MagicMock(version=2),
            MagicMock(version=3),
        ]
        out = await check_relevant(results, confluence, index_queue)
        assert len(out) == 1
        assert out[0]["page_id"] == "p2"


class TestStaleDetection:
    async def test_removes_stale_page(self, confluence, index_queue):
        results = [_make_result("p1", 1)]
        confluence.get_page.return_value = MagicMock(version=2)
        out = await check_relevant(results, confluence, index_queue)
        assert out == []

    async def test_enqueues_stale_page(self, confluence, index_queue):
        results = [_make_result("p1", 1)]
        confluence.get_page.return_value = MagicMock(version=2)
        await check_relevant(results, confluence, index_queue)
        index_queue.enqueue.assert_awaited_once_with("p1")

    async def test_no_enqueue_without_queue(self, confluence):
        results = [_make_result("p1", 1)]
        confluence.get_page.return_value = MagicMock(version=2)
        await check_relevant(results, confluence)



class TestVersionZero:
    async def test_stored_version_zero_skips(self, confluence, index_queue):
        results = [_make_result("p1", 0)]
        confluence.get_page.return_value = MagicMock(version=2)
        out = await check_relevant(results, confluence, index_queue)
        assert len(out) == 1
        index_queue.enqueue.assert_not_called()

    async def test_live_version_zero_stale(self, confluence, index_queue):
        results = [_make_result("p1", 1)]
        confluence.get_page.return_value = MagicMock(version=0)
        out = await check_relevant(results, confluence, index_queue)
        assert len(out) == 0
        index_queue.enqueue.assert_awaited_once_with("p1")


class TestBatchSize:
    async def test_respects_batch_size(self, confluence, index_queue):
        results = [_make_result(f"p{i}", 1) for i in range(20)]
        confluence.get_page.side_effect = [
            MagicMock(version=2) for _ in range(10)
        ]
        await check_relevant(results, confluence, index_queue, batch_size=10)
        assert confluence.get_page.call_count == 10

    async def test_default_batch_is_10(self, confluence, index_queue):
        results = [_make_result(f"p{i}", 1) for i in range(15)]
        confluence.get_page.side_effect = [
            MagicMock(version=2) for _ in range(10)
        ]
        await check_relevant(results, confluence, index_queue)
        assert confluence.get_page.call_count == 10


class TestErrorHandling:
    async def test_get_page_error_logged_and_skipped(self, confluence, index_queue):
        results = [_make_result("p1", 1)]
        confluence.get_page.side_effect = RuntimeError("API down")
        out = await check_relevant(results, confluence, index_queue)
        assert len(out) == 1
        index_queue.enqueue.assert_not_called()


class TestDeduplication:
    async def test_duplicate_page_ids_deduplicated(self, confluence, index_queue):
        results = [
            _make_result("p1", 1),
            _make_result("p1", 1),
        ]
        confluence.get_page.return_value = MagicMock(version=2)
        out = await check_relevant(results, confluence, index_queue)
        assert confluence.get_page.call_count == 1
        assert len(out) == 0
