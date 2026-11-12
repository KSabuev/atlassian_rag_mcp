"""Unit tests for Qdrant error propagation (mocked client — no live server).

The real Qdrant backend must never swallow write/create failures: errors
must propagate upward as BaseError with the underlying cause preserved.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atlassian_rag_mcp.exceptions import BaseError
from atlassian_rag_mcp.storage.vectordb.qdrant_client import QdrantClient


def _make_client(**kwargs):
    c = QdrantClient(
        host="127.0.0.1",
        port=6333,
        collection="utest",
        vector_size=4,
        **kwargs,
    )
    mock = AsyncMock()
    c._client = mock
    return c, mock


class TestCreateCollectionErrors:
    async def test_raises_backend_error_on_sdk_failure(self):
        c, mock = _make_client()
        mock.collection_exists = AsyncMock(return_value=False)
        mock.create_collection = AsyncMock(side_effect=RuntimeError("disk full"))
        with pytest.raises(BaseError, match="disk full"):
            await c.create_collection(vector_size=4)

    async def test_returns_true_when_collection_exists(self):
        c, mock = _make_client()
        mock.collection_exists = AsyncMock(return_value=True)
        assert await c.create_collection(vector_size=4) is True


class TestUpsertErrors:
    @staticmethod
    def _points(n=2):
        return [
            {
                "id": f"c{i}",
                "vector": [0.1 * i, 0.2, 0.3, 0.4],
                "payload": {"chunk_id": f"c{i}", "text": "x", "page_id": "p1"},
            }
            for i in range(1, n + 1)
        ]

    async def test_raises_and_rolls_back_partial_upsert(self):
        c, mock = _make_client(upsert_batch_size=1)
        mock.upsert = AsyncMock(side_effect=[None, RuntimeError("upstream down")])
        mock.delete = AsyncMock()
        with pytest.raises(BaseError, match="upstream down"):
            await c.upsert_points(self._points(2))
        mock.delete.assert_awaited_once()

    async def test_raises_without_rollback_when_nothing_written(self):
        c, mock = _make_client(upsert_batch_size=2)
        mock.upsert = AsyncMock(side_effect=RuntimeError("upstream down"))
        mock.delete = AsyncMock()
        with pytest.raises(BaseError, match="upstream down"):
            await c.upsert_points(self._points(2))
        mock.delete.assert_not_awaited()

    async def test_returns_true_for_empty_points(self):
        c, mock = _make_client()
        assert await c.upsert_points([]) is True
