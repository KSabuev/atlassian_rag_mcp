"""Tests for the FTS rebuild helper (storage.fts_rebuild)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atlassian_rag_mcp.storage.fts_rebuild import rebuild_fts_from_vdb


def _vdb(batches) -> MagicMock:
    vdb = MagicMock()
    vdb.scroll_points = AsyncMock(side_effect=batches)
    return vdb


def _fts() -> MagicMock:
    fts = MagicMock()
    fts.delete_by_filter = AsyncMock(return_value=True)
    fts.index_batch = AsyncMock(return_value=True)
    return fts


class TestRebuildFtsFromVdb:
    async def test_scoped_delete_and_scroll_filter(self):
        vdb = _vdb([([{"payload": {"chunk_id": "c1"}}], None)])
        fts = _fts()

        n = await rebuild_fts_from_vdb(vdb, fts, ["S1", "S2"])

        assert n == 1
        fts.delete_by_filter.assert_awaited_once_with(
            {"must": [{"key": "space_key", "match": {"any": ["S1", "S2"]}}]}
        )
        assert vdb.scroll_points.await_args.kwargs["filter_condition"] == {
            "must": [{"key": "space_key", "match": {"any": ["S1", "S2"]}}]
        }

    async def test_dedupes_composites_across_batches(self):
        """Sub-points of one split composite share a chunk_id and can straddle
        the scroll batch boundary — the global dedup must index each logical
        chunk exactly once."""
        batch1 = [{"payload": {"chunk_id": "base"}} for _ in range(2)]
        batch2 = [{"payload": {"chunk_id": "base"}} for _ in range(2)]
        batch2.append({"payload": {"chunk_id": "solo"}})
        vdb = _vdb([(batch1, 3), (batch2, None)])
        fts = _fts()

        n = await rebuild_fts_from_vdb(vdb, fts, ["S1"])

        assert n == 2
        written = [d for call in fts.index_batch.await_args_list for d in call.args[0]]
        assert [d["chunk_id"] for d in written] == ["base", "solo"]

    async def test_empty_index_returns_zero(self):
        vdb = _vdb([([], None)])
        fts = _fts()

        n = await rebuild_fts_from_vdb(vdb, fts, ["S1"])

        assert n == 0
        fts.index_batch.assert_not_called()

    async def test_requires_at_least_one_space(self):
        with pytest.raises(ValueError, match="at least one space"):
            await rebuild_fts_from_vdb(_vdb([]), _fts(), [])

    async def test_empty_spaces_after_strip_raise(self):
        with pytest.raises(ValueError, match="at least one space"):
            await rebuild_fts_from_vdb(_vdb([]), _fts(), [" ", ""])

    async def test_delete_failure_raises(self):
        fts = _fts()
        fts.delete_by_filter = AsyncMock(return_value=False)

        with pytest.raises(RuntimeError, match="delete_by_filter"):
            await rebuild_fts_from_vdb(_vdb([]), fts, ["S1"])

    async def test_index_batch_failure_raises(self):
        vdb = _vdb([([{"payload": {"chunk_id": "c1"}}], None)])
        fts = _fts()
        fts.index_batch = AsyncMock(return_value=False)

        with pytest.raises(RuntimeError, match="index_batch"):
            await rebuild_fts_from_vdb(vdb, fts, ["S1"])

    async def test_points_without_payload_skipped(self):
        vdb = _vdb([(
            [{"payload": None}, {"payload": {"chunk_id": "c1"}}, {}],
            None,
        )])
        fts = _fts()

        n = await rebuild_fts_from_vdb(vdb, fts, ["S1"])

        assert n == 1
        written = [d for call in fts.index_batch.await_args_list for d in call.args[0]]
        assert [d["chunk_id"] for d in written] == ["c1"]
