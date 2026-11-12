"""Unit tests for the Confluence/VectorDB reconciler (mocked services)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from atlassian_rag_mcp.confluence.models import CqlResult, PageContent
from atlassian_rag_mcp.confluence.services.reconciler import _normalize_space_keys, reconcile


def _page(pid: str, version: int) -> dict:
    return {"content": {"id": pid, "version": {"number": version}}}


def _build(
    cql_batches: list[list[dict]] | None = None,
    vdb_versions: dict[str, int] | None = None,
    delete_ok: bool = True,
    delete_raises: bool = False,
    chunks: dict[str, list] | None = None,
    page_versions: dict[str, int] | None = None,
    page_errors: dict[str, str] | None = None,
    vdb_spaces: dict[str, str] | None = None,
):
    cql_batches = list(cql_batches) if cql_batches is not None else [[]]
    if cql_batches and cql_batches[-1]:
        cql_batches.append([])
    vdb_versions = vdb_versions or {}

    confluence = MagicMock()
    confluence.cql = AsyncMock(
        side_effect=[CqlResult(results=batch) for batch in cql_batches]
    )

    def _get_page(pid, expand=None):
        version = (page_versions or {}).get(pid, 0)
        error = (page_errors or {}).get(pid)
        return PageContent(page_id=pid, version=version, error=error)

    confluence.get_page = AsyncMock(side_effect=_get_page)

    vector_service = MagicMock()
    vector_service.vector_db = MagicMock()
    vector_service.vector_db.get_page_versions_map = AsyncMock(return_value=vdb_versions)

    def _scroll(filter_condition=None, limit=10000, offset=None, with_vectors=False):
        points = [
            {"id": i, "payload": {"page_id": pid, "space_key": sk}}
            for i, (pid, sk) in enumerate((vdb_spaces or {}).items())
        ]
        return points, None

    vector_service.vector_db.scroll_points = AsyncMock(side_effect=_scroll)
    if delete_raises:
        vector_service.delete_page = AsyncMock(side_effect=RuntimeError("vdb down"))
    else:
        vector_service.delete_page = AsyncMock(return_value=delete_ok)

    def _get_chunks(pid, limit=1):
        return (chunks or {}).get(pid, [])

    vector_service.get_chunks_by_filter = AsyncMock(side_effect=lambda page_id, limit=1: _get_chunks(page_id))

    queue = MagicMock()
    queue.enqueue = AsyncMock()

    return confluence, vector_service, queue


class TestReconcileBasic:
    async def test_empty_space(self):
        confluence, vs, queue = _build(cql_batches=[[]])
        result = await reconcile(confluence, vs, queue, "SP")
        assert result == {
            "status": "ok",
            "pages_scanned": 0,
            "pages_queued": 0,
            "pages_deleted": 0,
            "fts_inconsistent": 0,
        }
        queue.enqueue.assert_not_awaited()

    async def test_missing_page_queued(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1), _page("p2", 2)]],
            vdb_versions={"p2": 2},
        )
        result = await reconcile(confluence, vs, queue, "SP")
        assert result["pages_scanned"] == 2
        assert result["pages_queued"] == 1
        queue.enqueue.assert_awaited_once_with("p1")

    async def test_stale_version_queued(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 3)]],
            vdb_versions={"p1": 1},
        )
        result = await reconcile(confluence, vs, queue, "SP")
        assert result["pages_queued"] == 1
        queue.enqueue.assert_awaited_once_with("p1")

    async def test_matching_version_not_queued(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 2)]],
            vdb_versions={"p1": 2},
        )
        result = await reconcile(confluence, vs, queue, "SP")
        assert result["pages_queued"] == 0
        queue.enqueue.assert_not_awaited()

    async def test_zero_live_version_not_queued(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 0)]],
            vdb_versions={"p1": 5},
        )
        result = await reconcile(confluence, vs, queue, "SP")
        assert result["pages_queued"] == 0

    async def test_page_without_id_skipped(self):
        confluence, vs, queue = _build(
            cql_batches=[[{"content": {"id": ""}}, _page("p1", 1)]],
            vdb_versions={},
        )
        result = await reconcile(confluence, vs, queue, "SP")
        assert result["pages_scanned"] == 1
        assert result["pages_queued"] == 1

    async def test_single_space_cql(self):
        confluence, vs, queue = _build(cql_batches=[[]])
        await reconcile(confluence, vs, queue, "SP")
        call_cql = confluence.cql.call_args[1]["cql"]
        assert call_cql == 'type=page AND space = "SP"'

    async def test_multi_space_list_cql(self):
        confluence, vs, queue = _build(cql_batches=[[]])
        await reconcile(confluence, vs, queue, ["SP", "EEE"])
        call_cql = confluence.cql.call_args[1]["cql"]
        assert call_cql == 'type=page AND space IN ("SP", "EEE")'

    async def test_multi_space_comma_string_cql(self):
        confluence, vs, queue = _build(cql_batches=[[]])
        await reconcile(confluence, vs, queue, "SP, EEE")
        call_cql = confluence.cql.call_args[1]["cql"]
        assert call_cql == 'type=page AND space IN ("SP", "EEE")'

    async def test_no_space_scans_all(self):
        confluence, vs, queue = _build(cql_batches=[[]])
        await reconcile(confluence, vs, queue, "")
        call_cql = confluence.cql.call_args[1]["cql"]
        assert call_cql == "type=page"


class TestNormalizeSpaceKeys:
    def test_string(self):
        assert _normalize_space_keys("SP") == ["SP"]

    def test_comma_string(self):
        assert _normalize_space_keys("SP, EEE ,") == ["SP", "EEE"]

    def test_list(self):
        assert _normalize_space_keys(["SP", "EEE"]) == ["SP", "EEE"]

    def test_empty_string(self):
        assert _normalize_space_keys("") == []

    def test_empty_list(self):
        assert _normalize_space_keys([]) == []

    def test_none(self):
        assert _normalize_space_keys(None) == []


class TestReconcilePagination:
    async def test_multiple_batches_until_empty(self):
        confluence, vs, queue = _build(
            cql_batches=[
                [_page("p1", 1), _page("p2", 1)],
                [_page("p3", 1)],
                [],
            ],
            vdb_versions={},
        )
        result = await reconcile(confluence, vs, queue, "SP", cql_batch_size=2)
        assert result["pages_scanned"] == 3
        assert result["pages_queued"] == 3
        starts = [c.kwargs["start"] for c in confluence.cql.await_args_list]
        assert starts == [0, 2, 4]
        limits = [c.kwargs["limit"] for c in confluence.cql.await_args_list]
        assert all(limit == 2 for limit in limits)

    async def test_dedupes_pages_across_batches(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)], [_page("p1", 1)], []],
            vdb_versions={},
        )
        result = await reconcile(confluence, vs, queue, "SP")
        assert result["pages_queued"] == 1
        assert queue.enqueue.await_count == 2


class TestReconcileOrphans:
    async def test_orphan_in_vdb_deleted(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1, "p9": 4},
            vdb_spaces={"p9": "SP"},
        )
        result = await reconcile(confluence, vs, queue, "SP")
        assert result["pages_deleted"] == 1
        vs.delete_page.assert_awaited_once_with("p9")

    async def test_orphan_delete_failure_not_counted(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1, "p9": 4},
            delete_raises=True,
            vdb_spaces={"p9": "SP"},
        )
        result = await reconcile(confluence, vs, queue, "SP")
        assert result["pages_deleted"] == 0

    async def test_orphan_delete_false_not_counted(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1, "p9": 4},
            delete_ok=False,
            vdb_spaces={"p9": "SP"},
        )
        result = await reconcile(confluence, vs, queue, "SP")
        assert result["pages_deleted"] == 0

    async def test_orphan_from_other_space_not_deleted(self):
        """A restricted scan must not delete pages from unscanned spaces."""
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1, "p9": 4},
            vdb_spaces={"p9": "OTHER"},
        )
        result = await reconcile(confluence, vs, queue, "SP")
        assert result["pages_deleted"] == 0
        vs.delete_page.assert_not_awaited()

    async def test_orphan_unknown_space_not_deleted(self):
        """A page with no stored space_key is never deleted by a restricted scan."""
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1, "p9": 4},
            vdb_spaces={},
        )
        result = await reconcile(confluence, vs, queue, "SP")
        assert result["pages_deleted"] == 0
        vs.delete_page.assert_not_awaited()

    async def test_unrestricted_scan_deletes_orphan_without_scroll(self):
        """With no space restriction every missing page is an orphan (no scroll)."""
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1, "p9": 4},
        )
        result = await reconcile(confluence, vs, queue, "")
        assert result["pages_deleted"] == 1
        vs.delete_page.assert_awaited_once_with("p9")
        vs.vector_db.scroll_points.assert_not_awaited()

    async def test_multi_space_orphan_deleted_when_in_scanned_space(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1, "p9": 4, "p8": 4},
            vdb_spaces={"p9": "EEE", "p8": "ZZZ"},
        )
        result = await reconcile(confluence, vs, queue, "SP, EEE")
        assert result["pages_deleted"] == 1
        vs.delete_page.assert_awaited_once_with("p9")


class TestReconcileForce:
    async def test_force_enqueues_all_pages(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 2), _page("p2", 2)]],
            vdb_versions={"p1": 2, "p2": 2},
        )
        result = await reconcile(confluence, vs, queue, "SP", force=True)
        assert result["pages_queued"] == 2
        enqueued = [c.args[0] for c in queue.enqueue.await_args_list]
        assert sorted(enqueued) == ["p1", "p2"]

    async def test_force_skips_deep_check(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1},
            vdb_spaces={"p1": "SP"},
            chunks={"p1": []},
        )
        result = await reconcile(confluence, vs, queue, "SP", deep=True, force=True)
        assert result["fts_inconsistent"] == 0
        confluence.get_page.assert_not_awaited()


class TestReconcileDeleteStale:
    async def test_delete_stale_false_keeps_orphans(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1, "p9": 4},
            vdb_spaces={"p9": "SP"},
        )
        result = await reconcile(confluence, vs, queue, "SP", delete_stale=False)
        assert result["pages_deleted"] == 0
        vs.delete_page.assert_not_awaited()


class TestReconcileDeep:
    async def test_deep_missing_chunks_inconsistent(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1},
            chunks={},
            vdb_spaces={"p1": "SP"},
        )
        result = await reconcile(confluence, vs, queue, "SP", deep=True)
        assert result["fts_inconsistent"] == 1
        queue.enqueue.assert_awaited()

    async def test_deep_version_mismatch_inconsistent(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1},
            chunks={"p1": [{"chunk_id": "c"}]},
            page_versions={"p1": 2},
            vdb_spaces={"p1": "SP"},
        )
        result = await reconcile(confluence, vs, queue, "SP", deep=True)
        assert result["fts_inconsistent"] == 1

    async def test_deep_consistent_page_clean(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1},
            chunks={"p1": [{"chunk_id": "c"}]},
            page_versions={"p1": 1},
            vdb_spaces={"p1": "SP"},
        )
        result = await reconcile(confluence, vs, queue, "SP", deep=True)
        assert result["fts_inconsistent"] == 0
        queue.enqueue.assert_not_awaited()

    async def test_deep_confluence_page_error_skipped(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1},
            chunks={"p1": [{"chunk_id": "c"}]},
            page_versions={"p1": 9},
            page_errors={"p1": "not found"},
            vdb_spaces={"p1": "SP"},
        )
        result = await reconcile(confluence, vs, queue, "SP", deep=True)
        assert result["fts_inconsistent"] == 0

    async def test_deep_zero_confluence_version_not_counted(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1},
            chunks={"p1": [{"chunk_id": "c"}]},
            page_versions={"p1": 0},
            vdb_spaces={"p1": "SP"},
        )
        result = await reconcile(confluence, vs, queue, "SP", deep=True)
        assert result["fts_inconsistent"] == 0

    async def test_deep_exception_swallowed(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1},
            vdb_spaces={"p1": "SP"},
        )
        vs.get_chunks_by_filter = AsyncMock(side_effect=RuntimeError("vdb boom"))
        result = await reconcile(confluence, vs, queue, "SP", deep=True)
        assert result["status"] == "ok"
        assert result["fts_inconsistent"] == 0

    async def test_deep_respects_space_scope(self):
        """A restricted deep check must not touch pages from unscanned spaces."""
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1, "p9": 4},
            chunks={"p1": [{"chunk_id": "c"}], "p9": [{"chunk_id": "c9"}]},
            page_versions={"p1": 1, "p9": 9},
            vdb_spaces={"p1": "SP", "p9": "OTHER"},
        )
        result = await reconcile(confluence, vs, queue, "SP", deep=True)
        checked = [c.args[0] for c in confluence.get_page.await_args_list]
        assert checked == ["p1"]
        assert result["fts_inconsistent"] == 0
        queue.enqueue.assert_not_awaited()

    async def test_deep_unrestricted_checks_all_vdb_pages(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1, "p9": 4},
            chunks={"p1": [{"chunk_id": "c"}], "p9": [{"chunk_id": "c9"}]},
            page_versions={"p1": 1, "p9": 4},
            vdb_spaces={"p9": "OTHER"},
        )
        result = await reconcile(confluence, vs, queue, "", deep=True)
        checked = {c.args[0] for c in confluence.get_page.await_args_list}
        assert checked == {"p1", "p9"}
        assert result["fts_inconsistent"] == 0
        vs.vector_db.scroll_points.assert_not_awaited()

    async def test_not_deep_skips_fts_check(self):
        confluence, vs, queue = _build(
            cql_batches=[[_page("p1", 1)]],
            vdb_versions={"p1": 1},
        )
        result = await reconcile(confluence, vs, queue, "SP", deep=False)
        assert result["fts_inconsistent"] == 0
        vs.get_chunks_by_filter.assert_not_awaited()
