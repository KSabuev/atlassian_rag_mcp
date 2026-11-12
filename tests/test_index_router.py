"""Tests for REST API index endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from atlassian_rag_mcp.api.dependencies import get_ctx
from atlassian_rag_mcp.api.lifespan import ServerContext
from atlassian_rag_mcp.api.routers.index import router


@pytest.fixture
def mock_ctx():
    ctx = MagicMock(spec=ServerContext)
    ctx.vector_db = MagicMock()
    ctx.vector_service = MagicMock()
    ctx.vector_service.delete_page = AsyncMock(return_value=True)
    ctx.vector_service.get_chunks_by_filter = AsyncMock(
        return_value=[{"chunk_id": "c1"}]
    )
    ctx.confluence_client = AsyncMock()
    ctx.search_deps = MagicMock()
    ctx.search_deps.confluence_space_key = "SP"
    ctx.index_queue = MagicMock()
    return ctx


@pytest.fixture
def client(mock_ctx):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_ctx] = lambda: mock_ctx
    return TestClient(app)


class TestDeletePage:
    def test_delete_calls_vector_service_delete_page(self, client, mock_ctx):
        resp = client.delete("/api/v1/index/page/p42")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["deleted"] is True
        mock_ctx.vector_service.delete_page.assert_awaited_once_with("p42")

    def test_delete_page_not_found(self, client, mock_ctx):
        mock_ctx.vector_service.get_chunks_by_filter = AsyncMock(return_value=[])
        resp = client.delete("/api/v1/index/page/p42")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["deleted"] is False
        mock_ctx.vector_service.delete_page.assert_not_called()

    def test_delete_page_failure_returns_error(self, client, mock_ctx):
        mock_ctx.vector_service.delete_page = AsyncMock(return_value=False)
        resp = client.delete("/api/v1/index/page/p42")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "error"
        assert data["deleted"] is False


_RESULT = {
    "status": "ok", "pages_scanned": 3, "pages_queued": 2,
    "pages_deleted": 0, "fts_inconsistent": 0,
}


class TestSync:
    def test_sync_passes_body_to_reconcile_fn(self, client, mock_ctx):
        mock_ctx.reconcile_fn = AsyncMock(return_value=dict(_RESULT))
        resp = client.post("/api/v1/index/sync", json={
            "force": True, "spaces": [" SP ", "MMS"], "delete_stale": True, "deep": True,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        mock_ctx.reconcile_fn.assert_awaited_once_with(
            space_key=["SP", "MMS"], deep=True, force=True, delete_stale=True,
        )

    def test_sync_default_body(self, client, mock_ctx):
        mock_ctx.reconcile_fn = AsyncMock(return_value=dict(_RESULT))
        resp = client.post("/api/v1/index/sync", json={})
        assert resp.status_code == 200
        mock_ctx.reconcile_fn.assert_awaited_once_with(
            space_key=None, deep=False, force=False, delete_stale=False,
        )

    def test_sync_omitted_or_empty_spaces_maps_to_none(self, client, mock_ctx):
        mock_ctx.reconcile_fn = AsyncMock(return_value=dict(_RESULT))
        client.post("/api/v1/index/sync", json={"spaces": []})
        client.post("/api/v1/index/sync", json={"force": True})
        for call in mock_ctx.reconcile_fn.await_args_list:
            assert call.kwargs["space_key"] is None

    def test_sync_returns_reconcile_result(self, client, mock_ctx):
        mock_ctx.reconcile_fn = AsyncMock(return_value=dict(_RESULT, pages_deleted=7))
        resp = client.post("/api/v1/index/sync", json={"delete_stale": True})
        data = resp.json()
        assert data["pages_deleted"] == 7
        assert data["pages_scanned"] == 3

    def test_sync_error(self, client, mock_ctx):
        mock_ctx.reconcile_fn = AsyncMock(side_effect=RuntimeError("cql down"))
        resp = client.post("/api/v1/index/sync", json={})
        data = resp.json()
        assert data["status"] == "error"
        assert "cql down" in data["error"]

    def test_reconcile_endpoint_removed(self, client, mock_ctx):
        resp = client.post("/api/v1/index/reconcile", params={"deep": True})
        assert resp.status_code == 404

    def test_fts_rebuild_endpoint_removed(self, client, mock_ctx):
        resp = client.post("/api/v1/index/fts-rebuild")
        assert resp.status_code == 404


class TestSyncFtsRebuild:
    def test_fts_rebuild_calls_helper_with_spaces(self, client, mock_ctx):
        mock_ctx.reconcile_fn = AsyncMock(return_value=dict(_RESULT))
        with patch("atlassian_rag_mcp.api.routers.index.rebuild_fts_from_vdb",
                   new=AsyncMock(return_value=5)) as helper:
            resp = client.post("/api/v1/index/sync", json={
                "fts_rebuild": True, "spaces": ["S1", "S2"],
            })
        assert resp.status_code == 200
        data = resp.json()
        assert data["fts_rebuilt"] == 5
        assert data["pages_scanned"] == 3
        helper.assert_awaited_once_with(
            mock_ctx.vector_db, mock_ctx.search_deps.fts_index, ["S1", "S2"],
        )

    def test_fts_rebuild_default_spaces_from_config(self, client, mock_ctx):
        mock_ctx.reconcile_fn = AsyncMock(return_value=dict(_RESULT))
        mock_ctx.search_deps.confluence_space_key = "SP1, SP2"
        with patch("atlassian_rag_mcp.api.routers.index.rebuild_fts_from_vdb",
                   new=AsyncMock(return_value=1)) as helper:
            client.post("/api/v1/index/sync", json={"fts_rebuild": True})
        helper.assert_awaited_once()
        assert helper.await_args.args[2] == ["SP1", "SP2"]

    def test_sync_without_fts_rebuild_does_not_rebuild(self, client, mock_ctx):
        mock_ctx.reconcile_fn = AsyncMock(return_value=dict(_RESULT))
        with patch("atlassian_rag_mcp.api.routers.index.rebuild_fts_from_vdb",
                   new=AsyncMock(return_value=0)) as helper:
            resp = client.post("/api/v1/index/sync", json={})
        helper.assert_not_called()
        assert "fts_rebuilt" not in resp.json()

    def test_fts_rebuild_error(self, client, mock_ctx):
        mock_ctx.reconcile_fn = AsyncMock(return_value=dict(_RESULT))
        with patch("atlassian_rag_mcp.api.routers.index.rebuild_fts_from_vdb",
                   new=AsyncMock(side_effect=RuntimeError("no spaces configured"))):
            resp = client.post("/api/v1/index/sync", json={"fts_rebuild": True})
        data = resp.json()
        assert data["status"] == "error"
        assert "no spaces configured" in data["error"]
