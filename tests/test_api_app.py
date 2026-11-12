"""Tests for the FastAPI app (middleware, /health) and the health_check MCP tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from atlassian_rag_mcp.api.app import ApiKeyMiddleware, TrailingSlashMiddleware
from atlassian_rag_mcp.api.dependencies import get_ctx
from atlassian_rag_mcp.api.lifespan import ServerContext
from atlassian_rag_mcp.api.mcp_server import set_mcp_ctx
from atlassian_rag_mcp.api.routers.index import router
from atlassian_rag_mcp.api.tools.health import health_check
from atlassian_rag_mcp.config import settings


class FakeQdrantClient:
    """Minimal VectorDB double with a stable class name for db_name derivation."""

    def __init__(self):
        self._health = {"status": "healthy", "count": 1}

    async def health_check(self):
        if isinstance(self._health, Exception):
            raise self._health
        return self._health


def make_ctx() -> MagicMock:
    ctx = MagicMock(spec=ServerContext)
    db = FakeQdrantClient()
    ctx.vector_service = MagicMock()
    ctx.vector_service.vector_db = db
    ctx.confluence_client = AsyncMock()
    ctx.confluence_client.get_all_spaces = AsyncMock(
        return_value=[{"key": "SP", "name": "Space"}]
    )
    ctx.jira_client = MagicMock()
    ctx.jira_client.validate = MagicMock()
    ctx.jira_client.plugins = {}
    ctx.embedding = MagicMock()
    ctx.embedding.get_embedding_dimension = AsyncMock(return_value=4)
    ctx.embedding.model_name = "test-model"
    ctx.index_queue = AsyncMock()
    ctx.index_queue.get_status = AsyncMock(return_value={"queue_size": 0})
    return ctx


@pytest.fixture
def app():
    fastapi_app = FastAPI()
    fastapi_app.add_middleware(TrailingSlashMiddleware)
    fastapi_app.add_middleware(ApiKeyMiddleware)
    fastapi_app.include_router(router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_ctx] = lambda: make_ctx()

    @fastapi_app.get("/health")
    async def health():
        return {"status": "ok"}

    @fastapi_app.api_route("/mcp/", methods=["POST", "GET"])
    async def mcp_stub():
        return {"ok": True}

    return fastapi_app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestTrailingSlashMiddleware:
    def test_post_mcp_rewritten(self, client):
        resp = client.post("/mcp")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

    def test_get_mcp_rewritten(self, client):
        resp = client.get("/mcp")
        assert resp.status_code == 200

    def test_other_method_not_rewritten(self, client):
        resp = client.patch("/mcp")
        assert resp.status_code == 405


class TestApiKeyMiddleware:
    def test_missing_key_forbidden(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "secret")
        resp = client.get("/api/v1/index/status")
        assert resp.status_code == 403
        assert "X-API-Key" in resp.json()["error"]

    def test_wrong_key_forbidden(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "secret")
        resp = client.get("/api/v1/index/status", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 403

    def test_correct_key_allowed(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "secret")
        resp = client.get("/api/v1/index/status", headers={"X-API-Key": "secret"})
        assert resp.status_code == 200

    def test_key_not_required_when_unset(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_key", None)
        resp = client.get("/api/v1/index/status")
        assert resp.status_code == 200

    def test_mcp_path_not_protected(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "secret")
        resp = client.post("/mcp")
        assert resp.status_code == 200

    def test_health_path_not_protected(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "secret")
        resp = client.get("/health")
        assert resp.status_code == 200


class TestGetCtx:
    def test_reads_from_app_state(self):
        fastapi_app = FastAPI()
        expected = make_ctx()

        @fastapi_app.get("/probe")
        async def probe(ctx=Depends(get_ctx)):
            return {"is_ctx": ctx is expected}

        fastapi_app.state.ctx = expected
        resp = TestClient(fastapi_app).get("/probe")
        assert resp.status_code == 200
        assert resp.json() == {"is_ctx": True}


class TestHealthCheckTool:
    @pytest.fixture(autouse=True)
    def _ctx(self):
        set_mcp_ctx(None)
        yield
        set_mcp_ctx(None)

    async def test_all_healthy(self):
        ctx = make_ctx()
        set_mcp_ctx(ctx)
        result = await health_check()
        assert result.status == "healthy"
        assert result.services["fakeqdrant"].status == "healthy"
        assert result.services["confluence"].status == "healthy"
        assert result.services["jira"].status == "healthy"
        assert result.services["embedding"].status == "healthy"
        assert result.services["embedding"].dimension == 4
        assert result.services["embedding"].model == "test-model"
        assert result.services["jira_plugins"].status == "disabled"

    async def test_db_unhealthy_degrades(self):
        ctx = make_ctx()
        ctx.vector_service.vector_db._health = {"status": "unhealthy"}
        set_mcp_ctx(ctx)
        result = await health_check()
        assert result.status == "degraded"
        assert result.services["fakeqdrant"].status == "unhealthy"

    async def test_db_error_degrades(self):
        ctx = make_ctx()
        ctx.vector_service.vector_db._health = RuntimeError("boom")
        set_mcp_ctx(ctx)
        result = await health_check()
        assert result.status == "degraded"
        assert result.services["fakeqdrant"].status == "unhealthy"
        assert "boom" in result.services["fakeqdrant"].error

    async def test_confluence_none_degrades(self):
        ctx = make_ctx()
        ctx.confluence_client.get_all_spaces = AsyncMock(return_value=None)
        set_mcp_ctx(ctx)
        result = await health_check()
        assert result.status == "degraded"
        assert result.services["confluence"].status == "unhealthy"

    async def test_confluence_error_degrades(self):
        ctx = make_ctx()
        ctx.confluence_client.get_all_spaces = AsyncMock(
            side_effect=RuntimeError("conf down")
        )
        set_mcp_ctx(ctx)
        result = await health_check()
        assert result.status == "degraded"
        assert "conf down" in result.services["confluence"].error

    async def test_jira_error_degrades(self):
        ctx = make_ctx()
        ctx.jira_client.validate = MagicMock(side_effect=RuntimeError("jira down"))
        set_mcp_ctx(ctx)
        result = await health_check()
        assert result.status == "degraded"
        assert "jira down" in result.services["jira"].error

    async def test_plugin_healthy(self):
        ctx = make_ctx()
        plugin = MagicMock()
        plugin.validate = AsyncMock(return_value=True)
        ctx.jira_client.plugins = {"zephyr": plugin}
        set_mcp_ctx(ctx)
        result = await health_check()
        assert result.services["jira_plugin_zephyr"].status == "healthy"
        assert result.status == "healthy"

    async def test_plugin_not_available(self):
        ctx = make_ctx()
        plugin = MagicMock()
        plugin.validate = AsyncMock(return_value=False)
        ctx.jira_client.plugins = {"zephyr": plugin}
        set_mcp_ctx(ctx)
        result = await health_check()
        assert result.services["jira_plugin_zephyr"].status == "not_available"

    async def test_plugin_error_degrades(self):
        ctx = make_ctx()
        plugin = MagicMock()
        plugin.validate = AsyncMock(side_effect=RuntimeError("plugin boom"))
        ctx.jira_client.plugins = {"zephyr": plugin}
        set_mcp_ctx(ctx)
        result = await health_check()
        assert result.status == "degraded"
        assert "plugin boom" in result.services["jira_plugin_zephyr"].error

    async def test_embedding_error_degrades(self):
        ctx = make_ctx()
        ctx.embedding.get_embedding_dimension = AsyncMock(
            side_effect=RuntimeError("emb down")
        )
        set_mcp_ctx(ctx)
        result = await health_check()
        assert result.status == "degraded"
        assert "emb down" in result.services["embedding"].error

    async def test_no_ctx_raises(self):
        with pytest.raises(RuntimeError, match="ServerContext not initialized"):
            await health_check()


class TestAppLifespan:
    def test_startup_and_shutdown(self, monkeypatch):
        import asyncio
        import importlib

        import atlassian_rag_mcp.api.app as app_module
        from atlassian_rag_mcp.api.app import app

        worker_module = importlib.import_module(
            "atlassian_rag_mcp.confluence.services.indexer.worker"
        )

        ctx = make_ctx()
        ctx.aclose = AsyncMock()
        started = []

        async def fake_init():
            return ctx

        async def fake_worker(**kwargs):
            started.append(kwargs)
            await asyncio.Event().wait()

        monkeypatch.setattr(app_module, "_init", fake_init)
        monkeypatch.setattr(worker_module, "worker", fake_worker)
        set_mcp_calls: list = []
        real_set_mcp_ctx = app_module.set_mcp_ctx

        def spy_set_mcp_ctx(ctx):
            set_mcp_calls.append(ctx)
            real_set_mcp_ctx(ctx)

        monkeypatch.setattr(app_module, "set_mcp_ctx", spy_set_mcp_ctx)

        with TestClient(app) as client:
            assert app.state.ctx is ctx
            assert set_mcp_calls[0] is ctx
            assert len(started) == max(1, settings.embedding_max_workers)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

        ctx.aclose.assert_awaited_once()
        assert set_mcp_calls[-1] is None
