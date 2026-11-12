"""Lifespan startup: _init() — factory mocks + validator tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atlassian_rag_mcp.api.lifespan import ServerContext, _init, reset_ctx
from atlassian_rag_mcp.exceptions import AuthenticationError, BaseError


def _build_ctx(monkeypatch, **overrides):
    """Common helper: build a set of mocks that _init() can use."""

    m_confluence = overrides.get("m_confluence", MagicMock())
    m_confluence.validate = overrides.get("confluence_validate", MagicMock())
    m_jira = overrides.get("m_jira", MagicMock())
    m_jira.validate = overrides.get("jira_validate", MagicMock())
    m_vector_db = overrides.get("m_vector_db", AsyncMock())
    m_vector_db.validate = overrides.get("vector_db_validate", AsyncMock())
    m_vector_db._ensure_collection = overrides.get("ensure_collection", AsyncMock())
    m_embedding = overrides.get("m_embedding", AsyncMock())
    m_embedding.validate = overrides.get("embedding_validate", AsyncMock())
    m_fts = overrides.get("m_fts", AsyncMock())
    m_fts.validate = overrides.get("fts_validate", AsyncMock())
    m_vs = overrides.get("m_vs", AsyncMock())
    m_vs._ensure_collection = AsyncMock()

    monkeypatch.setattr("atlassian_rag_mcp.api.lifespan.create_confluence_client", lambda *a, **kw: m_confluence)
    monkeypatch.setattr("atlassian_rag_mcp.api.lifespan.create_jira_client", lambda *a, **kw: m_jira)
    monkeypatch.setattr(
        "atlassian_rag_mcp.api.lifespan.create_vector_db",
        AsyncMock(return_value=m_vector_db),
    )
    monkeypatch.setattr("atlassian_rag_mcp.api.lifespan.build_embedding_client", lambda s=None: m_embedding)
    monkeypatch.setattr(
        "atlassian_rag_mcp.api.lifespan.create_fts_index",
        AsyncMock(return_value=m_fts),
    )
    monkeypatch.setattr("atlassian_rag_mcp.api.lifespan.VectorService", lambda *a, **kw: m_vs)

    mock_iq = overrides.get("m_iq", MagicMock())
    monkeypatch.setattr("atlassian_rag_mcp.api.lifespan.IndexQueue", lambda *a, **kw: mock_iq)

    async def mock_inject_conf(_, **kw):
        pass

    async def mock_inject_jira(_, **kw):
        pass

    monkeypatch.setattr("atlassian_rag_mcp.api.lifespan._inject_confluence_metadata", mock_inject_conf)
    monkeypatch.setattr("atlassian_rag_mcp.api.lifespan._inject_jira_metadata", mock_inject_jira)

    return {
        "m_confluence": m_confluence,
        "m_jira": m_jira,
        "m_vector_db": m_vector_db,
        "m_embedding": m_embedding,
        "m_fts": m_fts,
        "m_vs": m_vs,
        "m_iq": mock_iq,
    }


@pytest.fixture(autouse=True)
def clear_ctx():
    reset_ctx()
    yield
    reset_ctx()


class TestInitSingleton:
    async def test_first_call_creates(self, monkeypatch):
        m = _build_ctx(monkeypatch)

        ctx = await _init()
        assert isinstance(ctx, ServerContext)
        assert ctx.confluence_client is m["m_confluence"]
        assert ctx.jira_client is m["m_jira"]
        assert ctx.vector_db is m["m_vector_db"]
        assert ctx.embedding is m["m_embedding"]
        assert ctx.index_queue is m["m_iq"]

    async def test_second_call_returns_cached(self, monkeypatch):
        _build_ctx(monkeypatch)

        ctx1 = await _init()
        ctx2 = await _init()
        assert ctx1 is ctx2
        from atlassian_rag_mcp.api import lifespan

        assert lifespan._ctx is ctx1


class TestValidators:
    async def test_all_validators_called(self, monkeypatch):
        m = _build_ctx(monkeypatch)
        await _init()

        m["m_confluence"].validate.assert_called_once()
        m["m_jira"].validate.assert_called_once()
        m["m_embedding"].validate.assert_awaited_once()
        m["m_vector_db"].validate.assert_awaited_once()
        m["m_fts"].validate.assert_awaited_once()

    async def test_sync_validators_run_in_thread(self, monkeypatch):
        m = _build_ctx(monkeypatch)
        m["m_confluence"].validate = MagicMock()
        m["m_jira"].validate = MagicMock()
        await _init()
        m["m_confluence"].validate.assert_called_once()
        m["m_jira"].validate.assert_called_once()


class TestValidatorErrors:
    async def test_confluence_auth_error_graceful(self, monkeypatch):
        bad = MagicMock()
        bad.validate = MagicMock(side_effect=AuthenticationError("bad creds"))
        _build_ctx(monkeypatch, m_confluence=bad)
        ctx = await _init()
        assert ctx is not None

    async def test_embedding_backend_error_graceful(self, monkeypatch):
        bad = AsyncMock()
        bad.validate = AsyncMock(side_effect=BaseError("down"))
        _build_ctx(monkeypatch, m_embedding=bad)
        ctx = await _init()
        assert ctx is not None

    async def test_fts_auth_error_graceful(self, monkeypatch):
        bad = AsyncMock()
        bad.validate = AsyncMock(side_effect=AuthenticationError("no access"))
        _build_ctx(monkeypatch, m_fts=bad)
        ctx = await _init()
        assert ctx is not None


class TestSearchDeps:
    async def test_references_same_objects(self, monkeypatch):
        m = _build_ctx(monkeypatch)
        ctx = await _init()

        assert ctx.search_deps.vector_db is ctx.vector_db
        assert ctx.search_deps.embedding_client is ctx.embedding
        assert ctx.search_deps.confluence_client is ctx.confluence_client
        assert ctx.search_deps.fts_index is m["m_fts"]


class TestBuildEmbeddingClient:
    def test_uses_settings(self, monkeypatch):
        from atlassian_rag_mcp.api.lifespan import build_embedding_client
        from atlassian_rag_mcp.config import settings

        seen: dict = {}

        def fake_factory(**kw):
            seen.update(kw)
            return MagicMock()

        monkeypatch.setattr("atlassian_rag_mcp.api.lifespan.create_embedding_client", fake_factory)

        client = build_embedding_client(settings)
        assert client is not None
        assert "max_token_size" not in seen
        assert seen["model"] == settings.embedding_model
        assert seen["dim"] == settings.embedding_dim


class TestAwaitWithTimeout:
    async def test_returns_value(self):
        from atlassian_rag_mcp.api.lifespan import _await_with_timeout

        async def coro():
            return 42

        assert await _await_with_timeout(coro(), timeout=5) == 42

    async def test_timeout_raises_backend_error(self):
        import asyncio

        from atlassian_rag_mcp.api.lifespan import _await_with_timeout

        async def slow():
            await asyncio.sleep(10)

        with pytest.raises(BaseError, match="Timed out"):
            await _await_with_timeout(slow(), timeout=0.01)


class TestApplyPreambleToDescriptions:
    @staticmethod
    def _register_tool(monkeypatch, name, doc):
        import atlassian_rag_mcp.api.tools as tools

        def tool():
            pass

        tool.__doc__ = doc
        monkeypatch.setattr(tools, name, tool, raising=False)
        return tool

    def test_prepends_preamble_and_returns_updated(self, monkeypatch):
        from atlassian_rag_mcp.api.lifespan import _apply_preamble_to_descriptions

        a = self._register_tool(monkeypatch, "tg_tool_a", "First tool description.")
        self._register_tool(monkeypatch, "tg_tool_b", "Second tool description.")

        updated = _apply_preamble_to_descriptions(("tg_tool_a", "tg_tool_b"), "PREAMBLE\n")
        assert updated == ["tg_tool_a", "tg_tool_b"]
        assert a.__doc__ == "PREAMBLE\nFirst tool description."

    def test_unknown_tool_name_skipped(self, monkeypatch):
        from atlassian_rag_mcp.api.lifespan import _apply_preamble_to_descriptions

        a = self._register_tool(monkeypatch, "tg_tool_a", "First tool description.")

        updated = _apply_preamble_to_descriptions(("tg_tool_a", "does_not_exist"), "PREAMBLE\n")
        assert updated == ["tg_tool_a"]
        assert "PREAMBLE" in a.__doc__

    def test_none_docstring_handled(self, monkeypatch):
        from atlassian_rag_mcp.api.lifespan import _apply_preamble_to_descriptions

        a = self._register_tool(monkeypatch, "tg_tool_nodoc", None)

        updated = _apply_preamble_to_descriptions(("tg_tool_nodoc",), "PREAMBLE\n")
        assert updated == ["tg_tool_nodoc"]
        assert a.__doc__ == "PREAMBLE\n"

    def test_registered_mcp_tool_description_updated(self, monkeypatch):
        import atlassian_rag_mcp.api.tools as tools
        from atlassian_rag_mcp.api.lifespan import _apply_preamble_to_descriptions
        from atlassian_rag_mcp.api.mcp_server import mcp as mcp_server

        def tool():
            pass

        tool.__doc__ = "Original tool description."
        mcp_server.add_tool(tool, name="tg_tool_registered")
        monkeypatch.setattr(tools, "tg_tool_registered", tool, raising=False)
        try:
            updated = _apply_preamble_to_descriptions(("tg_tool_registered",), "PREAMBLE\n")
            assert updated == ["tg_tool_registered"]
            registered = mcp_server._tool_manager.get_tool("tg_tool_registered")
            assert registered is not None
            assert registered.description == "PREAMBLE\nOriginal tool description."
            assert tool.__doc__ == "PREAMBLE\nOriginal tool description."
        finally:
            mcp_server.remove_tool("tg_tool_registered")


class TestInjectJiraMetadata:
    @pytest.fixture(autouse=True)
    def _reset_value_maps(self):
        from atlassian_rag_mcp.jira.methods import reset_value_maps

        reset_value_maps()
        yield
        reset_value_maps()

    @staticmethod
    def _jira_client(monkeypatch):
        from atlassian_rag_mcp.config import settings
        from atlassian_rag_mcp.jira.models import ListResult

        monkeypatch.setattr(settings, "jira_project_key", "PPP,MM,TTT")
        client = MagicMock()
        types_by_key = {
            "PPP": [{"name": "Задача", "id": "10002"}],
            "MM": [{"name": "Задача", "id": "10100"}],
            "TTT": [{"name": "Ошибка", "id": "10004"}],
        }
        statuses_by_key = {
            "PPP": [{"name": "Сделать", "id": "10000"}],
            "MM": [{"name": "Закрыт", "id": "6"}],
            "TTT": [{"name": "Решенные", "id": "5"}],
        }
        client.issues.get_issue_types = MagicMock(
            side_effect=lambda key: ListResult(items=types_by_key.get(key, []), total=0))
        client.issues.get_project_statuses = MagicMock(
            side_effect=lambda key: ListResult(items=statuses_by_key.get(key, []), total=0))
        client.issues.get_priorities = MagicMock(
            return_value=ListResult(items=[{"name": "Высокий", "id": "2"}], total=1))
        client.issues.get_resolutions = MagicMock(
            return_value=ListResult(items=[{"name": "Готово", "id": "10400"}], total=1))
        return client

    async def test_injects_lines_for_all_configured_projects(self, monkeypatch, caplog):
        import logging

        import atlassian_rag_mcp.api.tools as tools
        from atlassian_rag_mcp.api.lifespan import _inject_jira_metadata
        from atlassian_rag_mcp.api.mcp_server import mcp as mcp_server

        client = self._jira_client(monkeypatch)
        func = tools.jira_execute_method
        tool = mcp_server._tool_manager.get_tool("jira_execute_method")
        original_doc = func.__doc__
        original_desc = tool.description if tool is not None else None

        doc = ""
        try:
            with caplog.at_level(logging.INFO, logger="atlassian_rag_mcp"):
                await _inject_jira_metadata(client, project_keys=["PPP", "MM", "TTT"])
            doc = func.__doc__ or ""
        finally:
            func.__doc__ = original_doc
            if tool is not None:
                tool.description = original_desc
        assert "issue types (PPP): Задача (10002)" in doc
        assert "issue types (MM): Задача (10100)" in doc
        assert "issue types (TTT): Ошибка (10004)" in doc
        assert "statuses (PPP): Сделать (10000)" in doc
        assert "statuses (MM): Закрыт (6)" in doc
        assert "statuses (TTT): Решенные (5)" in doc
        assert "priorities: Высокий (2)" in doc
        assert "resolutions: Готово (10400)" in doc
        assert "(use numeric ID in JQL)" in doc
        assert any("Injected Jira metadata (8 sections)" in r.message for r in caplog.records)
