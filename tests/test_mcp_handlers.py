"""MCP context and handlers: set_mcp_ctx, _get_svc, reset_ctx — isolated unit tests."""

from __future__ import annotations

import pytest

from atlassian_rag_mcp.api import mcp_server as mcp_module
from atlassian_rag_mcp.api.lifespan import ServerContext, reset_ctx
from atlassian_rag_mcp.api.mcp_server import _get_svc, set_mcp_ctx


class TestGetSvc:
    def test_raises_when_not_initialized(self):
        set_mcp_ctx(None)
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                _get_svc(None)
        finally:
            set_mcp_ctx(None)

    def test_returns_ctx_when_set(self):
        mock_ctx = object.__new__(ServerContext)
        mock_ctx.confluence_client = "conf"
        mock_ctx.jira_client = object()
        mock_ctx.vector_db = object()
        mock_ctx.vector_service = object()
        mock_ctx.embedding = object()
        mock_ctx.index_queue = object()
        mock_ctx.search_deps = object()
        set_mcp_ctx(mock_ctx)
        try:
            result = _get_svc(None)
            assert result.confluence_client == "conf"
        finally:
            set_mcp_ctx(None)


class TestResetCtx:
    def test_reset_ctx_clears_lifespan(self):
        from atlassian_rag_mcp.api import lifespan

        original = lifespan._ctx
        try:
            mock_ctx = object.__new__(ServerContext)
            mock_ctx.confluence_client = mock_ctx.jira_client = object()
            mock_ctx.vector_db = mock_ctx.vector_service = object()
            mock_ctx.embedding = mock_ctx.index_queue = object()
            mock_ctx.search_deps = object()
            lifespan._ctx = mock_ctx
            reset_ctx()
            assert lifespan._ctx is None
        finally:
            lifespan._ctx = original

    def test_mcp_svc_independent_of_lifespan_reset(self):
        from atlassian_rag_mcp.api import lifespan

        original_mcp = mcp_module._mcp_ctx
        original_ls = lifespan._ctx
        try:
            mcp_ctx = object.__new__(ServerContext)
            mcp_ctx.confluence_client = "mcp"
            mcp_ctx.jira_client = mcp_ctx.vector_db = mcp_ctx.vector_service = object()
            mcp_ctx.embedding = mcp_ctx.index_queue = mcp_ctx.search_deps = object()
            set_mcp_ctx(mcp_ctx)

            ls_ctx = object.__new__(ServerContext)
            ls_ctx.confluence_client = "lifespan"
            ls_ctx.jira_client = ls_ctx.vector_db = ls_ctx.vector_service = object()
            ls_ctx.embedding = ls_ctx.index_queue = ls_ctx.search_deps = object()
            lifespan._ctx = ls_ctx

            reset_ctx()
            assert lifespan._ctx is None
            assert mcp_module._mcp_ctx is mcp_ctx
        finally:
            set_mcp_ctx(original_mcp)
            lifespan._ctx = original_ls


class TestSearchDeps:
    def test_search_deps_references_same_objects(self):
        original = mcp_module._mcp_ctx
        try:
            mock_db = object()
            mock_embed = object()
            from atlassian_rag_mcp.confluence.search import SearchDeps
            mock_deps = SearchDeps(
                vector_db=mock_db,
                embedding_client=mock_embed,
                confluence_client=object(),
                fts_index=object(),
            )
            mock_ctx = object.__new__(ServerContext)
            mock_ctx.confluence_client = mock_ctx.jira_client = object()
            mock_ctx.vector_db = mock_db
            mock_ctx.vector_service = object()
            mock_ctx.embedding = mock_embed
            mock_ctx.index_queue = object()
            mock_ctx.search_deps = mock_deps
            set_mcp_ctx(mock_ctx)

            result = _get_svc(None)
            assert result.search_deps.vector_db is mock_ctx.vector_db
            assert result.search_deps.embedding_client is mock_ctx.embedding
        finally:
            set_mcp_ctx(original)
