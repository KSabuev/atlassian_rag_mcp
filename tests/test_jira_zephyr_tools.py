"""Unit tests for MCP Jira/Zephyr tools — the single error boundary per surface.

execute_method (Jira/Zephyr) is pure and raises; the tool layer converts the
exception to a structured error result — same policy as Confluence's
confluence_execute_algorithm.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from atlassian_rag_mcp.api.lifespan import ServerContext
from atlassian_rag_mcp.api.mcp_server import _mcp_ctx, set_mcp_ctx
from atlassian_rag_mcp.api.tools.jira import MethodName, jira_execute_method
from atlassian_rag_mcp.api.tools.zephyr import ZephyrMethodName, zephyr_execute_method


def _ctx(jira_client: MagicMock) -> ServerContext:
    ctx = object.__new__(ServerContext)
    ctx.jira_client = jira_client
    return ctx


class TestJiraToolErrorBoundary:
    async def test_exception_converted_to_error_result(self):
        client = MagicMock()
        client.issues.get_issue = MagicMock(side_effect=RuntimeError("boom"))
        original = _mcp_ctx
        set_mcp_ctx(_ctx(client))
        try:
            res = await jira_execute_method(MethodName.get_issue, {"issue_key": "PROJ-1"})
        finally:
            set_mcp_ctx(original)
        assert res.method == "get_issue"
        assert res.data is None
        assert res.error == "boom"


class TestZephyrToolErrorBoundary:
    async def test_plugin_not_enabled_converted_to_error_result(self):
        client = MagicMock()
        client.plugins = None
        original = _mcp_ctx
        set_mcp_ctx(_ctx(client))
        try:
            res = await zephyr_execute_method(ZephyrMethodName.get_projects, {})
        finally:
            set_mcp_ctx(original)
        assert res.method == "get_projects"
        assert res.data is None
        assert "not enabled" in res.error
