"""Jira client package with composition-based architecture."""

from __future__ import annotations

from atlassian import Jira

from atlassian_rag_mcp.jira.client import JiraClient, _is_cloud
from atlassian_rag_mcp.jira.issues import IssueService
from atlassian_rag_mcp.jira.jql_builder import (
    JQLBuilder,
    JqlSearchParams,
    build_jql_query,
    sanitize_jql_list,
    sanitize_jql_value,
)
from atlassian_rag_mcp.jira.plugins import create_plugins
from atlassian_rag_mcp.jira.search import SearchService
from atlassian_rag_mcp.utils.http_retry import sdk_transport_retry
from atlassian_rag_mcp.utils.query import parse_date


def create_jira_client(
    url: str = "",
    username: str = "",
    password: str = "",
    timeout: int = 30,
    project_key: str = "",
    plugins: str = "",
) -> JiraClient:
    """Factory: create JiraClient (with optional test-management plugins).

    The SDK's request() is the only HTTP entry point (internal self.request
    calls pick up the wrapper), so it is wrapped once with the retry
    transport — the single retry point for all SDK calls.
    """
    jira = Jira(
        url=url, username=username, password=password,
        cloud=_is_cloud(url), timeout=timeout,
    )
    jira.request = sdk_transport_retry()(jira.request)
    plugin_dict = create_plugins(jira, plugins, base_url=url, username=username, password=password)
    return JiraClient(jira, project_key=project_key or None, plugins=plugin_dict)


__all__ = [
    "IssueService",
    "JQLBuilder",
    "JiraClient",
    "JqlSearchParams",
    "SearchService",
    "build_jql_query",
    "create_jira_client",
    "parse_date",
    "sanitize_jql_list",
    "sanitize_jql_value",
]
