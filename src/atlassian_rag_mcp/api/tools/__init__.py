"""MCP tools package — all tools registered via submodule imports."""

from __future__ import annotations

from atlassian_rag_mcp.api.tools import confluence, health, jira, zephyr
from atlassian_rag_mcp.api.tools.confluence import (
    AlgorithmName,
    confluence_execute_algorithm,
    confluence_get_algorithms,
    confluence_get_page_content,
    confluence_get_structure,
)
from atlassian_rag_mcp.api.tools.health import health_check
from atlassian_rag_mcp.api.tools.jira import (
    MethodName,
    jira_execute_method,
    jira_get_methods,
)
from atlassian_rag_mcp.api.tools.zephyr import (
    ZephyrMethodName,
    zephyr_execute_method,
    zephyr_get_methods,
)

__all__ = [
    "AlgorithmName",
    "MethodName",
    "ZephyrMethodName",
    "confluence",
    "confluence_execute_algorithm",
    "confluence_get_algorithms",
    "confluence_get_page_content",
    "confluence_get_structure",
    "health",
    "health_check",
    "jira",
    "jira_execute_method",
    "jira_get_methods",
    "zephyr",
    "zephyr_execute_method",
    "zephyr_get_methods",
]
