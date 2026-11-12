"""Zephyr Scale MCP tools — meta tools over the Zephyr method registry."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from atlassian_rag_mcp.api.mcp_server import _get_svc, mcp
from atlassian_rag_mcp.jira.plugins.zephyr import (
    ZephyrMethodResult,
    execute_method,
    get_methods,
)
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.registry import (
    DICT_PARAMS_DESC,
    MethodDescriptor,
    error_message,
    projects_example,
)

_method_ids = [m.id for m in get_methods()]
ZephyrMethodName = Enum("ZephyrMethodName", {name: name for name in _method_ids})

_ZEPHYR_EXECUTE_DESCRIPTION = f"""Execute a Zephyr Scale method by ID.

Read-only test-management operations: test cases (details, versions, execution
history, trace links to Jira issues / Confluence pages / web links, comments,
attachments), test runs (items, individual executions with per-step results),
test plans, folder trees, and project metadata (case statuses, priorities,
result statuses, custom fields, permissions, datasets). Also downloads files
attached to test executions (base64).

The configured projects are: {projects_example()}.
"""


@mcp.tool()
async def zephyr_get_methods(
    ctx: Context | None = None,
) -> list[MethodDescriptor]:
    """List all available Zephyr Scale test-management methods with detailed descriptions and parameter schemas."""
    return get_methods()


@mcp.tool(description=_ZEPHYR_EXECUTE_DESCRIPTION)
async def zephyr_execute_method(
    method_name: Annotated[
        ZephyrMethodName,
        Field(description="ID of the method from zephyr_get_methods (e.g. search_test_cases)."),
    ],
    params: Annotated[
        dict,
        Field(description=DICT_PARAMS_DESC),
    ],
    ctx: Context | None = None,
) -> ZephyrMethodResult:
    """Execute a Zephyr Scale method by ID (operation list: tool description)."""
    logger.debug(f"zephyr_execute_method: {method_name.value}, params={params}")

    try:
        svc = _get_svc(ctx)
        result = await execute_method(method_name.value, params, svc.jira_client)
        return ZephyrMethodResult(**result)

    except Exception as e:
        logger.error(f"zephyr_execute_method failed: {e}", exc_info=True)
        return ZephyrMethodResult(method=method_name.value, data=None, error=error_message(e))
