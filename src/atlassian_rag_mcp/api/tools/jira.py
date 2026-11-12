"""Jira MCP tools — meta tools over the Jira method registry."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from mcp.server.mcpserver import Context
from pydantic import Field

from atlassian_rag_mcp.api.mcp_server import _get_svc, mcp
from atlassian_rag_mcp.jira.methods import (
    execute_method,
    get_methods,
)
from atlassian_rag_mcp.jira.models import JiraMethodResult
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.registry import (
    DICT_PARAMS_DESC,
    MethodDescriptor,
    error_message,
    projects_example,
)

_method_ids = [m.id for m in get_methods()]
MethodName = Enum("MethodName", {name: name for name in _method_ids})

_JIRA_EXECUTE_DESCRIPTION = f"""Execute a Jira method by ID.

Read-only Jira API operations: issue data and context (tree, labels, attachments,
remote links, watchers, comments, changelog, worklog, transitions), search and
counts, project metadata (boards, epics, sprints, components, versions, statuses,
fields), users and groups.

The configured projects are: {projects_example()}.
"""


@mcp.tool()
async def jira_get_methods(
    ctx: Context | None = None,
) -> list[MethodDescriptor]:
    """List all available Jira methods with detailed descriptions and parameter schemas."""
    svc = _get_svc(ctx)
    return get_methods(client=svc.jira_client)


@mcp.tool(description=_JIRA_EXECUTE_DESCRIPTION)
async def jira_execute_method(
    method_name: Annotated[
        MethodName,
        Field(description="ID of the method from jira_get_methods (e.g. search)."),
    ],
    params: Annotated[
        dict,
        Field(description=DICT_PARAMS_DESC),
    ],
    ctx: Context | None = None,
) -> JiraMethodResult:
    """Execute a Jira method by ID (operation list: tool description)."""
    logger.debug(f"jira_execute_method: {method_name.value}, params={params}")

    try:
        svc = _get_svc(ctx)
        result = await execute_method(method_name.value, params, svc.jira_client)
        return JiraMethodResult(**result)

    except Exception as e:
        logger.error(f"jira_execute_method failed: {e}", exc_info=True)
        return JiraMethodResult(method=method_name.value, data=None, error=error_message(e))
