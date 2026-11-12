"""Health check MCP tool (async)."""

from __future__ import annotations

import asyncio

from mcp.server.mcpserver import Context

from atlassian_rag_mcp.api.health import HealthCheckResult
from atlassian_rag_mcp.api.mcp_server import _get_svc, mcp
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.registry import error_message


@mcp.tool()
async def health_check(ctx: Context | None = None) -> HealthCheckResult:
    """Check health of all backend services (VectorDB, Confluence, Jira, Embedding)."""
    logger.debug("Health check requested")

    status = {"status": "healthy", "services": {}}
    svc = _get_svc(ctx)

    db = svc.vector_service.vector_db
    db_name = type(db).__name__.removesuffix("Client").removesuffix("Storage").lower()
    try:
        db_health = await db.health_check()
        status["services"][db_name] = db_health
        if db_health.get("status") != "healthy":
            status["status"] = "degraded"
    except Exception as e:
        status["services"][db_name] = {"status": "unhealthy", "error": error_message(e)}
        status["status"] = "degraded"

    try:
        spaces = await svc.confluence_client.get_all_spaces(start=0, limit=1)
        if spaces is not None:
            status["services"]["confluence"] = {"status": "healthy"}
        else:
            status["services"]["confluence"] = {"status": "unhealthy"}
            status["status"] = "degraded"
    except Exception as e:
        status["services"]["confluence"] = {"status": "unhealthy", "error": error_message(e)}
        status["status"] = "degraded"

    try:
        await asyncio.to_thread(svc.jira_client.validate)
        status["services"]["jira"] = {"status": "healthy"}
    except Exception as e:
        status["services"]["jira"] = {"status": "unhealthy", "error": error_message(e)}
        status["status"] = "degraded"

    for pname, plugin in svc.jira_client.plugins.items():
        try:
            ok = await plugin.validate()
            status["services"][f"jira_plugin_{pname}"] = {
                "status": "healthy" if ok else "not_available",
                "plugin": pname,
            }
        except Exception as e:
            status["services"][f"jira_plugin_{pname}"] = {"status": "unhealthy", "error": error_message(e)}
            status["status"] = "degraded"
    if not svc.jira_client.plugins:
        status["services"]["jira_plugins"] = {"status": "disabled", "plugin": []}

    try:
        dim = await svc.embedding.get_embedding_dimension()
        status["services"]["embedding"] = {
            "status": "healthy",
            "dimension": dim,
            "model": svc.embedding.model_name,
        }
    except Exception as e:
        status["services"]["embedding"] = {"status": "unhealthy", "error": error_message(e)}
        status["status"] = "degraded"

    return HealthCheckResult.model_validate(status)
