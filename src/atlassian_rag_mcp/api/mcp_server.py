"""MCP server singleton and shared utilities.

ServerContext is exposed to tools through the MCP server lifespan
(``ctx.request_context.lifespan_context``). A module-level holder keeps a
reference for direct invocation and tests — tools read it via ``_get_svc()``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from mcp.server.mcpserver import Context, MCPServer

from atlassian_rag_mcp.api.lifespan import ServerContext

try:
    _VERSION = _pkg_version("atlassian-rag-mcp")
except PackageNotFoundError:
    _VERSION = "0.0.0"

_mcp_ctx: ServerContext | None = None


def set_mcp_ctx(ctx: ServerContext | None) -> None:
    """Set/clear the runtime ServerContext shared with MCP tools."""
    global _mcp_ctx
    _mcp_ctx = ctx


@asynccontextmanager
async def _mcp_lifespan(_server: MCPServer) -> AsyncIterator[ServerContext]:
    """Expose the shared ServerContext to tool handlers via lifespan context."""
    if _mcp_ctx is None:
        raise RuntimeError("ServerContext not initialized — call app_lifespan() first")
    yield _mcp_ctx


mcp = MCPServer(
    "atlassian-rag-mcp-search",
    version=_VERSION,
    lifespan=_mcp_lifespan,
)


def _get_svc(ctx: Context | None = None) -> ServerContext:
    """Return the ServerContext for the current request.

    Prefers the lifespan context (real MCP requests); falls back to the
    module holder for direct invocation and tests.
    """
    if ctx is not None:
        request_ctx = getattr(ctx, "request_context", None)
        lifespan_ctx = getattr(request_ctx, "lifespan_context", None)
        if lifespan_ctx is not None:
            return lifespan_ctx
    if _mcp_ctx is None:
        raise RuntimeError("ServerContext not initialized — call app_lifespan() first")
    return _mcp_ctx
