"""FastAPI dependencies — access ServerContext via request.app.state.

REST endpoints declare ``ctx: ServerContext = Depends(get_ctx)``
instead of calling ``from atlassian_rag_mcp.api.lifespan import _init``.
"""

from __future__ import annotations

from fastapi import Request

from atlassian_rag_mcp.api.lifespan import ServerContext


async def get_ctx(request: Request) -> ServerContext:
    """Return ServerContext from request.app.state."""
    return request.app.state.ctx
