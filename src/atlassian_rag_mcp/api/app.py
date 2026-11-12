"""FastAPI app — mounts MCP server and REST API for indexing."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from atlassian_rag_mcp.api.lifespan import _init
from atlassian_rag_mcp.api.mcp_server import mcp, set_mcp_ctx
from atlassian_rag_mcp.api.routers import index_router
from atlassian_rag_mcp.config import settings
from atlassian_rag_mcp.log.logger import logger, setup_logging


class TrailingSlashMiddleware(BaseHTTPMiddleware):
    """Internally add trailing slash to /mcp without redirecting the client."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path
        if path == "/mcp" and request.method in ("POST", "GET", "DELETE"):
            request.scope["path"] = "/mcp/"
        return await call_next(request)


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Check X-API-Key on REST endpoints when API_KEY is configured."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if settings.api_key and request.url.path.startswith("/api/v1/"):
            key = request.headers.get("X-API-Key")
            if key != settings.api_key:
                return Response(
                    content='{"error":"Forbidden: invalid or missing X-API-Key"}',
                    status_code=403,
                    media_type="application/json",
                )
        return await call_next(request)


@asynccontextmanager
async def app_lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging(level=settings.log_level, log_file=settings.log_file)
    ctx = await _init()
    fastapi_app.state.ctx = ctx
    set_mcp_ctx(ctx)

    from atlassian_rag_mcp.confluence.services.indexer.worker import worker

    async with mcp.session_manager.run():
        num_workers = max(1, settings.embedding_max_workers)
        tasks = [
            asyncio.create_task(
                worker(
                    queue=ctx.index_queue,
                    confluence_client=ctx.confluence_client,
                    vector_service=ctx.vector_service,
                    worker_id=i,
                    idle_sleep=settings.worker_idle_sleep,
                )
            )
            for i in range(num_workers)
        ]
        persist_task = asyncio.create_task(ctx.index_queue.periodic_persist())
        sweep_task = asyncio.create_task(ctx.index_queue.dead_letter_sweep_loop())
        logger.info(f"Indexer workers started: {num_workers}")
        try:
            yield
        finally:
            persist_task.cancel()
            sweep_task.cancel()
            for t in tasks:
                t.cancel()
            await asyncio.gather(persist_task, sweep_task, *tasks, return_exceptions=True)
            await ctx.aclose()
            set_mcp_ctx(None)
            logger.info("Shutdown complete — all resources closed")

app = FastAPI(lifespan=app_lifespan, title="Atlassian RAG MCP")

app.add_middleware(TrailingSlashMiddleware)
app.add_middleware(ApiKeyMiddleware)

mcp_asgi = mcp.streamable_http_app(
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)
app.mount("/mcp", mcp_asgi)

app.include_router(index_router, prefix="/api/v1")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
