"""Entry point for MCP server + REST API."""
from __future__ import annotations

import logging

import uvicorn

from atlassian_rag_mcp.config import settings
from atlassian_rag_mcp.log.logger import logger, setup_logging


def main() -> None:
    """Start the MCP server + REST API on settings.host:settings.port."""
    host = settings.host
    port = settings.port

    setup_logging(level=settings.log_level, log_file=settings.log_file)

    logger.info("Starting Atlassian RAG MCP server...")
    logger.info(f"MCP endpoint:     http://{host}:{port}/mcp")
    logger.info(f"REST API:         http://{host}:{port}/api/v1/index")
    logger.info(f"Swagger:          http://{host}:{port}/docs")
    logger.info(f"Log level: {logging.getLevelName(logger.getEffectiveLevel())}")

    from atlassian_rag_mcp.api.app import app

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
