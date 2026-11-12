"""Confluence client package — thin async facade over sync SDK services."""

from __future__ import annotations

from atlassian import Confluence

from atlassian_rag_mcp.confluence.client import ConfluenceClient
from atlassian_rag_mcp.confluence.cql_builder import CQLBuilder, sanitize_cql_value
from atlassian_rag_mcp.confluence.models import (
    AttachmentInfo,
    Chunk,
    CqlResult,
    Page,
    PageContent,
    SpaceResult,
    SpacesResult,
)
from atlassian_rag_mcp.confluence.services.cql import CqlService
from atlassian_rag_mcp.confluence.services.pages import PageService
from atlassian_rag_mcp.confluence.services.spaces import SpaceService
from atlassian_rag_mcp.utils.http_retry import sdk_transport_retry


def create_confluence_client(
    url: str = "",
    username: str = "",
    password: str = "",
    timeout: int = 60,
) -> ConfluenceClient:
    """Factory: create ConfluenceClient from scalar params.

    The SDK's request() is the only HTTP entry point (internal self.request
    calls pick up the wrapper), so it is wrapped once with the retry
    transport — the single retry point for all SDK calls.
    """
    confluence = Confluence(
        url=url, username=username, password=password,
        cloud=False, timeout=timeout,
    )
    confluence.request = sdk_transport_retry()(confluence.request)
    return ConfluenceClient(confluence)


__all__ = [
    "AttachmentInfo",
    "CQLBuilder",
    "Chunk",
    "ConfluenceClient",
    "CqlResult",
    "CqlService",
    "Page",
    "PageContent",
    "PageService",
    "SpaceResult",
    "SpacesResult",
    "SpaceService",
    "create_confluence_client",
    "sanitize_cql_value",
]
