"""ConfluenceClient — thin async facade composing sync services over the atlassian SDK."""

from __future__ import annotations

import asyncio

from atlassian import Confluence
from requests.exceptions import ConnectionError as RequestsConnectionError

from atlassian_rag_mcp.confluence.models import CqlResult, PageContent, SpaceResult, SpacesResult
from atlassian_rag_mcp.confluence.services.cql import CqlService
from atlassian_rag_mcp.confluence.services.pages import PageService
from atlassian_rag_mcp.confluence.services.spaces import SpaceService
from atlassian_rag_mcp.exceptions import AuthenticationError
from atlassian_rag_mcp.log.logger import logger


class ConfluenceClient:
    """Main Confluence client — composition of sync services plus async adapters.

    Mirrors JiraClient: SDK requests and normalization live in sync services
    (pages, cql_service, spaces); the client is the composition root. The
    async methods are thin adapters over ``asyncio.to_thread`` — the single
    async/thread boundary for the domain, serving async callers (indexer,
    search algorithms, page_content, REST, tools).
    """

    def __init__(self, confluence: Confluence) -> None:
        self.confluence = confluence
        self.pages = PageService(confluence)
        self.cql_service = CqlService(confluence)
        self.spaces = SpaceService(confluence)

    def validate(self, url: str = "") -> None:
        """Validate credentials by making a simple Confluence API call.

        Raises AuthenticationError if connection or auth fails.
        """
        try:
            logger.debug("Testing Confluence credentials...")
            spaces = self.confluence.get_all_spaces(start=0, limit=1)
            if spaces is not None:
                count = len(spaces.get("results", []))
                logger.info(f"Confluence OK — {count} spaces accessible")
            else:
                raise AuthenticationError(
                    "Confluence API returned None — check CONFLUENCE_URL and credentials."
                )
        except RequestsConnectionError as e:
            raise AuthenticationError(
                f"Cannot connect to Confluence at {url}. "
                f"Check CONFLUENCE_URL and network connectivity."
            ) from e
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(
                f"Confluence validation failed: {e}"
            ) from e

    async def get_page(self, page_id: str, expand: str = "version,ancestors") -> PageContent:
        """Get a Confluence page by ID (async)."""
        return await asyncio.to_thread(self.pages.get_page, page_id, expand=expand)

    async def cql(
        self, cql: str, start: int = 0, limit: int = 50, expand: str = "content.version",
    ) -> CqlResult:
        """Execute a CQL query (async)."""
        logger.info(f"CQL -> Confluence: {cql}")
        return await asyncio.to_thread(
            self.cql_service.cql, cql, start=start, limit=limit, expand=expand,
        )

    async def get_all_spaces(self, limit: int = 50, start: int = 0) -> SpacesResult:
        """Get all accessible spaces (async)."""
        return await asyncio.to_thread(self.spaces.get_all_spaces, limit=limit, start=start)

    async def get_space(self, space_key: str, expand: str = "") -> SpaceResult:
        """Get space info by key (async)."""
        return await asyncio.to_thread(self.spaces.get_space, space_key, expand=expand)

    async def list_attachments(self, page_id: str, limit: int = 100) -> list[dict]:
        """List attachments of a page (async)."""
        return await asyncio.to_thread(self.pages.list_attachments, page_id, limit=limit)
