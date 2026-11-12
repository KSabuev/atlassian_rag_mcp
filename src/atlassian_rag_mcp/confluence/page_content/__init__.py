"""Page content mode registry — mode name → PageContentMode class."""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.confluence.models import PageContent
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.registry import error_message

from ..search import build_page_url
from .attachments_mode import AttachmentsMode
from .base import PageContentMode
from .chunks import ChunksMode, TableMode
from .files import FilesMode
from .html import HtmlMode
from .info import InfoMode

_registry: dict[str, type[PageContentMode]] = {
    mode.name(): mode for mode in (InfoMode, HtmlMode, ChunksMode, TableMode,
                                    FilesMode, AttachmentsMode)
}

VALID_MODES: list[str] = sorted(_registry.keys())

__all__ = [
    "AttachmentsMode",
    "ChunksMode",
    "FilesMode",
    "HtmlMode",
    "InfoMode",
    "PageContentMode",
    "TableMode",
    "VALID_MODES",
    "execute_page_content",
]


async def execute_page_content(
    mode: str,
    page_id: str,
    svc: Any,
    *,
    limit: int = 1000,
    offset: int = 0,
    block_index: int | None = None,
    chunk_index: int | None = None,
    titles: list[str] | None = None,
) -> PageContent:
    """Execute a page content mode and return PageContent.

    Handles page fetching, error handling, and delegates to the registered mode class.
    """
    result_kwargs: dict[str, Any] = dict(page_id=page_id)

    try:
        need_html = mode == "html"
        page = await svc.confluence_client.get_page(
            page_id,
            expand="body.view,version" if need_html else "version",
        )
        if page.error:
            return PageContent(**result_kwargs, error=page.error)

        result_kwargs.update(
            page_title=page.page_title,
            page_url=page.page_url or build_page_url(page_id, svc.search_deps.confluence_url),
            version=page.version,
            last_updated=page.last_updated,
        )

        mode_cls = _registry.get(mode)
        if mode_cls is None:
            return PageContent(
                **result_kwargs,
                error=(
                    f"Unknown what={mode!r}. Use 'info', 'html', 'chunks', 'table', "
                    f"'files' or 'attachments'."
                ),
            )

        return await mode_cls.execute(
            page_id=page_id,
            page=page,
            result_kwargs=result_kwargs,
            svc=svc,
            limit=limit,
            offset=offset,
            block_index=block_index,
            chunk_index=chunk_index,
            titles=titles,
        )

    except Exception as e:
        logger.error(f"confluence_get_content failed: {e}", exc_info=True)
        return PageContent(**result_kwargs, error=error_message(e))
