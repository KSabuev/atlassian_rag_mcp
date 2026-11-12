"""Info mode — lightweight page metadata only."""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.confluence.models import PageContent

from .base import PageContentMode


class InfoMode(PageContentMode):
    @classmethod
    def name(cls) -> str:
        return "info"

    @classmethod
    async def execute(
        cls,
        *,
        page_id: str,
        page: Any,
        result_kwargs: dict[str, Any],
        **kw: Any,
    ) -> PageContent:
        """Return page metadata without content."""
        return PageContent(**result_kwargs)
