"""Base class for page content modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from atlassian_rag_mcp.confluence.models import PageContent


class PageContentMode(ABC):
    """Abstract base for page content modes (info, html, chunks, etc.)."""

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        """Return mode identifier (e.g. 'info', 'html', 'chunks')."""

    @classmethod
    @abstractmethod
    async def execute(
        cls,
        *,
        page_id: str,
        page: Any,
        result_kwargs: dict[str, Any],
        svc: Any,
        limit: int = 1000,
        offset: int = 0,
        block_index: int | None = None,
        chunk_index: int | None = None,
        titles: list[str] | None = None,
    ) -> PageContent:
        """Execute mode and return PageContent."""
