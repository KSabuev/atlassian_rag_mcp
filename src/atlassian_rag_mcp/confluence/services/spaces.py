"""Space service — sync SDK operations for Confluence spaces."""

from __future__ import annotations

from atlassian import Confluence

from atlassian_rag_mcp.confluence.models import SpaceResult, SpacesResult


class SpaceService:
    """Sync service for Confluence space operations."""

    def __init__(self, confluence: Confluence) -> None:
        self.confluence = confluence

    def get_all_spaces(self, limit: int = 50, start: int = 0) -> SpacesResult:
        """Get all accessible spaces (normalized)."""
        raw = self.confluence.get_all_spaces(start=start, limit=limit)
        return SpacesResult(
            spaces=raw.get("results", []),
            total=raw.get("size", 0),
        )

    def get_space(self, space_key: str, expand: str = "") -> SpaceResult:
        """Get space info by key (normalized)."""
        raw = self.confluence.get_space(space_key, expand=expand)
        homepage = raw.get("homepage") or {}
        return SpaceResult(
            key=raw.get("key", ""),
            name=raw.get("name", ""),
            homepage_id=homepage.get("id"),
        )
