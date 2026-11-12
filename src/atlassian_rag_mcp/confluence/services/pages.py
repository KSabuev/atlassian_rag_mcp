"""Page service — sync SDK operations for Confluence pages and attachments."""

from __future__ import annotations

from urllib.parse import quote

from atlassian import Confluence

from atlassian_rag_mcp.confluence.models import AttachmentInfo, PageContent
from atlassian_rag_mcp.log.logger import logger


class PageService:
    """Sync service for Confluence page operations (mirrors Jira IssueService)."""

    def __init__(self, confluence: Confluence) -> None:
        self.confluence = confluence

    def get_page(self, page_id: str, expand: str = "version,ancestors") -> PageContent:
        """Get a Confluence page by ID with normalized output."""
        logger.debug(f"page={page_id}")
        expand = expand or ""
        if "space" not in expand.split(","):
            expand = f"{expand},space".strip(",")
        raw = self.confluence.get_page_by_id(page_id, status="current", expand=expand)
        if not raw:
            return PageContent(error=f"Page {page_id} not found")
        ancestors = raw.get("ancestors") or []
        space = raw.get("space") or {}
        return PageContent(
            page_id=raw.get("id"),
            page_title=raw.get("title"),
            space_key=space.get("key"),
            version=(raw.get("version") or {}).get("number", 0),
            last_updated=(raw.get("version") or {}).get("when"),
            parent_id=ancestors[-1]["id"] if ancestors else None,
            html=raw.get("body", {}).get("view", {}).get("value"),
            error=None,
        )

    def list_attachments(self, page_id: str, limit: int = 100) -> list[dict]:
        """List attachments of a page.

        Returns normalized dicts: title, media_type, size, version_number,
        created, download_url.
        """
        raw = self.confluence.get_attachments_from_content(
            page_id, start=0, limit=limit, expand="version",
        )
        results = (raw or {}).get("results", [])
        base_url = getattr(self.confluence, "url", None) or ""
        items: list[dict] = []
        for a in results:
            title = a.get("title") or ""
            version = a.get("version") or {}
            download_url = (
                f"{base_url}/download/attachments/{page_id}/{quote(title)}"
                if base_url and title else None
            )
            items.append(AttachmentInfo(
                title=title,
                media_type=a.get("mediaType"),
                size=a.get("size"),
                version_number=version.get("number"),
                created=version.get("when"),
                download_url=download_url,
            ).model_dump())
        return items
