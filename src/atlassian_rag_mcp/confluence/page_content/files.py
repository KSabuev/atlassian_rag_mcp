"""Files mode — attached files with download links."""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.confluence.models import AttachmentInfo, PageContent

from .base import PageContentMode


class FilesMode(PageContentMode):
    @classmethod
    def name(cls) -> str:
        return "files"

    @classmethod
    async def execute(
        cls,
        *,
        page_id: str,
        page: Any,
        result_kwargs: dict[str, Any],
        svc: Any,
        titles: list[str] | None = None,
        **kw: Any,
    ) -> PageContent:
        """Return list of attached files with download URLs.

        ``titles`` — optional filter by exact file names (case-insensitive).
        """
        files_raw = await svc.confluence_client.list_attachments(page_id)
        files = [AttachmentInfo(**f) for f in files_raw]
        if titles:
            wanted = {t.lower() for t in titles}
            files = [f for f in files if f.title.lower() in wanted]
        return PageContent(**result_kwargs, files=files)
