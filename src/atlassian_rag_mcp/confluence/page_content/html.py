"""HTML mode — cleaned live HTML of the page."""

from __future__ import annotations

from typing import Any

from lxml.html import tostring

from atlassian_rag_mcp.confluence.chunker.html_cleaner import clean_confluence_html as _clean_html
from atlassian_rag_mcp.confluence.models import PageContent

from .base import PageContentMode


class HtmlMode(PageContentMode):
    @classmethod
    def name(cls) -> str:
        return "html"

    @classmethod
    async def execute(
        cls,
        *,
        page_id: str,
        page: Any,
        result_kwargs: dict[str, Any],
        **kw: Any,
    ) -> PageContent:
        """Return cleaned HTML content of the page ("" for empty pages)."""
        tree = _clean_html(page.html or "")
        if tree is None:
            return PageContent(**result_kwargs, html="")
        body = tree.find("body")
        empty = body is None or (len(body) == 0 and not (body.text or "").strip())
        return PageContent(**result_kwargs, html="" if empty else tostring(tree, encoding="unicode"))
