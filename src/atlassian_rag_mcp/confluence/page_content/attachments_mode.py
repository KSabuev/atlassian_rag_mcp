"""Attachments mode — attached files with extracted text."""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.config.settings import settings
from atlassian_rag_mcp.confluence.attachments import (
    AttachmentError,
    classify,
    download_attachment,
    extract_text,
)
from atlassian_rag_mcp.confluence.models import AttachmentFile, AttachmentInfo, PageContent
from atlassian_rag_mcp.log.logger import logger

from .base import PageContentMode


def _base(info: AttachmentInfo, status: str, text: str | None = None,
          truncated: bool = False) -> AttachmentFile:
    return AttachmentFile(**info.model_dump(), status=status, text=text, truncated=truncated)


class AttachmentsMode(PageContentMode):
    @classmethod
    def name(cls) -> str:
        return "attachments"

    @classmethod
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
        """Download and convert attached files to text.

        ``titles`` — exact file names (case-insensitive); when set, only
        those attachments are processed and missing ones are reported
        with status "not_found". Otherwise all attachments are processed
        (offset/limit pagination, capped by ATTACHMENT_MAX_FILES).
        """
        raw = await svc.confluence_client.list_attachments(page_id)
        items = [AttachmentInfo(**f) for f in raw]

        not_found: list[str] = []
        if titles:
            wanted = {t.lower() for t in titles}
            items = [it for it in items if it.title.lower() in wanted]
            matched = {it.title.lower() for it in items}
            not_found = [t for t in titles if t.lower() not in matched]

        cap = settings.attachment_max_files
        items = items[offset:offset + min(limit, cap)]

        max_bytes = settings.attachment_max_size_mb * 1024 * 1024
        out: list[AttachmentFile] = []
        for it in items:
            reason = classify(it.title, it.media_type)
            if reason:
                out.append(_base(it, reason))
                continue
            if it.size and it.size > max_bytes:
                out.append(_base(it, "skipped:too_large"))
                continue
            if not it.download_url:
                out.append(_base(it, "error: no download url"))
                continue
            try:
                data = await download_attachment(it.download_url, max_bytes)
                text = extract_text(it.title, data) or "(empty file)"
                text = f"Файл: {it.title}\n\n{text}"
                truncated = len(text) > settings.attachment_text_max_chars
                if truncated:
                    text = text[: settings.attachment_text_max_chars]
                out.append(_base(it, "ok", text=text, truncated=truncated))
            except AttachmentError as e:
                out.append(_base(it, f"error: {e}"))
            except Exception as e:
                logger.warning(f"Attachment {it.title!r} on page {page_id} failed: {e}")
                out.append(_base(it, f"error: {e}"))

        for t in not_found:
            out.append(AttachmentFile(title=t, status="not_found"))

        return PageContent(**result_kwargs, attachment_files=out)
