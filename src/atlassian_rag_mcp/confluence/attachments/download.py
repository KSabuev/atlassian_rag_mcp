"""Download an attachment from Confluence (httpx, size-capped)."""

from __future__ import annotations

import httpx

from atlassian_rag_mcp.config.settings import settings
from atlassian_rag_mcp.utils.http_retry import httpx_transport_retry

from .errors import AttachmentTooLarge


@httpx_transport_retry()
async def download_attachment(
    download_url: str,
    max_bytes: int,
    timeout: float = 120.0,
) -> bytes:
    """Stream the file with basic auth; raise AttachmentTooLarge when the cap is hit.

    A mid-stream connection drop restarts the download (GET is idempotent);
    AttachmentTooLarge is a business error and is never retried.
    """
    auth = (settings.confluence_username, settings.confluence_password)
    async with httpx.AsyncClient(auth=auth, timeout=timeout, follow_redirects=True) as client:
        chunks: list[bytes] = []
        total = 0
        async with client.stream("GET", download_url) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code} downloading {download_url}")
            async for chunk in resp.aiter_bytes(65536):
                total += len(chunk)
                if total > max_bytes:
                    raise AttachmentTooLarge(
                        f"file exceeds {max_bytes // (1024 * 1024)} MB limit"
                    )
                chunks.append(chunk)
    return b"".join(chunks)
