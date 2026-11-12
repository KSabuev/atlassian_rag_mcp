"""Check indexed page relevance vs live Confluence version."""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.log.logger import logger

from .. import ConfluenceClient
from .indexer import IndexQueue


async def check_relevant(
    results_list: list[dict[str, Any]],
    confluence_client: ConfluenceClient,
    index_queue: IndexQueue | None = None,
    *,
    batch_size: int = 10,
) -> list[dict[str, Any]]:
    """Return results with irrelevant (stale/deleted) pages removed (may enqueue re-index)."""
    if not results_list:
        return results_list

    stale_ids = set()
    unique_pages = []
    seen = set()
    for r in results_list:
        pid = r.get("page_id", "")
        if pid and pid not in seen:
            seen.add(pid)
            unique_pages.append(pid)

    top_pages = unique_pages[:batch_size]

    for pid in top_pages:
        try:
            live = await confluence_client.get_page(pid, expand="version")
            live_version = live.version
            stored_ver = 0
            for r in results_list:
                if r.get("page_id") == pid:
                    v = r.get("version") or 0
                    if v > stored_ver:
                        stored_ver = v
            if 0 < stored_ver != live_version:
                stale_ids.add(pid)
                if index_queue is not None:
                    await index_queue.enqueue(pid)
        except Exception as e:
            logger.warning(f"Stale check failed for page {pid}: {e}")

    if stale_ids:
        return [
            r for r in results_list
            if r.get("page_id", "") not in stale_ids
        ]

    return results_list
