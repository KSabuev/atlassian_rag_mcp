"""Chunks and Table modes — page chunks from the index."""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.confluence.models import Chunk, PageContent

from .base import PageContentMode


async def _get_chunks(
    *,
    page_id: str,
    svc: Any,
    block_index: int | None,
    limit: int,
    offset: int,
    chunk_index: int | None,
    result_kwargs: dict[str, Any],
) -> PageContent:
    """Fetch chunks by filter, optionally filtering by block_index and chunk_index."""
    in_work = await svc.index_queue.get_in_work_versions()
    raw = await svc.vector_service.get_chunks_by_filter(
        page_id=page_id,
        block_index=block_index,
        limit=limit,
        offset=offset,
        in_work_versions=in_work,
    )
    chunks = [Chunk.model_validate(c) for c in raw]

    if chunk_index is not None:
        all_raw = await svc.vector_service.get_chunks_by_filter(
            page_id=page_id, block_index=block_index, limit=10000,
            in_work_versions=in_work,
        )
        chunks = [
            Chunk.model_validate(c)
            for c in all_raw
            if c.get("chunk_index") == chunk_index
        ]

    return PageContent(**result_kwargs, chunks=chunks)


class ChunksMode(PageContentMode):
    @classmethod
    def name(cls) -> str:
        return "chunks"

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
        chunk_index: int | None = None,
        **kw: Any,
    ) -> PageContent:
        """Return page chunks, optionally filtered by chunk_index."""
        return await _get_chunks(
            page_id=page_id, svc=svc, block_index=None,
            limit=limit, offset=offset, chunk_index=chunk_index,
            result_kwargs=result_kwargs,
        )


class TableMode(PageContentMode):
    @classmethod
    def name(cls) -> str:
        return "table"

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
        **kw: Any,
    ) -> PageContent:
        """Return all rows/parts of one table block (block_index required)."""
        if block_index is None:
            return PageContent(**result_kwargs, error="what='table' requires block_index")
        return await _get_chunks(
            page_id=page_id, svc=svc, block_index=block_index,
            limit=limit, offset=offset, chunk_index=chunk_index,
            result_kwargs=result_kwargs,
        )
