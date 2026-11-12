"""Chunking pipeline — standalone function for indexing."""

from __future__ import annotations

import logging

from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.utils.tokenizer import count_qwen3_tokens

from ..models import ChunkPayload
from .html_chunker import HtmlChunker


def chunking_pipeline(
    page_id: str,
    page_title: str,
    html: str,
    page_version: int | None = None,
    updated_at: str | None = None,
    parent_id: str | None = None,
    max_tokens: int | None = None,
) -> list[ChunkPayload]:
    """Clean, chunk, and return ChunkPayload list ready for vector database indexing."""
    logger.debug(
        f"chunking_pipeline input: page_id={page_id} page_title={page_title!r} "
        f"html_len={len(html)} page_version={page_version} updated_at={updated_at} "
        f"parent_id={parent_id} max_tokens={max_tokens}"
    )

    chunks = HtmlChunker().process(
        page_id,
        page_title,
        html,
        page_version=page_version,
        updated_at=updated_at,
        parent_id=parent_id,
        max_tokens=max_tokens,
    )

    by_type: dict[str, int] = {}
    for chunk in chunks:
        by_type[chunk.chunk_type] = by_type.get(chunk.chunk_type, 0) + 1
    total_tokens = (
        sum(count_qwen3_tokens(c.text) for c in chunks)
        if logger.isEnabledFor(logging.DEBUG)
        else 0
    )
    summary = [f"chunks={len(chunks)}"]
    summary.extend(f"{t}={n}" for t, n in sorted(by_type.items()))
    summary.append(f"total_tokens={total_tokens}")
    logger.debug(f"chunking_pipeline output: page_id={page_id} " + " ".join(summary))
    return chunks
