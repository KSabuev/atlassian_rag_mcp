"""Shared utilities for vector operations — backend-agnostic."""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.confluence.models import ChunkPayload
from atlassian_rag_mcp.storage.base import VectorDB


def build_point(
    chunk: ChunkPayload,
    vector: list[float],
    *,
    point_id: str | None = None,
) -> dict[str, Any]:
    """Build a point dict from a chunk payload (compatible with all vector storage backends).

    ``point_id`` defaults to ``chunk.chunk_id``.  For split composites multiple
    sub-chunks may share ``chunk_id``/``text`` but must have distinct valid
    backend ids (UUID/int); pass an explicit unique ``point_id`` in that case.
    """
    return {
        "id": point_id or chunk.chunk_id,
        "vector": vector,
        "payload": chunk.model_dump(),
    }


def dedupe_chunks_by_id(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse entries that share the same ``chunk_id``.

    Split composites store the full ``text`` on every sub-chunk with the same
    ``chunk_id``, so a page would otherwise repeat the whole text N times.
    Only the first occurrence is kept.
    """
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for c in chunks:
        cid = c.get("chunk_id")
        if cid in seen:
            continue
        seen.add(cid)
        result.append(c)
    return result


def build_filter(
    page_ids: list[str] | None = None,
    page_version: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    spaces: list[str] | None = None,
) -> dict[str, Any] | None:
    """Build a filter dict from optional parameters (Qdrant-compatible format)."""
    conditions = []
    if page_ids:
        conditions.append({"key": "page_id", "match": {"any": page_ids}})
    if page_version is not None:
        conditions.append({"key": "page_version", "match": {"value": page_version}})
    if date_from or date_to:
        range_cond = {}
        if date_from:
            range_cond["gte"] = date_from
        if date_to:
            range_cond["lte"] = date_to
        conditions.append({"key": "updated_at", "range": range_cond})
    if spaces:
        conditions.append({"key": "space_key", "match": {"any": list(spaces)}})
    if not conditions:
        return None
    return {"must": conditions}


async def build_page_versions_map(vector_db: VectorDB) -> dict[str, int]:
    """Return dict[page_id -> max page_version] — delegates to VectorDB.

    Raises on failure so callers never treat a broken map as "no pages
    indexed" (which would otherwise trigger a full re-index storm).
    """
    return await vector_db.get_page_versions_map()
