"""FTS rebuild — re-index the FTS index from stored VectorDB payloads."""

from __future__ import annotations

from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.storage.base import VectorDB
from atlassian_rag_mcp.storage.fts.base import FTSIndex


async def rebuild_fts_from_vdb(
    vector_db: VectorDB,
    fts: FTSIndex,
    spaces: list[str],
    *,
    limit: int = 1000,
) -> int:
    """Rebuild the FTS docs of the given spaces from stored VDB payloads.

    Deletes the spaces' FTS docs, then scrolls the VDB payloads for those
    spaces (batched) and re-indexes one FTS doc per logical chunk. The
    VectorDB and the Confluence source are untouched — no re-embedding.

    Dedup is GLOBAL across scroll batches: sub-points of one split composite
    share a ``chunk_id`` and can straddle the batch boundary, so a per-batch
    dedup would index them twice. One FTS doc per logical chunk — split
    composites keep the full text on every sub-chunk under the same chunk_id.

    Returns the number of FTS docs indexed. Raises on failure.
    """
    space_keys = [s.strip() for s in (spaces or []) if s and s.strip()]
    if not space_keys:
        raise ValueError("rebuild_fts_from_vdb requires at least one space key")

    filter_condition = {"must": [{"key": "space_key", "match": {"any": space_keys}}]}

    if not await fts.delete_by_filter(filter_condition):
        raise RuntimeError(
            f"FTS rebuild failed: delete_by_filter rejected for spaces={space_keys}"
        )

    total = 0
    offset: str | None = None
    seen_chunk_ids: set[str] = set()
    while True:
        points, next_offset = await vector_db.scroll_points(
            filter_condition=filter_condition, limit=limit, offset=offset,
        )
        if not points:
            break
        payloads = [
            pl
            for pl in (p["payload"] for p in points if p.get("payload"))
            if pl["chunk_id"] not in seen_chunk_ids
            and not seen_chunk_ids.add(pl["chunk_id"])
        ]
        if payloads:
            if not await fts.index_batch(payloads):
                raise RuntimeError(f"FTS rebuild failed at {total} docs: index_batch rejected")
            total += len(payloads)
        if next_offset is None:
            break
        offset = next_offset
        logger.info(f"FTS rebuild progress: {total} docs")

    logger.info(f"FTS rebuild complete: {total} docs (spaces={space_keys})")
    return total
