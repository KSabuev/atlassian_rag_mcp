"""REST API endpoints for index operations."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from atlassian_rag_mcp.api.dependencies import get_ctx
from atlassian_rag_mcp.api.lifespan import ServerContext
from atlassian_rag_mcp.confluence.services.reconciler import _normalize_space_keys
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.storage.fts_rebuild import rebuild_fts_from_vdb

router = APIRouter(tags=["index"])


class IndexPageResult(BaseModel):
    status: str
    page_id: str


class SyncRequest(BaseModel):
    force: bool = False
    spaces: list[str] | None = None
    delete_stale: bool = False
    deep: bool = False
    fts_rebuild: bool = False


class SyncResult(BaseModel):
    status: str
    pages_scanned: int = 0
    pages_queued: int = 0
    error: str | None = None


class DeleteResult(BaseModel):
    status: str
    page_id: str
    deleted: bool = False
    message: str | None = None


@router.post("/index/page/{page_id}")
async def index_page(
    page_id: str,
    ctx: ServerContext = Depends(get_ctx),
) -> dict:
    """Enqueue a page for re-indexing."""
    await ctx.index_queue.enqueue(page_id)
    logger.info(f"REST /index/page/{page_id} → enqueued")
    return IndexPageResult(status="accepted", page_id=page_id).model_dump(exclude_none=True)


@router.post("/index/sync")
async def sync_space(
    body: SyncRequest,
    ctx: ServerContext = Depends(get_ctx),
) -> dict:
    """Sync spaces: CQL scan → version compare → enqueue stale/missing pages.

    Body flags:
      - ``force``: re-enqueue every scanned page regardless of version
        (full re-index of the scanned scope — use after a chunker change).
      - ``spaces``: restrict the sync to these space keys; omitted/empty →
        all configured spaces (CONFLUENCE_SPACE_KEY).
      - ``delete_stale``: delete pages that are gone from Confluence (orphans)
        within the scanned scope.
      - ``deep``: per-page verification against live Confluence (chunk presence
        + stored-vs-live version) for every indexed page in scope. Skipped
        when ``force`` is set; costs one get_page per indexed page.
      - ``fts_rebuild``: rebuild the FTS index from the stored VectorDB
        payloads of the target spaces (no re-embedding; VectorDB and
        Confluence untouched). The scope follows ``spaces`` (default: the
        configured ``CONFLUENCE_SPACE_KEY``); requires at least one space.
        Runs after the scan, so orphan deletions are already reflected.
    """
    space_keys = [k.strip() for k in (body.spaces or []) if k.strip()] or None

    try:
        result = await ctx.reconcile_fn(
            space_key=space_keys,
            deep=body.deep,
            force=body.force,
            delete_stale=body.delete_stale,
        )
        if body.fts_rebuild:
            rebuild_spaces = space_keys or _normalize_space_keys(ctx.search_deps.confluence_space_key)
            result = {
                **result,
                "fts_rebuilt": await rebuild_fts_from_vdb(
                    ctx.vector_db, ctx.search_deps.fts_index, rebuild_spaces,
                ),
            }
        logger.info(
            f"REST /index/sync → scanned={result.get('pages_scanned')}, "
            f"queued={result.get('pages_queued')}, deleted={result.get('pages_deleted')}, "
            f"fts_rebuilt={result.get('fts_rebuilt')}"
        )
        return result
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        return SyncResult(status="error", error=str(e)).model_dump(exclude_none=True)


@router.delete("/index/page/{page_id}")
async def delete_page(
    page_id: str,
    ctx: ServerContext = Depends(get_ctx),
) -> dict:
    """Delete a page from VectorDB (via queue)."""
    logger.info(f"REST DELETE /index/page/{page_id}")

    old = await ctx.vector_service.get_chunks_by_filter(page_id=page_id, limit=1)
    if not old:
        return DeleteResult(
            status="ok",
            page_id=page_id,
            deleted=False,
            message="not found in vector store",
        ).model_dump(exclude_none=True)

    ok = await ctx.vector_service.delete_page(page_id)
    if ok:
        logger.info(f"Deleted page {page_id} from vector store and FTS")
    return DeleteResult(
        status="ok" if ok else "error",
        page_id=page_id,
        deleted=ok,
    ).model_dump(exclude_none=True)


@router.get("/index/status")
async def index_status(
    ctx: ServerContext = Depends(get_ctx),
) -> dict:
    """Current indexing queue status."""
    return await ctx.index_queue.get_status()
