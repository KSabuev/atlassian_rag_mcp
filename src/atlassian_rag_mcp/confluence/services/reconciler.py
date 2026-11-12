"""Universal reconciler — finds inconsistencies between stores and reindexes.

The reconciler scans Confluence via CQL, compares versions with VectorDB,
detects stale / missing pages, and enqueues them for reindexing.

Also optionally checks FTS consistency (deep mode).
"""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.storage.utils import build_page_versions_map

from ...storage.base import VectorDB
from .. import ConfluenceClient
from ..cql_builder import build_space_cql
from ..models import ReconcilerResult
from .indexer.queue import IndexQueue
from .vector_service import VectorService


def _normalize_space_keys(space_key: Any) -> list[str]:
    """Normalize a space key (comma-separated string or list) to a clean list."""
    if isinstance(space_key, str):
        return [k.strip() for k in space_key.split(",") if k.strip()]
    if isinstance(space_key, (list, tuple)):
        return [k.strip() for k in space_key if k and k.strip()]
    return []


async def _vdb_page_space_keys(vector_db: VectorDB) -> dict[str, str | None]:
    """Scroll the whole collection once: page_id -> space_key (from payloads).

    Used to decide which VectorDB pages belong to the scanned spaces before
    orphan deletion (a page from another space is not "missing from
    Confluence" and must not be deleted).
    """
    out: dict[str, str | None] = {}
    offset: str | None = None
    while True:
        points, next_offset = await vector_db.scroll_points(
            filter_condition={}, limit=1000, offset=offset,
        )
        for p in points:
            payload = p.get("payload") or {}
            pid = payload.get("page_id")
            if pid and pid not in out:
                out[pid] = payload.get("space_key")
        if not points or next_offset is None:
            break
        offset = next_offset
    return out


async def reconcile(
    confluence_client: ConfluenceClient,
    vector_service: VectorService,
    index_queue: IndexQueue,
    space_key: Any = "",
    cql_batch_size: int = 50,
    deep: bool = False,
    force: bool = False,
    delete_stale: bool = True,
) -> dict:
    """Full reconciliation: CQL scan + VDB/FTS consistency check.

    ``space_key`` may be a single space key, a comma-separated string, or a list
    of space keys. Empty → scans all spaces (no restriction).

    ``force``: re-enqueue every page found by the scan, regardless of version
    (full re-index of the scanned scope).

    ``delete_stale``: delete pages that exist in the VectorDB but are missing
    from the (scoped) CQL scan.

    ``deep``: per-page verification against live Confluence (chunk presence +
    stored-vs-live version) for every page in the index within the scanned
    scope. Skipped when ``force`` is set — the scan already enqueued everything.

    Orphan detection under a space restriction: the CQL scan is limited to the
    scanned spaces, so a VectorDB page from another space (or with an unknown
    space) is NOT "missing from Confluence" — only pages whose stored
    ``space_key`` is among the scanned spaces are deleted as orphans. The same
    scope applies to the ``deep`` check.

    Returns dict with scanned / queued / deleted / fts_inconsistent counts.
    """
    space_keys = _normalize_space_keys(space_key)
    space_cql = build_space_cql(space_keys)
    cql_prefix = "type=page" + (f" AND {space_cql}" if space_cql else "")
    logger.info(f"Reconciliation started: spaces={space_keys}, deep={deep}")

    vdb_versions = await build_page_versions_map(vector_service.vector_db)
    vdb_page_ids = set(vdb_versions.keys())

    queued: set[str] = set()
    deleted: set[str] = set()
    scanned = 0
    cql_page_ids: set[str] = set()

    start = 0
    while True:
        cql = cql_prefix
        batch = await confluence_client.cql(
            cql=cql, start=start, limit=cql_batch_size, expand="content.version",
        )
        results = batch.results
        if not results:
            break

        for r in results:
            content = r.get("content", {})
            pid = content.get("id", "")
            if not pid:
                continue
            scanned += 1
            cql_page_ids.add(pid)

            if force or pid not in vdb_versions:
                await index_queue.enqueue(pid)
                queued.add(pid)
                continue

            live_ver = (content.get("version") or {}).get("number", 0) or 0
            if live_ver > 0 and vdb_versions[pid] != live_ver:
                await index_queue.enqueue(pid)
                queued.add(pid)

        start += cql_batch_size

    pages_in_vdb_not_cql = vdb_page_ids - cql_page_ids
    vdb_spaces: dict[str, str | None] | None = None
    if space_keys and pages_in_vdb_not_cql:
        vdb_spaces = await _vdb_page_space_keys(vector_service.vector_db)
        pages_in_vdb_not_cql = {
            pid for pid in pages_in_vdb_not_cql
            if vdb_spaces.get(pid) in space_keys
        }
    if delete_stale:
        for pid in pages_in_vdb_not_cql:
            try:
                ok = await vector_service.delete_page(pid)
                if ok:
                    deleted.add(pid)
                    logger.info(f"Deleted stale page not in Confluence: {pid}")
            except Exception as e:
                logger.warning(f"Failed to delete stale page {pid}: {e}")

    fts_inconsistent = 0
    if deep and not force:
        deep_page_ids = vdb_page_ids
        if space_keys:
            if vdb_spaces is None:
                vdb_spaces = await _vdb_page_space_keys(vector_service.vector_db)
            deep_page_ids = {
                pid for pid in vdb_page_ids if vdb_spaces.get(pid) in space_keys
            }
        for pid in deep_page_ids:
            try:
                chunks = await vector_service.get_chunks_by_filter(page_id=pid, limit=1)
                if not chunks:
                    await index_queue.enqueue(pid)
                    queued.add(pid)
                    fts_inconsistent += 1
                    continue
                pag = await confluence_client.get_page(pid, expand="version")
                if not pag.error:
                    conflu_ver = pag.version or 0
                    stored_ver = vdb_versions.get(pid, 0)
                    if conflu_ver > 0 and stored_ver != conflu_ver:
                        await index_queue.enqueue(pid)
                        queued.add(pid)
                        fts_inconsistent += 1
            except Exception as e:
                logger.warning(f"Deep check failed for page {pid}: {e}")

    logger.info(
        f"Reconciliation complete: scanned={scanned}, "
        f"queued={len(queued)}, deleted={len(deleted)}, "
        f"fts_inconsistent={fts_inconsistent}"
    )
    return ReconcilerResult(
        status="ok",
        pages_scanned=scanned,
        pages_queued=len(queued),
        pages_deleted=len(deleted),
        fts_inconsistent=fts_inconsistent,
    ).model_dump()
