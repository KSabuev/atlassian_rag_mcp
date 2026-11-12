"""Background worker — drains the IndexQueue, indexes pages (async)."""

from __future__ import annotations

import asyncio

from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.registry import error_message

from ... import ConfluenceClient
from ...chunker.pipeline import chunking_pipeline
from ..vector_service import VectorService
from .queue import IndexQueue


async def worker(
    queue: IndexQueue,
    confluence_client: ConfluenceClient,
    vector_service: VectorService,
    worker_id: int = 0,
    idle_sleep: float = 5.0,
) -> None:
    """Background worker — drains the queue, indexes pages.

    At startup it ensures the vector collection exists once (not per page),
    retrying with a fixed delay: the app may start while the VectorDB or
    embedding host is briefly unavailable, and the worker must survive it.
    """
    logger.info(f"Indexer worker [{worker_id}] started")

    while True:
        try:
            await vector_service._ensure_collection()
            break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                f"Indexer worker [{worker_id}]: vector collection not ready "
                f"({error_message(e)}); retrying in 5s"
            )
            await asyncio.sleep(5.0)

    while True:
        page_id = await queue.dequeue()

        if page_id is None:
            await asyncio.sleep(idle_sleep)
            continue

        try:
            while True:
                try:
                    await _index_page(
                        page_id=page_id,
                        confluence_client=confluence_client,
                        vector_service=vector_service,
                        index_queue=queue,
                    )
                    await queue.record_success()
                    await queue.clear_retry(page_id)
                    logger.info(
                        f"Indexed page={page_id} "
                        f"(pending={queue.pending_count}, queue={queue.queue_size})"
                    )
                    break
                except Exception as e:
                    msg = error_message(e)
                    logger.error(f"Index failed page={page_id}: {msg}")
                    await queue.record_error(page_id, msg)
                    if await queue.should_retry(page_id):
                        await asyncio.sleep(await queue.retry_delay(page_id))
                        continue
                    await queue.record_dead_letter(page_id)
                    logger.warning(
                        f"Dead-letter page={page_id} after {queue._max_retries} retries "
                        f"(auto-sweep: up to {queue._max_sweeps} times)"
                    )
                    break
        except asyncio.CancelledError:
            await queue.requeue(page_id)
            raise
        finally:
            await queue.clear_in_work(page_id)
            await queue.mark_done(page_id)
            await queue.consume_requeue_after(page_id)


async def _index_page(
    page_id: str,
    confluence_client: ConfluenceClient,
    vector_service: VectorService,
    index_queue: IndexQueue | None = None,
) -> None:
    """Fetch, chunk, embed and index a single page (async).

    Strategy: all-or-nothing preparation followed by write-then-delete.
    1. Fetch page, chunk, embed, prepare FTS docs (no writes)
    2. Write the NEW chunks to VectorDB and FTS (all-or-nothing)
    3. After a successful write, remove any stale chunks for this page

    If step 2 fails, the page's previous data is left intact — it never
    disappears from search during a failed re-index.

    While this page is being (re)indexed, its target version is marked
    "in work" so search serves the last completed version instead of a
    partially-written one.
    """
    logger.debug(f"page={page_id}")

    page = await confluence_client.get_page(
        page_id, expand="body.view,version,ancestors"
    )
    if page.error:
        raise RuntimeError(f"Failed to fetch page {page_id}: {page.error}")
    title = page.page_title or ""
    html = page.html or ""
    version = page.version
    updated_at = page.last_updated
    parent_id = page.parent_id
    space_key = page.space_key

    if index_queue is not None:
        await index_queue.set_in_work(page_id, version)

    chunks = await asyncio.to_thread(
        chunking_pipeline,
        page_id=page_id,
        page_title=title,
        html=html,
        page_version=version,
        updated_at=updated_at,
        parent_id=parent_id,
    )
    for chunk in chunks:
        chunk.space_key = space_key

    if not chunks:
        existing_count = 0
        if vector_service is not None:
            try:
                existing_count = len(
                    await vector_service.get_chunks_by_filter(
                        page_id=page_id, limit=1
                    )
                )
            except Exception as e:
                logger.warning(
                    f"Page {page_id} produced 0 chunks; "
                    f"could not inspect existing data: {error_message(e)}"
                )
        if existing_count:
            if index_queue is not None:
                await index_queue.record_empty()
            logger.warning(
                f"Page {page_id} produced 0 chunks despite "
                f"{existing_count} existing chunk(s) in storage — "
                f"stale data kept as-is"
            )
        else:
            logger.info(
                f"Page {page_id} produced 0 chunks — "
                f"leaving storage untouched, page dropped from queue"
            )
        return

    await vector_service.index_page(
        page_id=page_id,
        chunks=chunks,
    )
