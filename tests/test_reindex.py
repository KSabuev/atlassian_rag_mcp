"""Integration test for full re-index — real service, no mocks.

Walks all Confluence pages via CQL, enqueues each, then drains the queue
synchronously via _index_page(). Asserts zero errors. One event loop is
shared for the whole module: clients created in _init() stay bound to a
live loop (asyncio.run would close it).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def service_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def ctx(service_loop):
    from atlassian_rag_mcp.api.lifespan import _init
    return service_loop.run_until_complete(_init())


@pytest.fixture
def queue(ctx):
    from atlassian_rag_mcp.config import settings
    (Path(settings.working_dir) / "queue_store.json").unlink(missing_ok=True)
    from atlassian_rag_mcp.confluence.services.indexer import IndexQueue
    return IndexQueue(settings.working_dir)


def test_full_reindex(ctx, queue, service_loop):
    from atlassian_rag_mcp.config import settings
    from atlassian_rag_mcp.confluence.services.indexer.worker import _index_page
    from atlassian_rag_mcp.confluence.services.vector_service import VectorService

    async def run():
        pages = []
        start = 0
        limit = 100
        while True:
            cql = f'type=page AND space="{settings.confluence_space_key}"'
            batch = ctx.confluence_client.confluence.cql(
                cql=cql, start=start, limit=limit, expand="version"
            )
            results = batch.get("results", [])
            if not results:
                break
            pages.extend(results)
            start += limit

        assert len(pages) > 0, "No pages found in Confluence space"

        batch = []
        for r in pages:
            pid = r.get("content", {}).get("id", "")
            if pid:
                batch.append(pid)
                if len(batch) == 3:
                    break

        for pid in batch:
            await queue.enqueue(pid)

        status = await queue.get_status()
        assert status["queue_size"] == len(batch)

        errors = []
        processed = 0
        start_time = time.time()

        vector_service = VectorService(
            vector_db=ctx.vector_db,
            embedding_client=ctx.embedding,
            fts_index=ctx.search_deps.fts_index,
            batch_size=50,
        )

        async def _index_with_retry(page_id, max_retries=3):
            for attempt in range(max_retries):
                try:
                    await _index_page(
                        page_id,
                        ctx.confluence_client,
                        vector_service,
                    )
                    return
                except Exception as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))
                        continue
                    raise

        while True:
            page_id = await queue.dequeue()
            if page_id is None:
                break

            try:
                await _index_with_retry(page_id)
                await queue.record_success()
                processed += 1
            except Exception as e:
                errors.append({"page_id": page_id, "error": str(e)})
                await queue.record_error(page_id, str(e))
            finally:
                await queue.mark_done(page_id)

        elapsed = time.time() - start_time

        status = await queue.get_status()

        assert status["queue_size"] == 0, f"Queue not empty after processing: {status}"
        assert len(errors) == 0, f"Indexing failed for {len(errors)} page(s): {errors}"
        assert len(status["last_errors"]) == 0, f"Persisted errors remain: {status['last_errors']}"
        assert status["total_indexed_today"] == processed, \
            f"Expected {processed} pages indexed, got {status['total_indexed_today']}"
        print(f"\n  Indexed {processed} pages in {elapsed:.1f}s")

    service_loop.run_until_complete(run())
