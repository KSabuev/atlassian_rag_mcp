"""Search context: first two rows of matched tables are appended as chunks.

A large table is indexed row-per-chunk (chunk_type='table_row'); only matched
rows are returned by search. The first two rows (header context) are fetched
from the vector store and appended. Deduping relies on chunk_id — an
already-matched header keeps the same chunk_id as the fetched copy.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from atlassian_rag_mcp.confluence.search import execute_algorithm


def _payload(chunk_id, *, block_index, block_part_id, page_id="p1", version=1, chunk_type="table_row", text="row"):
    return {
        "chunk_id": chunk_id,
        "chunk_type": chunk_type,
        "text": text,
        "page_id": page_id,
        "page_title": "Page 1",
        "page_version": version,
        "parent_id": "",
        "block_index": block_index,
        "block_part_id": block_part_id,
        "chunk_index": 0,
        "updated_at": "2024-01-01",
        "heading_hierarchy": [],
    }


def _point(chunk_id, **kwargs):
    return {"id": chunk_id, "score": 0.9, "payload": _payload(chunk_id, **kwargs)}


def _scroll_point(chunk_id, **kwargs):
    return {"id": chunk_id, "payload": _payload(chunk_id, **kwargs)}


async def _run(search_deps, points, *, scroll_points=None, scroll_error=None, in_work=None):
    search_deps.vector_db.search = AsyncMock(return_value=points)
    if scroll_error is not None:
        search_deps.vector_db.scroll_points = AsyncMock(side_effect=scroll_error)
    else:
        search_deps.vector_db.scroll_points = AsyncMock(return_value=scroll_points or ([], None))
    if in_work is not None:
        search_deps.get_in_work_versions = AsyncMock(return_value=in_work)
    else:
        search_deps.get_in_work_versions = AsyncMock(return_value={})
    return await execute_algorithm("vector_fts_fusion", {"vector": "query"}, search_deps)


class TestHeaderRowsAppended:
    async def test_first_two_rows_added_to_matched_rows(self, search_deps):
        result = await _run(
            search_deps,
            [_point("r5", block_index=3, block_part_id=5)],
            scroll_points=(
                [_scroll_point(f"r{i}", block_index=3, block_part_id=i) for i in range(1, 9)],
                None,
            ),
        )

        assert len(result["sources"]) == 1
        chunks = result["sources"][0]["chunks"]
        parts = {c["block_part_id"] for c in chunks if c["block_index"] == 3}
        assert parts == {1, 2, 5}
        assert all(c["chunk_type"] == "table_row" for c in chunks)

    async def test_matched_header_row_not_duplicated(self, search_deps):
        result = await _run(
            search_deps,
            [
                _point("r1", block_index=3, block_part_id=1),
                _point("r5", block_index=3, block_part_id=5),
            ],
            scroll_points=(
                [
                    _scroll_point("r1", block_index=3, block_part_id=1),
                    _scroll_point("r2", block_index=3, block_part_id=2),
                ],
                None,
            ),
        )

        chunks = result["sources"][0]["chunks"]
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))
        assert {c["block_part_id"] for c in chunks} == {1, 2, 5}

    async def test_table_with_single_row_adds_one(self, search_deps):
        result = await _run(
            search_deps,
            [_point("r3", block_index=3, block_part_id=3)],
            scroll_points=(
                [_scroll_point("r1", block_index=3, block_part_id=1)],
                None,
            ),
        )

        chunks = result["sources"][0]["chunks"]
        assert {c["block_part_id"] for c in chunks} == {1, 3}

    async def test_header_rows_do_not_affect_page_score(self, search_deps):
        result = await _run(
            search_deps,
            [_point("r5", block_index=3, block_part_id=5)],
            scroll_points=(
                [_scroll_point(f"r{i}", block_index=3, block_part_id=i) for i in (1, 2)],
                None,
            ),
        )

        assert result["sources"][0]["vector_score"] > 0.0


class TestHeaderEdgeCases:
    async def test_scroll_failure_does_not_break_search(self, search_deps):
        result = await _run(
            search_deps,
            [_point("r5", block_index=3, block_part_id=5)],
            scroll_error=RuntimeError("boom"),
        )

        assert result["error"] is None
        chunks = result["sources"][0]["chunks"]
        assert {c["block_part_id"] for c in chunks} == {5}

    async def test_completed_version_headers_fetched_while_new_in_work(self, search_deps):
        await _run(
            search_deps,
            [_point("r5", block_index=3, block_part_id=5, version=2)],
            in_work={"p1": 3},
        )

        assert search_deps.vector_db.scroll_points.await_count == 1

    async def test_in_work_version_headers_skipped(self, search_deps):
        result = await _run(
            search_deps,
            [_point("r5", block_index=3, block_part_id=5, version=3)],
            in_work={"p1": 3},
        )

        assert search_deps.vector_db.scroll_points.await_count == 0
        assert result["sources"] == []

    async def test_whole_table_chunk_not_augmented(self, search_deps):
        result = await _run(
            search_deps,
            [_point("t1", block_index=2, block_part_id=None, chunk_type="table", text="<table>...</table>")],
        )

        chunks = result["sources"][0]["chunks"]
        assert len(chunks) == 1
        assert chunks[0]["chunk_type"] == "table"

    async def test_no_table_rows_means_no_scroll(self, search_deps):
        result = await _run(
            search_deps,
            [_point("s1", block_index=None, block_part_id=None, chunk_type="section", text="text")],
        )

        assert search_deps.vector_db.scroll_points.await_count == 0
        assert len(result["sources"][0]["chunks"]) == 1
