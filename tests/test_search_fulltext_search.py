"""fulltext_search: pure FTS (Tantivy) lexical search — isolated unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atlassian_rag_mcp.confluence.search import execute_algorithm
from atlassian_rag_mcp.exceptions import FtsError


def _payload(chunk_id, page_id, page_title, **overrides):
    base = {
        "text": "some text",
        "page_id": page_id,
        "page_title": page_title,
        "page_version": 1,
        "block_index": None,
        "block_part_id": None,
        "chunk_index": 0,
        "updated_at": "2024-01-01",
        "heading_hierarchy": [],
    }
    base.update(overrides)
    base.setdefault("chunk_id", chunk_id)
    return base


def _hit(chunk_id, score, page_id, page_title, **overrides):
    return {"id": chunk_id, "score": score, "payload": _payload(chunk_id, page_id, page_title, **overrides)}


class TestBasicSearch:
    async def test_returns_ranked_results(self, search_deps):
        search_deps.fts_index.search = AsyncMock(return_value=[
            _hit("c1", 12.5, "p1", "Page 1"),
        ])
        result = await execute_algorithm("fulltext_search", {"fts": "процесс"}, search_deps)

        assert len(result["sources"]) == 1
        s = result["sources"][0]
        assert s["page_id"] == "p1"
        assert s["page_title"] == "Page 1"
        assert s["fts_score"] == 12.5
        assert len(s["chunks"]) == 1
        assert s["chunks"][0]["chunk_id"] == "c1"
        assert s["page_url"] == "https://c.example.com/pages/viewpage.action?pageId=p1"

    async def test_empty_search(self, search_deps):
        search_deps.fts_index.search = AsyncMock(return_value=[])
        result = await execute_algorithm("fulltext_search", {"fts": "zzz"}, search_deps)

        assert result["sources"] == []

    async def test_missing_fts_query(self, search_deps):
        result = await execute_algorithm("fulltext_search", {}, search_deps)

        assert result["sources"] == []
        assert "Missing required parameter(s): fts" in result["error"]

    async def test_unknown_param_rejected(self, search_deps):
        result = await execute_algorithm(
            "fulltext_search", {"fts": "q", "bogus": 1}, search_deps,
        )
        assert result["sources"] == []
        assert "bogus" in (result.get("error") or "")


class TestTopKTruncation:
    async def test_results_truncated_to_top_k(self, search_deps):
        hits = [_hit(f"c{i}", 100.0 - i, f"p{i}", f"P{i}") for i in range(40)]
        search_deps.fts_index.search = AsyncMock(return_value=hits)
        result = await execute_algorithm("fulltext_search", {"fts": "q"}, search_deps)

        assert len(result["sources"]) == 25
        scores = [s["fts_score"] for s in result["sources"]]
        assert scores == [100.0 - i for i in range(25)]


class TestFtsRanking:
    async def test_pages_ranked_by_best_fts(self, search_deps):
        search_deps.fts_index.search = AsyncMock(return_value=[
            _hit("c1", 10.0, "p1", "P1"),
            _hit("c2", 5.0, "p1", "P1"),
            _hit("c3", 15.0, "p2", "P2"),
            _hit("c4", 5.0, "p3", "P3"),
        ])
        result = await execute_algorithm("fulltext_search", {"fts": "q"}, search_deps)

        order = [s["page_id"] for s in result["sources"]]
        assert order == ["p2", "p1", "p3"]
        by_id = {s["page_id"]: s for s in result["sources"]}
        assert by_id["p1"]["fts_score"] == 10.0
        assert len(by_id["p1"]["chunks"]) == 2

    async def test_fts_error_propagates(self, search_deps):
        search_deps.fts_index.search = AsyncMock(side_effect=RuntimeError("index down"))
        with pytest.raises(RuntimeError, match="index down"):
            await execute_algorithm("fulltext_search", {"fts": "q"}, search_deps)

    async def test_malformed_fts_returns_readable_error(self, search_deps):
        search_deps.fts_index.search = AsyncMock(
            side_effect=FtsError("FTS query failed: text:((( — Syntax Error: text:((( ")
        )
        result = await execute_algorithm("fulltext_search", {"fts": "text:((("}, search_deps)
        assert result["sources"] == []
        assert "unbalanced parentheses" in result["error"]
        assert "text:(((" in result["error"]
        search_deps.fts_index.search.assert_not_awaited()

    async def test_unknown_field_fts_returns_readable_error(self, search_deps):
        search_deps.fts_index.search = AsyncMock(return_value=[])
        result = await execute_algorithm("fulltext_search", {"fts": "chunk_id:c1"}, search_deps)
        assert result["sources"] == []
        assert "unknown field 'chunk_id'" in result["error"]
        search_deps.fts_index.search.assert_not_awaited()

    async def test_index_fts_error_propagates(self, search_deps):
        search_deps.fts_index.search = AsyncMock(
            side_effect=FtsError("FTS query failed: text:(a AND b) — Syntax Error")
        )
        result = await execute_algorithm(
            "fulltext_search", {"fts": "text:(a AND b)"}, search_deps,
        )
        assert result["sources"] == []
        assert "FTS query failed" in result["error"]


class TestFieldIntegrity:
    async def test_source_has_no_vector_or_rank_score(self, search_deps):
        search_deps.fts_index.search = AsyncMock(return_value=[_hit("c1", 8.0, "p1", "P1")])
        result = await execute_algorithm("fulltext_search", {"fts": "q"}, search_deps)

        s = result["sources"][0]
        assert "score" not in s
        assert "vector_score" not in s
        assert "excerpt" not in s
        assert "fts_score" in s
        assert "chunks" in s


class TestSpaceAndPageFilters:
    async def test_spaces_embedded_in_fts_query(self, search_deps):
        search_deps.fts_index.search = AsyncMock(return_value=[])
        await execute_algorithm(
            "fulltext_search", {"fts": "term", "spaces": ["AAA", "BBB"]}, search_deps,
        )
        called_query = search_deps.fts_index.search.await_args.kwargs["query"]
        assert "space:AAA" in called_query
        assert "space:BBB" in called_query

    async def test_page_ids_embedded_in_fts_query(self, search_deps):
        search_deps.fts_index.search = AsyncMock(return_value=[])
        await execute_algorithm(
            "fulltext_search", {"fts": "term", "page_ids": ["111", "222"]}, search_deps,
        )
        called_query = search_deps.fts_index.search.await_args.kwargs["query"]
        assert "page:111" in called_query
        assert "page:222" in called_query
