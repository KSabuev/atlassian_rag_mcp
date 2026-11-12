"""Schema contract: all algorithms must produce consistent output fields.

Each test exercises a different algorithm with minimal mocks and
validates the common output schema (no per-algorithm logic).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atlassian_rag_mcp.confluence.models import CqlResult
from atlassian_rag_mcp.confluence.search import SearchDeps, execute_algorithm

BODY = {"text": "c", "page_id": "p1", "page_title": "P",
         "page_version": 1, "block_index": None, "block_part_id": None,
         "chunk_index": 0, "updated_at": "2024-01-01",
         "heading_hierarchy": []}


class TestChunkAlgorithms:
    @pytest.mark.parametrize("algo_name,deps_fixture,params,setup", [
        pytest.param(
            "vector_fts_fusion", "search_deps", {"vector": "q"},
            lambda d: setattr(d.vector_db, "search",
                               AsyncMock(return_value=[{"id": "c1", "score": 0.5, "payload": {**BODY}}])),
            id="vector_fts_fusion",
        ),
    ])
    async def test_chunk_schema(self, algo_name, deps_fixture, params, setup, request):
        deps = request.getfixturevalue(deps_fixture)
        setup(deps)

        result = await execute_algorithm(algo_name, params, deps)

        for s in result.get("sources", []):
            assert s["page_id"] == "p1"
            assert s["page_title"] == "P"
            assert "score" not in s
            assert isinstance(s.get("vector_score"), float)
            assert isinstance(s.get("fts_score"), float)
            assert "excerpt" not in s
            for c in s["chunks"]:
                assert "heading_text" not in c
                assert "block_index" in c
                assert c["block_index"] is None \
                       or isinstance(c["block_index"], int)
                assert c["chunk_id"]

    async def test_fulltext_schema(self, search_deps):
        search_deps.fts_index.search = AsyncMock(return_value=[
            {"id": "c1", "score": 9.0, "payload": {**BODY}},
        ])
        result = await execute_algorithm("fulltext_search", {"fts": "q"}, search_deps)

        assert len(result["sources"]) >= 1
        for s in result.get("sources", []):
            assert s["page_id"] == "p1"
            assert s["page_title"] == "P"
            assert "score" not in s
            assert "vector_score" not in s
            assert isinstance(s.get("fts_score"), float)
            assert "excerpt" not in s
            for c in s["chunks"]:
                assert "heading_text" not in c
                assert c["chunk_id"]

    async def test_cql_fts_fusion_schema(self, confluence_client):
        confluence_client.cql = AsyncMock(return_value=CqlResult(results=[
            {"content": {"id": "123", "title": "P", "version": {"number": 1},
                          "_expandable": {"space": "/rest/api/space/TEST"}},
              "excerpt": "excerpt text"},
        ]))
        fts = MagicMock()
        fts.search = AsyncMock(return_value=[])
        deps = SearchDeps(
            vector_db=confluence_client, embedding_client=confluence_client,
            confluence_client=confluence_client,
            fts_index=fts,
            top_k=25,
            confluence_space_key="TEST",
            confluence_url="https://c.example.com/",
        )

        result = await execute_algorithm("cql_fts_fusion", {"cql": "q"}, deps)

        assert len(result["sources"]) >= 1
        for s in result.get("sources", []):
            assert s["page_id"] == "123"
            assert s["page_title"] == "P"
            assert s["space_key"] == "TEST"
            assert isinstance(s["version"], int)
            assert s["excerpt"] == "excerpt text"
            assert isinstance(s["score"], float)
            assert isinstance(s["fts_score"], float)
            assert s["cql_rank"] == 1
            assert "chunks" not in s
            assert "vector_score" not in s
