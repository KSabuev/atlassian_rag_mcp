"""vector_fts_fusion: two branches fused by score — RRF or weighted
min-max normalized sum — instead of positional interleaving.

Tested: the pure fusion functions, the boosted FTS query builder, and the
full pipeline (execute + top_k cut + postprocess) via mocked deps.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atlassian_rag_mcp.confluence.search import execute_algorithm
from atlassian_rag_mcp.confluence.search.base import POOL_MULTIPLIER
from atlassian_rag_mcp.confluence.search.vector_fts_fusion import (
    FTS_BRANCH_TOP,
    build_fusion_query,
    fuse_branches,
)
from atlassian_rag_mcp.exceptions import FtsError


def _point(cid, score, page_id, page_title, **overrides):
    payload = {
        "text": "t", "page_id": page_id, "page_title": page_title,
        "page_version": 1, "block_index": None, "block_part_id": None,
        "chunk_index": 0, "updated_at": "2024-01-01", "heading_hierarchy": [],
    }
    payload.update(overrides)
    return {"id": cid, "score": score, "payload": payload}


def _fts_point(cid, score, page_id, page_title, **overrides):
    payload = {
        "chunk_id": cid, "text": "fts", "page_id": page_id, "page_title": page_title,
        "page_version": 1, "chunk_index": 0, "block_index": None, "block_part_id": None,
        "heading_hierarchy": [],
    }
    payload.update(overrides)
    return {"id": cid, "score": score, "payload": payload}


def _chunk_ids(result):
    out = []
    for src in result["sources"]:
        for c in src.get("chunks", []):
            out.append(c["chunk_id"])
    return out


async def _run(cls_params=None, deps=None):
    return await execute_algorithm("vector_fts_fusion", cls_params or {"vector": "q"}, deps)


class TestFuseBranchesPure:
    def test_rrf_chunk_in_both_branches_wins(self):
        vec = [("c1", 0.9), ("c2", 0.8), ("c3", 0.7)]
        fts = [("c3", 5.0)]
        scores = fuse_branches(vec, fts, mode="rrf", rrf_k=40)
        assert scores["c3"] > scores["c1"] > scores["c2"]

    def test_rrf_single_branch(self):
        scores = fuse_branches([("c1", 0.9), ("c2", 0.8)], [], mode="rrf", rrf_k=40)
        assert scores == {
            "c1": pytest.approx(1.0 / 40),
            "c2": pytest.approx(1.0 / 41),
        }

    def test_rrf_empty(self):
        assert fuse_branches([], [], mode="rrf") == {}

    def test_weighted_chunk_in_both_beats_singles(self):
        vec = [("c1", 0.9), ("c2", 0.8)]
        fts = [("c1", 5.0), ("c3", 4.0)]
        scores = fuse_branches(vec, fts, mode="weighted", w_kw=0.5)
        assert scores["c1"] == pytest.approx(1.0)
        assert scores["c2"] == pytest.approx(0.0)
        assert scores["c3"] == pytest.approx(0.0)

    def test_weighted_keyword_weight_flops_the_order(self):
        vec = [("c1", 0.9), ("c2", 0.8)]
        fts = [("c2", 5.0), ("c3", 4.0)]
        scores = fuse_branches(vec, fts, mode="weighted", w_kw=0.7)
        assert scores["c2"] > scores["c1"] > scores["c3"]

    def test_weighted_degenerate_branch_maps_to_one(self):
        scores = fuse_branches([("c1", 0.5)], [("c1", 7.0)], mode="weighted", w_kw=0.5)
        assert scores["c1"] == pytest.approx(1.0)

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            fuse_branches([("c1", 1.0)], [], mode="bogo")


class TestBuildFusionQuery:
    def test_default_title_boost_is_3(self):
        q = build_fusion_query("правила обработки")
        assert "title:(правила OR обрабатыва)" in q or "^3" in q
        assert "^3" in q
        assert "text:" in q

    def test_custom_title_boost(self):
        assert "^6" in build_fusion_query("q", title_boost=6)
        assert "^1" in build_fusion_query("q", title_boost=1)

    def test_spaces_and_page_ids_are_appended(self):
        q = build_fusion_query("q", spaces=["PROJ"], page_ids=["123"])
        assert "space:PROJ" in q
        assert "123" in q

    def test_raw_query_passes_through_without_boost(self):
        q = build_fusion_query("text:abc", title_boost=6)
        assert "^6" not in q
        assert q.strip() == "text:abc"

    def test_per_term_boost_branches(self):
        q = build_fusion_query("онбординг ABC.DEF.01^4 тест^2", title_boost=3)
        assert "pattern:(ABC.DEF.01)^4" in q
        assert "(title:(тест))^2" in q
        assert "(title:(онбординг))^3" in q
        assert "text:(онбординг)" in q

    def test_empty_query_no_filters_is_empty(self):
        assert build_fusion_query("   ") == ""

    def test_unknown_field_raises(self):
        with pytest.raises(FtsError, match="unknown field"):
            build_fusion_query("chunk_id:c1")
        with pytest.raises(FtsError, match="unknown field"):
            build_fusion_query("foo:barBaz")

    def test_empty_field_value_raises(self):
        with pytest.raises(FtsError, match="empty value"):
            build_fusion_query("text:")

    def test_unbalanced_quotes_raise(self):
        with pytest.raises(FtsError, match="quotes"):
            build_fusion_query('text:"abc')

    def test_unbalanced_parens_raise(self):
        with pytest.raises(FtsError, match="parentheses"):
            build_fusion_query("text:(abc")


class TestPipeline:
    async def test_vector_pool_limit(self, search_deps):
        search_deps.vector_db.search = AsyncMock(return_value=[_point("c1", 0.9, "p1", "P")])
        search_deps.fts_index.search = AsyncMock(return_value=[])
        await _run(deps=search_deps)
        assert search_deps.vector_db.search.await_args.kwargs["limit"] == (
            search_deps.top_k * POOL_MULTIPLIER
        )

    async def test_fts_branch_limit_is_60(self, search_deps):
        search_deps.vector_db.search = AsyncMock(return_value=[])
        search_deps.fts_index.search = AsyncMock(return_value=[])
        await _run(deps=search_deps)
        assert search_deps.fts_index.search.await_args.kwargs["limit"] == FTS_BRANCH_TOP == 60

    async def test_malformed_fts_returns_readable_error(self, search_deps):
        search_deps.vector_db.search = AsyncMock(return_value=[])
        search_deps.fts_index.search = AsyncMock(
            side_effect=FtsError("FTS query failed: text:((( — Syntax Error: text:((( ")
        )
        result = await _run(deps=search_deps)
        assert result["sources"] == []
        assert "FTS query failed" in result["error"]
        assert "text:(((" in result["error"]

    async def test_unknown_param_rejected(self, search_deps):
        search_deps.vector_db.search = AsyncMock(return_value=[])
        search_deps.fts_index.search = AsyncMock(return_value=[])
        result = await _run(cls_params={"vector": "q", "bogus": 1}, deps=search_deps)
        assert result["sources"] == []
        assert "bogus" in (result.get("error") or "")

    async def test_unknown_field_fts_returns_readable_error(self, search_deps):
        search_deps.vector_db.search = AsyncMock(return_value=[])
        search_deps.fts_index.search = AsyncMock(return_value=[])
        result = await _run(
            cls_params={"vector": "q", "fts": "chunk_id:c1"}, deps=search_deps
        )
        assert result["sources"] == []
        assert "unknown field 'chunk_id'" in result["error"]
        search_deps.fts_index.search.assert_not_awaited()

    async def test_weighted_fusion_orders_by_score(self, search_deps):
        search_deps.vector_db.search = AsyncMock(return_value=[
            _point("c1", 0.9, "p1", "P"),
            _point("c2", 0.8, "p2", "P2"),
        ])
        search_deps.fts_index.search = AsyncMock(return_value=[
            _fts_point("c2", 5.0, "p2", "P2"),
            _fts_point("c3", 4.0, "p3", "P3"),
        ])
        result = await _run(
            cls_params={"vector": "q", "fusion": "weighted", "w_kw": 0.7},
            deps=search_deps,
        )
        assert _chunk_ids(result) == ["c2", "c1", "c3"]

    async def test_rrf_fusion_orders_by_score(self, search_deps):
        search_deps.vector_db.search = AsyncMock(return_value=[
            _point("c1", 0.9, "p1", "P"),
            _point("c2", 0.8, "p2", "P2"),
            _point("c3", 0.7, "p3", "P3"),
        ])
        search_deps.fts_index.search = AsyncMock(return_value=[
            _fts_point("c3", 5.0, "p3", "P3"),
        ])
        result = await _run(
            cls_params={"vector": "q", "fusion": "rrf"},
            deps=search_deps,
        )
        assert _chunk_ids(result) == ["c3", "c1", "c2"]

    async def test_pool_cut_to_top_k(self, search_deps):
        search_deps.vector_db.search = AsyncMock(return_value=[
            _point(f"c{i}", 1.1 - i * 0.01, f"p{i}", f"P{i}") for i in range(30)
        ])
        search_deps.fts_index.search = AsyncMock(return_value=[])
        result = await _run(deps=search_deps)
        assert len(_chunk_ids(result)) == search_deps.top_k
        assert len(result["sources"]) == search_deps.top_k

    async def test_title_boost_reaches_fts_query(self, search_deps):
        search_deps.vector_db.search = AsyncMock(return_value=[])
        search_deps.fts_index.search = AsyncMock(return_value=[])
        await _run(
            cls_params={"vector": "q", "title_boost": 2},
            deps=search_deps,
        )
        called = search_deps.fts_index.search.await_args.kwargs["query"]
        assert "^2" in called

    async def test_source_has_branch_scores(self, search_deps):
        search_deps.vector_db.search = AsyncMock(return_value=[
            _point("c1", 0.9, "p1", "P"),
        ])
        search_deps.fts_index.search = AsyncMock(return_value=[
            _fts_point("c1", 5.0, "p1", "P"),
        ])
        result = await _run(deps=search_deps)
        s = result["sources"][0]
        assert "vector_score" in s
        assert "fts_score" in s
        assert s["vector_score"] == pytest.approx(0.9)
        assert s["fts_score"] == pytest.approx(5.0)
        assert "rerank_score" not in s

    @pytest.mark.parametrize("params,error_part", [
        ({"vector": "q", "fusion": "bogo"}, "fusion"),
        ({"vector": "q", "w_kw": 1.5}, "w_kw"),
        ({"vector": "q", "title_boost": 0}, "title_boost"),
        ({"vector": "q", "title_boost": "x"}, "title_boost"),
    ])
    async def test_invalid_params_return_error(self, search_deps, params, error_part):
        search_deps.vector_db.search = AsyncMock(return_value=[])
        search_deps.fts_index.search = AsyncMock(return_value=[])
        result = await _run(cls_params=params, deps=search_deps)
        assert result["sources"] == []
        assert error_part in (result.get("error") or "")
