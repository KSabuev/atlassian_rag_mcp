"""cql_fts_fusion: CQL + FTS page branches fused by score (weighted/RRF).

Candidate set is the UNION of both branches — the FTS branch surfaces pages
the CQL pool missed. Tested: the CQL rank-value helper, union/ordering via
the full pipeline with mocked Confluence + FTS, excerpt provenance, and
param validation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atlassian_rag_mcp.confluence.models import CqlResult
from atlassian_rag_mcp.confluence.search import execute_algorithm
from atlassian_rag_mcp.confluence.search.cql_fts_fusion import CqlFtsFusionSearch, _cql_rank_values
from atlassian_rag_mcp.exceptions import FtsError


def _cql_item(pid, title, space_key, version=5, when="2024-01-01T00:00:00.000Z",
              excerpt="<p>excerpt</p>"):
    return {
        "content": {
            "id": pid,
            "title": title,
            "version": {"number": version, "when": when},
            "_expandable": {"space": f"https://c.example.com/rest/api/space/{space_key}"},
        },
        "excerpt": excerpt,
    }


def _fts_item(cid, score, pid, title, space_key, text="chunk text", **overrides):
    payload = {
        "chunk_id": cid,
        "text": text,
        "page_id": pid,
        "page_title": title,
        "page_version": 3,
        "updated_at": "2024-02-02",
        "space_key": space_key,
    }
    payload.update(overrides)
    return {"id": cid, "score": score, "payload": payload}


async def _run(cls_params=None, deps=None):
    params = {"cql": "q"} if cls_params is None else cls_params
    return await execute_algorithm("cql_fts_fusion", params, deps)


class TestCqlRankValues:
    def test_empty(self):
        assert _cql_rank_values(0) == []

    def test_single_is_one(self):
        assert _cql_rank_values(1) == [1.0]

    def test_linear_in_unit_interval(self):
        vals = _cql_rank_values(4)
        assert vals == [1.0, 2 / 3, 1 / 3, 0.0]

    def test_min_max_shaped(self):
        vals = _cql_rank_values(5)
        assert max(vals) == 1.0 and min(vals) == 0.0
        assert vals == sorted(vals, reverse=True)


class TestDescriptor:
    def test_date_params_document_sign_semantics(self):
        """Live check: CQL relative dates work ONLY inside now(); - = the last N days;
        a bare -7d as a date value is rejected (400). No future (+) offsets documented."""
        d = CqlFtsFusionSearch.descriptor()
        params = {p.name: p for p in d.parameters}
        for name in ("created", "lastmodified"):
            desc = params[name].description
            assert 'now("-7d")' in desc, name
            assert "+7d" not in desc, name
            assert "- = the last N days" in desc, name
            assert "ONLY inside now()" in desc, name


class TestPipeline:
    async def test_fts_only_pages_surface(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[
            _cql_item("c1", "C1", "PROJ"),
            _cql_item("c2", "C2", "PROJ"),
        ]))
        search_deps.fts_index.search = AsyncMock(return_value=[
            _fts_item("f1", 9.0, "c1", "C1", "PROJ"),
            _fts_item("f2", 8.0, "f1", "F1", "PROJ"),
        ])
        result = await _run(cls_params={"cql": "q", "fusion": "weighted", "w_kw": 0.7},
                            deps=search_deps)
        ids = [s["page_id"] for s in result["sources"]]
        assert "f1" in ids
        assert "c1" in ids
        assert ids[0] == "c1"

    async def test_malformed_fts_query_returns_readable_error(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[]))
        search_deps.fts_index.search = AsyncMock(
            side_effect=FtsError('FTS query failed: text ~ "q" — Syntax Error')
        )
        result = await _run(cls_params={"cql": "q"}, deps=search_deps)
        assert result["sources"] == []
        assert "FTS query failed" in result["error"]

    async def test_invalid_field_in_fts_source_raises(self):
        # the FTS branch is built from the CQL text; an unknown field prefix
        # in it is rejected before any index access (defensive — valid CQL
        # does not carry unquoted ``field:`` tokens)
        from atlassian_rag_mcp.confluence.search.cql_fts_fusion import build_fusion_query

        with pytest.raises(FtsError, match="unknown field"):
            build_fusion_query('type = "page" AND text ~ a:b')

    async def test_cql_only_pages_kept(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[
            _cql_item("c1", "C1", "PROJ"),
            _cql_item("c2", "C2", "PROJ"),
        ]))
        search_deps.fts_index.search = AsyncMock(return_value=[])
        result = await _run(cls_params={"cql": "q"}, deps=search_deps)
        assert [s["page_id"] for s in result["sources"]] == ["c1", "c2"]

    async def test_weighted_fts_weight_flops_order(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[
            _cql_item("c1", "C1", "PROJ"),
            _cql_item("c2", "C2", "PROJ"),
            _cql_item("c3", "C3", "PROJ"),
        ]))
        search_deps.fts_index.search = AsyncMock(return_value=[
            _fts_item("f1", 9.0, "c3", "C3", "PROJ"),
        ])
        result = await _run(
            cls_params={"cql": "q", "fusion": "weighted", "w_kw": 0.7},
            deps=search_deps,
        )
        assert [s["page_id"] for s in result["sources"]][:2] == ["c3", "c1"]

    async def test_rrf_ordering(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[
            _cql_item("c1", "C1", "PROJ"),
            _cql_item("c2", "C2", "PROJ"),
            _cql_item("c3", "C3", "PROJ"),
        ]))
        search_deps.fts_index.search = AsyncMock(return_value=[
            _fts_item("f1", 9.0, "c3", "C3", "PROJ"),
        ])
        result = await _run(
            cls_params={"cql": "q", "fusion": "rrf"},
            deps=search_deps,
        )
        assert result["sources"][0]["page_id"] == "c3"

    async def test_pool_cut_to_top_k(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[
            _cql_item(f"c{i}", f"C{i}", "PROJ") for i in range(search_deps.top_k)
        ]))
        search_deps.fts_index.search = AsyncMock(return_value=[
            _fts_item(f"f{i}", 10.0 - i, f"fx{i}", f"FX{i}", "PROJ")
            for i in range(10)
        ])
        result = await _run(cls_params={"cql": "q"}, deps=search_deps)
        assert len(result["sources"]) == search_deps.top_k

    async def test_excerpt_provenance(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[
            _cql_item("c1", "C1", "PROJ", excerpt="<p>cql excerpt</p>"),
        ]))
        search_deps.fts_index.search = AsyncMock(return_value=[
            _fts_item("f1", 5.0, "c1", "C1", "PROJ", text="fts chunk body"),
            _fts_item("f2", 4.0, "f1", "F1", "PROJ", text="fts only body"),
        ])
        result = await _run(
            cls_params={"cql": "q", "fusion": "weighted", "w_kw": 0.5},
            deps=search_deps,
        )
        by_id = {s["page_id"]: s for s in result["sources"]}
        assert "cql excerpt" in (by_id["c1"]["excerpt"] or "")
        assert "fts only body" in (by_id["f1"]["excerpt"] or "")

    async def test_source_fields(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[
            _cql_item("c1", "C1", "PROJ"),
        ]))
        search_deps.fts_index.search = AsyncMock(return_value=[
            _fts_item("f1", 5.0, "c1", "C1", "PROJ"),
            _fts_item("f2", 4.0, "f1", "F1", "PROJ"),
        ])
        result = await _run(
            cls_params={"cql": "q", "fusion": "weighted", "w_kw": 0.5},
            deps=search_deps,
        )
        by_id = {s["page_id"]: s for s in result["sources"]}
        c1, f1 = by_id["c1"], by_id["f1"]
        assert c1["cql_rank"] == 1
        assert f1["cql_rank"] is None
        assert c1["fts_score"] == pytest.approx(5.0)
        assert f1["fts_score"] == pytest.approx(4.0)
        assert c1["score"] == pytest.approx(1.0)
        assert f1["score"] == pytest.approx(0.0)
        assert "chunks" not in c1

    async def test_empty_cql_pool_fts_only(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[]))
        search_deps.fts_index.search = AsyncMock(return_value=[
            _fts_item("f1", 5.0, "f1", "F1", "PROJ"),
        ])
        result = await _run(
            cls_params={"cql": "q", "fusion": "weighted"},
            deps=search_deps,
        )
        assert [s["page_id"] for s in result["sources"]] == ["f1"]

    async def test_no_query_at_all(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock()
        search_deps.fts_index.search = AsyncMock()
        result = await _run(cls_params={}, deps=search_deps)
        assert result["sources"] == []
        assert search_deps.confluence_client.cql.await_count == 0
        assert search_deps.fts_index.search.await_count == 0

    async def test_fts_query_uses_cql_text_and_title_boost(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[]))
        search_deps.fts_index.search = AsyncMock(return_value=[])
        await _run(
            cls_params={"cql": "правила обработки", "title_boost": 2},
            deps=search_deps,
        )
        called = search_deps.fts_index.search.await_args.kwargs["query"]
        assert "^2" in called

    async def test_space_applies_to_both_branches(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[]))
        search_deps.fts_index.search = AsyncMock(return_value=[])
        await _run(
            cls_params={"cql": "q", "space": ["PROJ"]},
            deps=search_deps,
        )
        cql_cql = search_deps.confluence_client.cql.await_args.kwargs["cql"]
        fts_q = search_deps.fts_index.search.await_args.kwargs["query"]
        assert "PROJ" in cql_cql
        assert "PROJ" in fts_q

    async def test_unresolved_space_key_is_error(self, search_deps):
        item = _cql_item("c1", "C1", "")
        item["content"]["_expandable"] = {}
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[item]))
        search_deps.fts_index.search = AsyncMock(return_value=[])
        result = await _run(cls_params={"cql": "q"}, deps=search_deps)
        assert result["sources"] == []
        assert "space_key" in (result.get("error") or "")

    @pytest.mark.parametrize("params,error_part", [
        ({"cql": "q", "fusion": "bogo"}, "fusion"),
        ({"cql": "q", "w_kw": 1.5}, "w_kw"),
        ({"cql": "q", "title_boost": 0}, "title_boost"),
        ({"cql": "q", "title_boost": "x"}, "title_boost"),
        ({"cql": "q", "titel": "x"}, "titel"),
    ])
    async def test_invalid_params_return_error(self, search_deps, params, error_part):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[]))
        search_deps.fts_index.search = AsyncMock(return_value=[])
        result = await _run(cls_params=params, deps=search_deps)
        assert result["sources"] == []
        assert error_part in (result.get("error") or "")
