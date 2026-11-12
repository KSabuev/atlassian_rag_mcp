"""search_params: optional per-request constants object — isolated unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atlassian_rag_mcp.confluence.search import execute_algorithm, get_algorithms, resolve_search_params
from atlassian_rag_mcp.confluence.search.base import POOL_MULTIPLIER, RRF_K

N_CANDIDATES = 25
TOP_K = 25


def _points(n=N_CANDIDATES):
    out = []
    for i in range(n):
        out.append(
            {
                "id": f"c{i}",
                "score": 1.0 - i / 100,
                "payload": {
                    "chunk_id": f"c{i}",
                    "chunk_type": "section",
                    "text": f"текст чанка {i}",
                    "page_id": f"p{i}",
                    "page_title": f"Page {i}",
                    "page_version": 1,
                    "block_index": None,
                    "block_part_id": None,
                    "chunk_index": 0,
                    "updated_at": "2024-01-01",
                    "heading_hierarchy": [],
                },
            }
        )
    return out


def _chunk_ids(result):
    return [c["chunk_id"] for s in result["sources"] for c in s["chunks"]]


def _wire(search_deps, n=N_CANDIDATES):
    search_deps.vector_db.search = AsyncMock(return_value=_points(n))
    search_deps.fts_index.search = AsyncMock(return_value=[])


class TestResolveSearchParams:
    def test_absent(self):
        assert resolve_search_params({"vector": "q"}) == ({}, None)

    def test_none(self):
        assert resolve_search_params({"search_params": None}) == ({}, None)

    def test_empty_dict(self):
        assert resolve_search_params({"search_params": {}}) == ({}, None)

    def test_all_keys(self):
        overrides, error = resolve_search_params({
            "search_params": {"top_k": 10, "rrf_k": 40},
        })
        assert error is None
        assert overrides == {"top_k": 10, "rrf_k": 40}

    def test_numeric_string_accepted(self):
        overrides, error = resolve_search_params({"search_params": {"top_k": "10"}})
        assert error is None
        assert overrides == {"top_k": 10}

    def test_integer_float_accepted(self):
        overrides, error = resolve_search_params({"search_params": {"top_k": 10.0}})
        assert error is None
        assert overrides == {"top_k": 10}

    def test_none_values_skipped(self):
        overrides, error = resolve_search_params({"search_params": {"top_k": None, "rrf_k": None}})
        assert error is None
        assert overrides == {}

    def test_not_a_dict(self):
        overrides, error = resolve_search_params({"search_params": "top_k=10"})
        assert overrides == {}
        assert error == "search_params must be an object"

    @pytest.mark.parametrize("bad", ["abc", 0, -3, 2.5, True, ["10"]])
    def test_invalid_top_k(self, bad):
        overrides, error = resolve_search_params({"search_params": {"top_k": bad}})
        assert overrides == {}
        assert error is not None
        assert "search_params.top_k" in error

    @pytest.mark.parametrize("key", ["rrf_k"])
    def test_invalid_int_keys(self, key):
        _, error = resolve_search_params({"search_params": {key: "x"}})
        assert error is not None
        assert key in error

    def test_unknown_key(self):
        overrides, error = resolve_search_params({"search_params": {"topk": 10}})
        assert overrides == {}
        assert "unknown search_params key" in error


class TestTopKOverride:
    async def test_override_changes_vector_limit(self, search_deps):
        _wire(search_deps, n=5)
        result = await execute_algorithm(
            "vector_fts_fusion",
            {"vector": "q", "search_params": {"top_k": 2}},
            search_deps,
        )
        assert search_deps.vector_db.search.await_args.kwargs["limit"] == 2 * POOL_MULTIPLIER
        assert sum(len(s["chunks"]) for s in result["sources"]) == 2

    async def test_string_top_k_applies(self, search_deps):
        _wire(search_deps, n=5)
        await execute_algorithm(
            "vector_fts_fusion",
            {"vector": "q", "search_params": {"top_k": "2"}},
            search_deps,
        )
        assert search_deps.vector_db.search.await_args.kwargs["limit"] == 2 * POOL_MULTIPLIER

    async def test_invalid_top_k_returns_error(self, search_deps):
        _wire(search_deps)
        result = await execute_algorithm(
            "vector_fts_fusion",
            {"vector": "q", "search_params": {"top_k": "abc"}},
            search_deps,
        )
        assert result["sources"] == []
        assert "search_params.top_k" in result["error"]
        search_deps.vector_db.search.assert_not_called()

    async def test_not_a_dict_returns_error(self, search_deps):
        result = await execute_algorithm(
            "vector_fts_fusion",
            {"vector": "q", "search_params": 10},
            search_deps,
        )
        assert result["sources"] == []
        assert "search_params must be an object" in result["error"]

    async def test_unknown_key_returns_error(self, search_deps):
        result = await execute_algorithm(
            "vector_fts_fusion",
            {"vector": "q", "search_params": {"topk": 10}},
            search_deps,
        )
        assert result["sources"] == []
        assert "unknown search_params key" in result["error"]

    async def test_without_search_params_uses_deps_top_k(self, search_deps):
        _wire(search_deps, n=5)
        await execute_algorithm("vector_fts_fusion", {"vector": "q"}, search_deps)
        assert search_deps.vector_db.search.await_args.kwargs["limit"] == (
            search_deps.top_k * POOL_MULTIPLIER
        )

    async def test_shared_deps_not_mutated(self, search_deps):
        _wire(search_deps, n=5)
        await execute_algorithm(
            "vector_fts_fusion",
            {"vector": "q", "search_params": {"top_k": 3, "rrf_k": 7}},
            search_deps,
        )
        assert search_deps.top_k == 25
        assert search_deps.rrf_k == RRF_K


class TestRrfKOverride:
    """Ranking flips between rrf_k=40 (default) and rrf_k=1:
    A: vector rank 0 only            -> 1/(k+0)
    B: vector rank 1, fts rank 0     -> 1/(k+1) + 1/(k+0)
    C: vector rank 2, fts rank 1     -> 1/(k+2) + 1/(k+1)
    k=40: B > C > A;  k=1: B > A > C.
    """

    def _hybrid_points(self):
        base = _points(1)[0]["payload"]
        vec = [
            {"id": "cA", "score": 0.9,
             "payload": base | {"chunk_id": "cA", "page_id": "pA", "page_title": "A"}},
            {"id": "cB", "score": 0.8,
             "payload": base | {"chunk_id": "cB", "page_id": "pB", "page_title": "B"}},
            {"id": "cC", "score": 0.7,
             "payload": base | {"chunk_id": "cC", "page_id": "pC", "page_title": "C"}},
        ]
        fts = [
            {"id": "cB", "score": 10.0, "payload": vec[1]["payload"]},
            {"id": "cC", "score": 9.0, "payload": vec[2]["payload"]},
        ]
        return vec, fts

    async def _order(self, search_deps, search_params=None):
        vec, fts = self._hybrid_points()
        search_deps.vector_db.search = AsyncMock(return_value=vec)
        search_deps.fts_index.search = AsyncMock(return_value=fts)
        params = {"vector": "q", "fts": "q", "fusion": "rrf"}
        if search_params is not None:
            params["search_params"] = search_params
        result = await execute_algorithm("vector_fts_fusion", params, search_deps)
        return _chunk_ids(result)

    async def test_default_rrf_k_ordering(self, search_deps):
        assert await self._order(search_deps) == ["cB", "cC", "cA"]

    async def test_rrf_k_override_flips_ordering(self, search_deps):
        assert await self._order(search_deps, {"rrf_k": 1}) == ["cB", "cA", "cC"]


class TestSchemaExposure:
    def test_all_algorithms_expose_search_params(self):
        descs = get_algorithms()
        assert len(descs) == 4
        for d in descs:
            p = next(x for x in d.parameters if x.name == "search_params")
            assert p.type == "object"
            assert p.required is False
            assert "top_k" in p.description
