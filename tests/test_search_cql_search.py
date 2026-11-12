"""cql_search: pure CQL branch (diagnostics / builder verification).

No FTS branch, no fusion, no relevance filtering — pages in Confluence's own
CQL order. Tested: built CQL in the client call, cql_rank, param validation,
empty input, and the debug log of the built query.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from atlassian_rag_mcp.confluence.models import CqlResult
from atlassian_rag_mcp.confluence.search import execute_algorithm
from atlassian_rag_mcp.confluence.search.cql_search import CqlSearch


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


async def _run(params, deps):
    return await execute_algorithm("cql_search", params, deps)


class TestDescriptor:
    def test_registered_and_diagnostic(self):
        assert CqlSearch.skip_irrelevant is True
        d = CqlSearch.descriptor()
        assert d.id == "cql_search"
        names = {p.name for p in d.parameters}
        assert names == {
            "cql", "title", "label", "creator", "contributor", "space",
            "created", "lastmodified", "id", "ancestor", "parent", "search_params",
        }
        # no fusion-specific params
        assert not ({"fusion", "w_kw", "title_boost"} & names)

    def test_description_marks_diagnostic(self):
        d = CqlSearch.descriptor()
        assert "DIAGNOSTIC" in d.description

    def test_date_params_document_sign_semantics(self):
        """Live check: CQL relative dates work ONLY inside now(); - = the last N days;
        a bare -7d as a date value is rejected (400). No future (+) offsets documented."""
        d = CqlSearch.descriptor()
        params = {p.name: p for p in d.parameters}
        for name in ("created", "lastmodified"):
            desc = params[name].description
            assert 'now("-7d")' in desc, name
            assert "+7d" not in desc, name
            assert "- = the last N days" in desc, name
            assert "ONLY inside now()" in desc, name

    def test_cql_filter_params_shared_with_fts_fusion(self):
        """One fact — one home: the CQL filter parameter descriptions are a single
        source (cql_builder.cql_filter_params) shared by both CQL algorithms, so
        cql_search and cql_fts_fusion can never drift apart."""
        from atlassian_rag_mcp.confluence.search.cql_fts_fusion import CqlFtsFusionSearch

        d1 = CqlSearch.descriptor()
        d2 = CqlFtsFusionSearch.descriptor()
        p1 = {p.name: p for p in d1.parameters}
        p2 = {p.name: p for p in d2.parameters}
        for name in ("title", "label", "creator", "contributor", "created",
                     "lastmodified", "id", "ancestor", "parent"):
            assert p1[name].description == p2[name].description, name
        # space: same base, the fusion algorithm adds the branch-scope note
        assert "Omit to search across all spaces" in p1["space"].description
        assert "Applies to the CQL branch query and to the FTS branch" in p2["space"].description

    def test_cql_date_help_is_a_single_copy(self):
        """The CQL date semantics text (CQL_DATE_HELP) appears exactly twice per
        algorithm — in the created and lastmodified params and nowhere else."""
        from atlassian_rag_mcp.confluence.cql_builder import CQL_DATE_HELP
        from atlassian_rag_mcp.confluence.search.cql_fts_fusion import CqlFtsFusionSearch

        for d in (CqlSearch.descriptor(), CqlFtsFusionSearch.descriptor()):
            hits = [p.name for p in d.parameters if CQL_DATE_HELP in p.description]
            assert hits == ["created", "lastmodified"], d.id

    def test_cql_syntax_help_once_per_algorithm(self):
        """CQL_SYNTAX_HELP lives only in the `cql` parameter of each CQL algorithm."""
        from atlassian_rag_mcp.confluence.cql_builder import CQL_SYNTAX_HELP
        from atlassian_rag_mcp.confluence.search.cql_fts_fusion import CqlFtsFusionSearch

        for d in (CqlSearch.descriptor(), CqlFtsFusionSearch.descriptor()):
            hits = [p.name for p in d.parameters if CQL_SYNTAX_HELP in p.description]
            assert hits == ["cql"], d.id


class TestBuiltCql:
    """The CQL string sent to Confluence is exactly what the builder produced."""

    async def test_raw_cql_verbatim(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[]))
        await _run({"cql": 'title ~ "x"'}, search_deps)
        assert search_deps.confluence_client.cql.await_args.kwargs["cql"] == \
            'type = "page" AND title ~ "x"'

    async def test_filters_only(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[]))
        await _run({"space": ["PROJ"], "title": "roadmap"}, search_deps)
        assert search_deps.confluence_client.cql.await_args.kwargs["cql"] == \
            'type = "page" AND space = "PROJ" AND title ~ "roadmap"'

    async def test_id_ancestor_parent(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[]))
        await _run({"id": ["123", "456"], "ancestor": "789", "parent": "101"}, search_deps)
        assert search_deps.confluence_client.cql.await_args.kwargs["cql"] == \
            'type = "page" AND id IN (123,456) AND ancestor = 789 AND parent = 101'

    async def test_mixed_cql_and_filters(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[]))
        await _run({"cql": "api", "space": ["DEV"], "lastmodified": "2025-01-01"}, search_deps)
        assert search_deps.confluence_client.cql.await_args.kwargs["cql"] == \
            'type = "page" AND space = "DEV" AND lastmodified >= "2025-01-01" AND text ~ "api"'

    async def test_builtin_query_logged_at_debug(self, search_deps, caplog):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[]))
        with caplog.at_level("DEBUG"):
            await _run({"space": ["PROJ"], "title": "roadmap"}, search_deps)
        assert any(
            'cql_search: built CQL query: \'type = "page" AND space = "PROJ" AND title ~ "roadmap"\''
            in rec.message
            for rec in caplog.records
        )


class TestResults:
    async def test_confluence_order_and_cql_rank(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[
            _cql_item("c1", "C1", "PROJ"),
            _cql_item("c2", "C2", "PROJ"),
            _cql_item("c3", "C3", "PROJ"),
        ]))
        result = await _run({"cql": "q"}, search_deps)
        assert [s["page_id"] for s in result["sources"]] == ["c1", "c2", "c3"]
        assert [s["cql_rank"] for s in result["sources"]] == [1, 2, 3]
        assert "score" not in result["sources"][0]
        assert "chunks" not in result["sources"][0]

    async def test_excerpt_kept(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[
            _cql_item("c1", "C1", "PROJ", excerpt="<p>cql excerpt</p>"),
        ]))
        result = await _run({"cql": "q"}, search_deps)
        assert "cql excerpt" in (result["sources"][0]["excerpt"] or "")

    async def test_no_fts_calls(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[
            _cql_item("c1", "C1", "PROJ"),
        ]))
        search_deps.fts_index.search = AsyncMock()
        await _run({"cql": "q"}, search_deps)
        assert search_deps.fts_index.search.await_count == 0

    async def test_unresolved_space_key_is_error(self, search_deps):
        item = _cql_item("c1", "C1", "")
        item["content"]["_expandable"] = {}
        search_deps.confluence_client.cql = AsyncMock(return_value=CqlResult(results=[item]))
        result = await _run({"cql": "q"}, search_deps)
        assert result["sources"] == []
        assert "space_key" in (result.get("error") or "")


class TestParamValidation:
    async def test_empty_params_is_error(self, search_deps):
        search_deps.confluence_client.cql = AsyncMock()
        result = await _run({}, search_deps)
        assert result["sources"] == []
        assert "No CQL query" in (result.get("error") or "")
        assert search_deps.confluence_client.cql.await_count == 0

    async def test_unknown_cql_field_raises(self, search_deps):
        """Builder validation errors propagate (the MCP layer turns them into
        {"sources": [], "error": ...}) — same behavior as cql_fts_fusion."""
        search_deps.confluence_client.cql = AsyncMock()
        with pytest.raises(ValueError, match="Unknown CQL field"):
            await _run({"cql": 'status = "X"'}, search_deps)
        assert search_deps.confluence_client.cql.await_count == 0

    @pytest.mark.parametrize("params", [
        {"ancestor": "abc"},
        {"id": ["12a"]},
        {"parent": "12345x"},
    ])
    async def test_non_numeric_page_id_raises(self, search_deps, params):
        search_deps.confluence_client.cql = AsyncMock()
        with pytest.raises(ValueError, match="numeric page id"):
            await _run(params, search_deps)
        assert search_deps.confluence_client.cql.await_count == 0

    @pytest.mark.parametrize("params,error_part", [
        ({"cql": "q", "titel": "x"}, "titel"),
        ({"cql": "q", "fusion": "rrf"}, "fusion"),
    ])
    async def test_invalid_params_return_error(self, search_deps, params, error_part):
        search_deps.confluence_client.cql = AsyncMock()
        result = await _run(params, search_deps)
        assert result["sources"] == []
        assert error_part in (result.get("error") or "")
        assert search_deps.confluence_client.cql.await_count == 0
