"""Unit tests for MCP Confluence tools — mocks only, no live services."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlassian_rag_mcp.api.lifespan import ServerContext
from atlassian_rag_mcp.api.mcp_server import set_mcp_ctx
from atlassian_rag_mcp.confluence.models import Page, PageContent


@pytest.fixture
def mock_ctx():
    ctx = object.__new__(ServerContext)
    ctx.confluence_client = AsyncMock()
    ctx.confluence_client.get_page = AsyncMock(return_value=PageContent(
        page_id="123",
        page_title="Test Page",
        version=3,
        last_updated="2024-01-15",
        html="<p>hello</p>",
    ))
    ctx.jira_client = MagicMock()
    ctx.vector_db = MagicMock()
    ctx.vector_service = AsyncMock()
    ctx.vector_service.get_chunks_by_filter = AsyncMock(return_value=[
        {"chunk_id": "c1", "text": "chunk1", "chunk_type": "text",
         "page_id": "123", "page_title": "Test Page", "chunk_index": 0},
    ])
    ctx.embedding = MagicMock()
    ctx.index_queue = AsyncMock()
    ctx.index_queue.get_in_work_versions = AsyncMock(return_value={})
    ctx.relevant_filter = None
    ctx.search_deps = MagicMock()
    ctx.search_deps.confluence_url = "https://c.example.com/"
    ctx.search_deps.confluence_space_key = "TEST"
    return ctx


@pytest.fixture
def setup_mcp(mock_ctx):
    from atlassian_rag_mcp.api import mcp_server as mcp_module

    original = mcp_module._mcp_ctx
    set_mcp_ctx(mock_ctx)
    yield
    set_mcp_ctx(original)




async def test_get_algorithms_returns_at_least_three():
    from atlassian_rag_mcp.api.tools import confluence_get_algorithms
    result = await confluence_get_algorithms()
    assert len(result) >= 3


async def test_get_algorithms_includes_all_ids():
    from atlassian_rag_mcp.api.tools import confluence_get_algorithms
    result = await confluence_get_algorithms()
    algo_ids = [a.id for a in result]
    assert "fulltext_search" in algo_ids
    assert "vector_fts_fusion" in algo_ids
    assert "cql_fts_fusion" in algo_ids


async def test_get_algorithms_descriptions_are_meaningful():
    from atlassian_rag_mcp.api.tools import confluence_get_algorithms
    result = await confluence_get_algorithms()
    for a in result:
        assert len(a.description) > 30, f"{a.id} description too short"
    desc_map = {a.id: a.description.lower() for a in result}
    assert "fts" in desc_map["fulltext_search"] or "lexical" in desc_map["fulltext_search"]
    assert "fusion" in desc_map["vector_fts_fusion"]
    assert "cql" in desc_map["cql_fts_fusion"]


async def test_get_algorithms_params_have_descriptions():
    from atlassian_rag_mcp.api.tools import confluence_get_algorithms
    result = await confluence_get_algorithms()
    for a in result:
        for p in a.parameters:
            assert p.description, f"{a.id}.{p.name} has empty description"
            assert len(p.description) > 10, f"{a.id}.{p.name} description too short"


async def test_get_algorithms_cql_syntax_in_param_description():
    from atlassian_rag_mcp.api.tools import confluence_get_algorithms
    result = await confluence_get_algorithms()
    for a in result:
        for p in a.parameters:
            if p.name == "cql":
                assert "AND" in p.description or "and" in p.description
                assert "\"" in p.description


async def test_get_algorithms_cql_fts_fusion_cql_branch_only():
    from atlassian_rag_mcp.api.tools import confluence_get_algorithms
    result = await confluence_get_algorithms()
    algo = next(a for a in result if a.id == "cql_fts_fusion")
    assert "apply to the CQL branch only" in algo.description


async def test_get_algorithms_no_pagination_or_truncation():
    from atlassian_rag_mcp.api.tools import confluence_get_algorithms
    result = await confluence_get_algorithms()
    assert len(result) == 4
    assert "cql_search" in {a.id for a in result}


def test_algorithm_routing_attributes():
    from atlassian_rag_mcp.confluence.search.base import algorithms

    for name in ("cql_fts_fusion", "fulltext_search", "vector_fts_fusion"):
        routing = getattr(algorithms.get(name), "routing")
        assert isinstance(routing, str) and routing, f"{name}: routing missing"
    assert getattr(algorithms.get("cql_search"), "routing") is None


def test_execute_description_composed_from_registry():
    from atlassian_rag_mcp.api.tools.confluence import _execute_description

    desc = _execute_description()
    assert "PICK BY QUERY TYPE" in desc
    for name in ("cql_fts_fusion", "fulltext_search", "vector_fts_fusion"):
        assert f"-> {name}" in desc, f"{name} missing from the routing table"
    # cql_search (routing=None) is absent from the table
    table = desc.split("RULES", 1)[0]
    assert "cql_search" not in table
    # the only generic rule is the cross-algorithm merge line; everything
    # per-algorithm (score scale, omitted chunks, spaces, search_params)
    # lives in confluence_get_algorithms and must not be duplicated here
    assert "run the other two and merge" in desc
    for needle in ("search_params", "[omitted] full text", "NOT comparable",
                   "overall leader", "Configured spaces:"):
        assert needle not in desc, f"duplicated per-algorithm text: {needle}"


def test_execute_description_rows_follow_registry_order():
    from atlassian_rag_mcp.api.tools.confluence import _execute_description
    from atlassian_rag_mcp.confluence.search import get_algorithms

    desc = _execute_description()
    table = desc.split("PICK BY QUERY TYPE", 1)[1].split("RULES", 1)[0]
    rows = [line for line in table.splitlines() if "->" in line]
    ids_in_table = [line.rsplit("->", 1)[1].strip() for line in rows]
    expected = [a.id for a in get_algorithms() if f"-> {a.id}" in table]
    assert ids_in_table == expected
    assert len(ids_in_table) == 3


def test_get_algorithms_spaces_example_from_configured_keys(monkeypatch):
    from atlassian_rag_mcp.config.settings import settings
    from atlassian_rag_mcp.confluence.search import get_algorithms

    monkeypatch.setattr(settings, "confluence_space_key", "AAA,DDD")
    result = get_algorithms()
    for a in result:
        p = next(p for p in a.parameters if p.name in ("space", "spaces"))
        assert '["AAA", "DDD"]' in p.description, f"{a.id}: {p.description}"


def test_get_algorithms_spaces_example_single_key(monkeypatch):
    from atlassian_rag_mcp.config.settings import settings
    from atlassian_rag_mcp.confluence.search import get_algorithms

    monkeypatch.setattr(settings, "confluence_space_key", "AAA")
    result = get_algorithms()
    for a in result:
        p = next(p for p in a.parameters if p.name in ("space", "spaces"))
        assert '["AAA"]' in p.description, f"{a.id}: {p.description}"




async def test_execute_returns_sources(setup_mcp):
    from atlassian_rag_mcp.api.tools import AlgorithmName, confluence_execute_algorithm
    from atlassian_rag_mcp.confluence.search.vector_fts_fusion import VectorFtsFusionSource

    with patch("atlassian_rag_mcp.api.tools.confluence.execute_algorithm") as mock_exec:
        mock_exec.return_value = {
            "sources": [
                {
                    "page_id": "123", "page_title": "Found", "page_url": "http://ex/",
                    "version": 3, "score": 0.95,
                    "chunks": [{"chunk_id": "c1", "text": "match", "chunk_type": "section"}],
                }
            ],
            "error": None,
        }
        result = await confluence_execute_algorithm(
            algorithm_name=AlgorithmName.vector_fts_fusion,
            params={"vector": "test"},
        )
    assert result.error is None
    assert len(result.sources) == 1
    assert result.sources[0].page_title == "Found"
    assert isinstance(result.sources[0], VectorFtsFusionSource)
    assert result.sources[0].chunks[0].chunk_id == "c1"


async def test_execute_handles_exception(setup_mcp):
    from atlassian_rag_mcp.api.tools import AlgorithmName, confluence_execute_algorithm

    with patch("atlassian_rag_mcp.api.tools.confluence.execute_algorithm") as mock_exec:
        mock_exec.side_effect = ConnectionError("confluence timeout")
        result = await confluence_execute_algorithm(
            algorithm_name=AlgorithmName.cql_fts_fusion,
            params={"cql": "test"},
        )
    assert result.error is not None
    assert "confluence timeout" in result.error


async def test_execute_empty_sources(setup_mcp):
    from atlassian_rag_mcp.api.tools import AlgorithmName, confluence_execute_algorithm

    with patch("atlassian_rag_mcp.api.tools.confluence.execute_algorithm") as mock_exec:
        mock_exec.return_value = {"sources": [], "error": None}
        result = await confluence_execute_algorithm(
            algorithm_name=AlgorithmName.vector_fts_fusion,
            params={"vector": "x"},
        )
    assert result.error is None
    assert result.sources == []


async def test_execute_cql_returns_excerpt_sources(setup_mcp):
    from atlassian_rag_mcp.api.tools import AlgorithmName, confluence_execute_algorithm
    from atlassian_rag_mcp.confluence.search.cql_fts_fusion import CqlFtsFusionSource

    with patch("atlassian_rag_mcp.api.tools.confluence.execute_algorithm") as mock_exec:
        mock_exec.return_value = {
            "sources": [
                {
                    "page_id": "123", "page_title": "Found", "page_url": "http://ex/",
                    "version": 3, "score": 0.95, "space_key": "TEST",
                    "excerpt": "some text",
                }
            ],
            "error": None,
        }
        result = await confluence_execute_algorithm(
            algorithm_name=AlgorithmName.cql_fts_fusion,
            params={"cql": "test"},
        )
    assert result.error is None
    assert isinstance(result.sources[0], CqlFtsFusionSource)
    assert result.sources[0].excerpt == "some text"




async def test_get_page_html_returns_cleaned_content(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_page_content

    mock_ctx.confluence_client.get_page.return_value = PageContent(
        page_id="123",
        page_title="Test Page",
        version=3,
        html="<p>Hello <b>world</b></p>",
    )
    result = await confluence_get_page_content(page_id="123", what="html")
    assert result.error is None
    assert result.page_title == "Test Page"
    assert result.html is not None
    assert "Hello" in result.html
    assert "world" in result.html


async def test_get_page_chunks_returns_chunks(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_page_content

    mock_ctx.confluence_client.get_page.return_value = PageContent(
        page_id="123", page_title="Test Page", version=3,
    )
    mock_ctx.vector_service.get_chunks_by_filter.return_value = [
        {"chunk_id": "c1", "text": "aaa", "chunk_type": "text",
         "page_id": "123", "page_title": "Test Page", "chunk_index": 0},
        {"chunk_id": "c2", "text": "bbb", "chunk_type": "text",
         "page_id": "123", "page_title": "Test Page", "chunk_index": 1},
    ]
    result = await confluence_get_page_content(page_id="123", what="chunks")
    assert result.error is None
    assert len(result.chunks) == 2
    assert result.chunks[0].text == "aaa"
    assert result.chunks[1].text == "bbb"


async def test_get_page_html_not_found(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_page_content

    mock_ctx.confluence_client.get_page.return_value = Page(error="Page 999 not found")
    result = await confluence_get_page_content(page_id="999", what="html")
    assert result.error is not None
    assert "not found" in result.error.lower()


async def test_get_page_chunks_not_found(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_page_content

    mock_ctx.confluence_client.get_page.return_value = Page(error="Page 999 not found")
    result = await confluence_get_page_content(page_id="999", what="chunks")
    assert result.error is not None
    assert "not found" in result.error.lower()


async def test_get_page_invalid_what(setup_mcp):
    from atlassian_rag_mcp.api.tools import confluence_get_page_content

    result = await confluence_get_page_content(page_id="123", what="xyz")
    assert result.error is not None
    assert "unknown" in result.error.lower() or "xyz" in result.error


async def test_get_page_handles_client_exception(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_page_content

    mock_ctx.confluence_client.get_page.side_effect = ConnectionError("no network")
    result = await confluence_get_page_content(page_id="123", what="html")
    assert result.error is not None
    assert "no network" in result.error




async def test_get_page_content_info_returns_metadata(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_page_content

    mock_ctx.confluence_client.get_page.return_value = Page(
        page_id="123",
        page_title="Info Page",
        version=5,
        last_updated="2024-06-01",
    )
    result = await confluence_get_page_content(page_id="123", what="info")
    assert result.error is None
    assert result.page_title == "Info Page"
    assert result.version == 5
    assert result.last_updated == "2024-06-01"
    assert result.chunks == []
    assert result.html is None


async def test_get_page_content_info_not_found(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_page_content

    mock_ctx.confluence_client.get_page.return_value = Page(error="Page 999 not found")
    result = await confluence_get_page_content(page_id="999", what="info")
    assert result.error is not None
    assert "not found" in result.error.lower()


async def test_get_page_content_info_missing_version(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_page_content

    mock_ctx.confluence_client.get_page.return_value = Page(
        page_id="123", page_title="Minimal",
    )
    result = await confluence_get_page_content(page_id="123", what="info")
    assert result.error is None
    assert result.page_title == "Minimal"
    assert result.version == 0
    assert result.last_updated is None


async def test_get_page_content_info_client_exception(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_page_content

    mock_ctx.confluence_client.get_page.side_effect = TimeoutError("timed out")
    result = await confluence_get_page_content(page_id="123", what="info")
    assert result.error is not None
    assert "timed out" in result.error




async def test_get_structure_returns_tree(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_structure

    with patch("atlassian_rag_mcp.api.tools.confluence.StructureService") as MockStruct:
        svc = AsyncMock()
        svc.render_tree.return_value = "# Confluence Structure\n- Root\n  - Child"
        MockStruct.return_value = svc

        result = await confluence_get_structure()
        assert "Confluence Structure" in result
        assert "Root" in result

        MockStruct.assert_called_once_with(
            mock_ctx.vector_db, space_key="TEST",
        )


async def test_get_structure_with_children(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_structure

    with patch("atlassian_rag_mcp.api.tools.confluence.StructureService") as MockStruct:
        svc = AsyncMock()
        svc.render_tree.return_value = "- Child\n  - Grandchild"
        MockStruct.return_value = svc

        result = await confluence_get_structure(
            page_id="c", children=True,
        )
        assert "Child" in result

        svc.render_tree.assert_called_once_with(
            page_id="c", children=True, parents=False,
        )


async def test_get_structure_with_parents(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_structure

    with patch("atlassian_rag_mcp.api.tools.confluence.StructureService") as MockStruct:
        svc = AsyncMock()
        svc.render_tree.return_value = "- Root\n  - Child"
        MockStruct.return_value = svc

        result = await confluence_get_structure(
            page_id="c", parents=True,
        )
        assert "Root" in result

        svc.render_tree.assert_called_once_with(
            page_id="c", children=False, parents=True,
        )


async def test_get_structure_handles_exception(setup_mcp, mock_ctx):
    from atlassian_rag_mcp.api.tools import confluence_get_structure

    with patch("atlassian_rag_mcp.api.tools.confluence.StructureService") as MockStruct:
        svc = AsyncMock()
        svc.render_tree.side_effect = RuntimeError("db error")
        MockStruct.return_value = svc

        result = await confluence_get_structure()
        assert "db error" in result
        assert "Failed to get structure" in result




def _chunk_dict(chunk_index, text, chunk_type="table_row", block_index=None, block_part_id=None, hh=None):
    return {
        "chunk_id": f"c{chunk_index}", "text": text, "chunk_type": chunk_type,
        "page_id": "123", "page_title": "Test Page", "chunk_index": chunk_index,
        "block_index": block_index, "block_part_id": block_part_id, "heading_hierarchy": hh or [],
    }


class TestGetPageContent:
    async def test_what_chunks_returns_chunks(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        mock_ctx.vector_service.get_chunks_by_filter.return_value = [
            _chunk_dict(0, "chunk one"), _chunk_dict(1, "chunk two"),
        ]
        result = await confluence_get_page_content(page_id="123", what="chunks")
        assert result.error is None
        assert len(result.chunks) == 2
        assert result.chunks[1].text == "chunk two"
        assert result.page_title == "Test Page"
        assert result.page_url == "https://c.example.com/pages/viewpage.action?pageId=123"
        mock_ctx.vector_service.get_chunks_by_filter.assert_awaited_once_with(
            page_id="123", block_index=None, limit=1000, offset=0,
            in_work_versions={},
        )

    async def test_what_chunks_passes_limit_offset(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        result = await confluence_get_page_content(page_id="123", what="chunks", limit=10, offset=20)
        mock_ctx.vector_service.get_chunks_by_filter.assert_awaited_once_with(
            page_id="123", block_index=None, limit=10, offset=20,
            in_work_versions={},
        )
        assert result.error is None

    async def test_what_table_requires_table_index(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        result = await confluence_get_page_content(page_id="123", what="table")
        assert result.error is not None
        assert "block_index" in result.error
        mock_ctx.vector_service.get_chunks_by_filter.assert_not_awaited()

    async def test_what_table_passes_table_index(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        mock_ctx.vector_service.get_chunks_by_filter.return_value = [
            _chunk_dict(0, "row 1", block_index=3, block_part_id=1),
            _chunk_dict(1, "row 2", block_index=3, block_part_id=2),
        ]
        result = await confluence_get_page_content(page_id="123", what="table", block_index=3)
        assert result.error is None
        assert len(result.chunks) == 2
        mock_ctx.vector_service.get_chunks_by_filter.assert_awaited_once_with(
            page_id="123", block_index=3, limit=1000, offset=0,
            in_work_versions={},
        )

    async def test_chunk_index_selects_single_chunk(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        mock_ctx.vector_service.get_chunks_by_filter.return_value = [
            _chunk_dict(0, "a"), _chunk_dict(1, "b"), _chunk_dict(2, "c"),
        ]
        result = await confluence_get_page_content(page_id="123", what="chunks", chunk_index=1)
        assert result.error is None
        assert len(result.chunks) == 1
        assert result.chunks[0].text == "b"

    async def test_chunk_index_not_found_returns_empty(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        mock_ctx.vector_service.get_chunks_by_filter.return_value = [
            _chunk_dict(0, "a"), _chunk_dict(1, "b"),
        ]
        result = await confluence_get_page_content(page_id="123", what="chunks", chunk_index=99)
        assert result.error is None
        assert result.chunks == []

    async def test_what_files_lists_attachments(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        mock_ctx.confluence_client.list_attachments = AsyncMock(return_value=[
            {
                "title": "file_v1.0.xsd", "media_type": "application/xml",
                "size": 100, "version_number": 2, "created": "2024-01-01",
                "download_url": "https://c.example.com/download/attachments/123/file_v1.0.xsd",
            },
        ])
        result = await confluence_get_page_content(page_id="123", what="files")
        assert result.error is None
        assert len(result.files) == 1
        assert result.files[0].title == "file_v1.0.xsd"
        assert result.files[0].download_url.endswith("/123/file_v1.0.xsd")
        mock_ctx.confluence_client.list_attachments.assert_awaited_once_with("123")

    async def test_what_files_empty(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        mock_ctx.confluence_client.list_attachments = AsyncMock(return_value=[])
        result = await confluence_get_page_content(page_id="123", what="files")
        assert result.error is None
        assert result.files == []

    async def test_page_not_found(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        mock_ctx.confluence_client.get_page.return_value = Page(error="Page 999 not found")
        result = await confluence_get_page_content(page_id="999", what="chunks")
        assert result.error is not None
        assert "not found" in result.error.lower()

    async def test_unknown_what(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        result = await confluence_get_page_content(page_id="123", what="bogus")
        assert result.error is not None
        assert "bogus" in result.error

    async def test_handler_exception_returns_error(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        mock_ctx.confluence_client.get_page.side_effect = ConnectionError("no network")
        result = await confluence_get_page_content(page_id="123", what="chunks")
        assert result.error is not None
        assert "no network" in result.error

    async def test_files_api_error_returned_as_error(self, setup_mcp, mock_ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        mock_ctx.confluence_client.list_attachments = AsyncMock(
            side_effect=RuntimeError("attachments api down")
        )
        result = await confluence_get_page_content(page_id="123", what="files")
        assert result.error is not None
        assert "attachments api down" in result.error




async def test_algorithm_name_matches_descriptors():
    from atlassian_rag_mcp.api.tools import confluence_get_algorithms
    from atlassian_rag_mcp.api.tools.confluence import AlgorithmName

    algos = await confluence_get_algorithms()
    enum_names = set(e.name for e in AlgorithmName)
    algo_ids = set(a.id for a in algos)
    assert enum_names == algo_ids
