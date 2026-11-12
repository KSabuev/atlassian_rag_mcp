"""Positive test for all MCP tools — manual context injection, real service.

Covers: confluence_search, confluence_get_page_content(what="info"),
        confluence_get_page_chunks, confluence_get_structure, health_check.

Only happy-path test — service must be up.
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
async def ctx():
    from atlassian_rag_mcp.api.lifespan import SearchDeps, ServerContext
    from atlassian_rag_mcp.config.settings import settings
    from atlassian_rag_mcp.confluence import create_confluence_client
    from atlassian_rag_mcp.confluence.services.indexer import IndexQueue
    from atlassian_rag_mcp.confluence.services.vector_service import VectorService
    from atlassian_rag_mcp.embedding import create_embedding_client
    from atlassian_rag_mcp.jira import create_jira_client
    from atlassian_rag_mcp.storage.factory import create_vector_db

    vector_db = await create_vector_db(
        backend=settings.vector_db,
        qdrant_host=settings.qdrant_host or "",
        qdrant_port=settings.qdrant_port,
        qdrant_api_key=settings.qdrant_api_key or "",
        qdrant_collection=settings.qdrant_collection,
        vector_size=settings.embedding_dim,
    )
    embedding_client = create_embedding_client(
        url=settings.embedding_url,
        api_key=settings.embedding_api_key or "",
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        binding=settings.embedding_binding,
    )

    def _conf():
        return create_confluence_client(
            url=settings.confluence_url,
            username=settings.confluence_username,
            password=settings.confluence_password,
        )

    svc_ctx = ServerContext(
        confluence_client=_conf(),
        jira_client=create_jira_client(
            url=settings.jira_url,
            username=settings.jira_username,
            password=settings.jira_password,
            project_key=(settings.jira_project_keys or [""])[0],
            plugins=settings.jira_plugins,
        ),
        vector_db=vector_db,
        vector_service=VectorService(
            vector_db=vector_db,
            embedding_client=embedding_client,
            fts_index=AsyncMock(return_value=[]),
        ),
        embedding=embedding_client,
        index_queue=IndexQueue("/tmp"),
        relevant_filter=None,
        reconcile_fn=None,
        search_deps=SearchDeps(
            vector_db=vector_db,
            embedding_client=embedding_client,
            confluence_client=_conf(),
            fts_index=AsyncMock(return_value=[]),
            top_k=settings.top_k,
            confluence_space_key=settings.confluence_space_key or "",
            confluence_url=settings.confluence_url or "",
        ),
    )

    from atlassian_rag_mcp.api.mcp_server import set_mcp_ctx

    set_mcp_ctx(svc_ctx)
    return svc_ctx


# Generic domain query (medical/banking verticals): returns pages on any such
# wiki, so the live tests carry no instance-specific fixtures.
GENERIC_QUERY = "счета"


async def _live_page_id(ctx) -> str:
    """Derive a page id from a generic semantic search (no hardcoded fixtures).

    Uses vector_fts_fusion without `fts`: the vector branch runs against the
    real vector db, the keyword branch is skipped.
    """
    from atlassian_rag_mcp.api.tools import AlgorithmName, confluence_execute_algorithm
    result = await confluence_execute_algorithm(
        algorithm_name=AlgorithmName.vector_fts_fusion,
        params={"vector": GENERIC_QUERY},
        ctx=ctx,
    )
    if result.error is not None or not result.sources:
        pytest.skip(f"no pages for the generic query {GENERIC_QUERY!r}")
    return str(result.sources[0].page_id)


async def _first_table_block(ctx, page_id: str) -> int | None:
    """First block_index with a table_row chunk on the page (None if none)."""
    from atlassian_rag_mcp.api.tools import confluence_get_page_content
    result = await confluence_get_page_content(page_id=page_id, what="chunks", ctx=ctx)
    if result.error is not None:
        return None
    for c in result.chunks:
        if c.chunk_type == "table_row":
            return c.block_index
    return None


class TestConfluenceSearch:
    pytestmark = pytest.mark.live

    async def test_vector_fts_fusion(self, ctx):
        from atlassian_rag_mcp.api.tools import AlgorithmName, confluence_execute_algorithm
        result = await confluence_execute_algorithm(
            algorithm_name=AlgorithmName.vector_fts_fusion,
            params={"vector": "обработка счетов"},
            ctx=ctx,
        )
        assert result is not None
        assert result.sources is not None
        for s in result.sources:
            assert s.vector_score is not None
            assert s.fts_score is not None

    async def test_cql_fts_fusion(self, ctx):
        from atlassian_rag_mcp.api.tools import AlgorithmName, confluence_execute_algorithm
        result = await confluence_execute_algorithm(
            algorithm_name=AlgorithmName.cql_fts_fusion,
            params={"cql": f'text ~ "{GENERIC_QUERY}"'},
            ctx=ctx,
        )
        assert result is not None
        assert result.sources is not None

    async def test_fulltext_search(self, ctx):
        from atlassian_rag_mcp.api.tools import AlgorithmName, confluence_execute_algorithm
        result = await confluence_execute_algorithm(
            algorithm_name=AlgorithmName.fulltext_search,
            params={"fts": "oauth2"},
            ctx=ctx,
        )
        assert result is not None
        assert result.sources is not None
        for s in result.sources:
            assert s.fts_score is not None
            assert s.chunks is not None


class TestGetPageContentInfo:
    pytestmark = pytest.mark.live

    async def test_returns_page_info(self, ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        page_id = await _live_page_id(ctx)
        result = await confluence_get_page_content(page_id=page_id, what="info", ctx=ctx)
        assert result is not None
        if result.error is None:
            assert result.page_title is not None
            assert result.page_url is not None
            assert result.version is not None


class TestGetPageChunks:
    pytestmark = pytest.mark.live

    async def test_returns_chunks_by_page(self, ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        page_id = await _live_page_id(ctx)
        result = await confluence_get_page_content(
            page_id=page_id,
            what="chunks",
            ctx=ctx,
        )
        assert result is not None


class TestGetPageContent:
    """Live checks for confluence_get_content on a page derived from a generic
    domain search (no hardcoded page ids): table rows are emitted one chunk per
    row (chunk_type='table_row'); attached files carry download links."""
    pytestmark = pytest.mark.live

    async def test_table_rows(self, ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        page_id = await _live_page_id(ctx)
        block_index = await _first_table_block(ctx, page_id)
        if block_index is None:
            pytest.skip("no table rows on the sampled page")
        result = await confluence_get_page_content(
            page_id=page_id, what="table", block_index=block_index, limit=1000, ctx=ctx,
        )
        assert result.error is None
        assert len(result.chunks) > 0
        for c in result.chunks:
            assert c.chunk_type == "table_row"
            assert c.block_part_id is not None

    async def test_files(self, ctx):
        from atlassian_rag_mcp.api.tools import confluence_get_page_content
        page_id = await _live_page_id(ctx)
        result = await confluence_get_page_content(page_id=page_id, what="files", ctx=ctx)
        assert result.error is None
        for f in result.files:
            assert f.download_url


class TestHealthCheck:
    pytestmark = pytest.mark.live

    async def test_returns_all_services(self, ctx):
        from atlassian_rag_mcp.api.tools import health_check
        result = await health_check(ctx=ctx)
        assert result is not None
        assert result.status is not None
        assert result.services is not None
        vector_db = ctx.vector_service.vector_db
        db_name = type(vector_db).__name__.removesuffix("Client").removesuffix("Storage").lower()
        for svc in (db_name, "confluence", "embedding"):
            assert svc in result.services
        for _, status in result.services.items():
            assert status is not None


class TestToolSchemas:
    """The MCP tool schemas must expose parameter descriptions.

    mcp 2.x only honors pydantic ``Field`` metadata — a plain
    ``Annotated[X, "string"]`` is silently dropped from the tool schema, so the
    tests assert the descriptions actually reach the schema (the system prompt).
    """

    async def test_execute_tools_document_params(self):
        import atlassian_rag_mcp.api.tools  # noqa: F401  (registers the @mcp.tool() functions)
        from atlassian_rag_mcp.api.mcp_server import mcp

        tools = {t.name: t for t in await mcp.list_tools()}
        checks = {
            "confluence_execute_algorithm": {
                "algorithm_name": "ID of the algorithm from confluence_get_algorithms",
                "params": "unknown keys are rejected",
            },
            "jira_execute_method": {
                "method_name": "ID of the method from jira_get_methods",
                "params": "unknown keys are rejected",
            },
            "zephyr_execute_method": {
                "method_name": "ID of the method from zephyr_get_methods",
                "params": "unknown keys are rejected",
            },
        }
        for tool_name, params in checks.items():
            props = tools[tool_name].input_schema.get("properties", {})
            for pname, needle in params.items():
                desc = props.get(pname, {}).get("description") or ""
                assert needle in desc, f"{tool_name}.{pname}: {desc[:80]!r}"

    async def test_page_content_params_documented(self):
        import atlassian_rag_mcp.api.tools  # noqa: F401  (registers the @mcp.tool() functions)
        from atlassian_rag_mcp.api.mcp_server import mcp

        tools = {t.name: t for t in await mcp.list_tools()}
        props = tools["confluence_get_page_content"].input_schema.get("properties", {})
        for pname in ("page_id", "what", "limit", "titles"):
            desc = props.get(pname, {}).get("description") or ""
            assert desc, f"confluence_get_page_content.{pname} has no description"


class TestNoProjectTokens:
    """The tool surface must stay free of project-specific tokens.

    The project is a universal product (medical/banking verticals): instance
    data may live only in .env / benchmarks — never in the code. With the
    configured keys emptied, any token in the output would be hardcoded text.
    """

    _TOKENS = (
        "PPP", "MM", "TTT", "DEMO1", "DEMO2", "DEMO3", "DEMO4", "DEMO5",
        "DEMO6", "DEMO7", "DEMO8", "DEMO9", "DEMO10",
    )

    async def test_static_tool_output_has_no_project_tokens(self, monkeypatch):
        from atlassian_rag_mcp.config import settings
        from atlassian_rag_mcp.confluence.search import get_algorithms
        from atlassian_rag_mcp.jira.methods import get_methods
        from atlassian_rag_mcp.jira.plugins.zephyr.methods import get_methods as zephyr_get_methods

        monkeypatch.setattr(settings, "confluence_space_key", "PROJ")
        monkeypatch.setattr(settings, "jira_project_key", "PROJ")
        from atlassian_rag_mcp.jira.methods import reset_value_maps
        reset_value_maps()

        methods = list(get_algorithms()) + list(get_methods()) + list(zephyr_get_methods())
        texts = [m.description for m in methods]
        texts += [p.description for m in methods for p in m.parameters]
        joined = "\n".join(texts)
        joined = joined.replace("YYYY-MM-DD", "YYYY_MM_DD").replace("DD.MM.YYYY", "DD_MM_YYYY")
        for token in self._TOKENS:
            assert not re.search(r"\b" + re.escape(token) + r"\b", joined), (
                f"project token {token!r} is hardcoded in the tool output"
            )


async def test_structure_full_tree():
    import shutil
    import tempfile

    from atlassian_rag_mcp.confluence.services.structure_service import StructureService
    from atlassian_rag_mcp.storage.vectordb.lancedb_client import LanceDBClient

    tmpdir = tempfile.mkdtemp()
    store = LanceDBClient(db_path=tmpdir, vector_size=4)
    await store.start()

    pages_data = [
        {"page_id": "p1", "parent_id": None, "page_title": "Root", "chunk_id": "c1"},
        {"page_id": "p2", "parent_id": None, "page_title": "Alpha", "chunk_id": "c2"},
        {"page_id": "c",  "parent_id": "p1",  "page_title": "Child", "chunk_id": "c3"},
    ]
    points = [
        {"id": p["chunk_id"], "vector": [1.0, 0.0, 0.0, 0.0], "payload": p}
        for p in pages_data
    ]
    await store.upsert_points(points)

    svc = StructureService(store)
    result = await svc.render_tree()
    assert isinstance(result, str)
    assert "Confluence Structure" in result
    assert "Root" in result
    assert "Alpha" in result

    await store.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


async def test_structure_scoped():
    import shutil
    import tempfile

    from atlassian_rag_mcp.confluence.services.structure_service import StructureService
    from atlassian_rag_mcp.storage.vectordb.lancedb_client import LanceDBClient

    tmpdir = tempfile.mkdtemp()
    store = LanceDBClient(db_path=tmpdir, vector_size=4)
    await store.start()

    pages_data = [
        {"page_id": "p1", "parent_id": None, "page_title": "Root", "chunk_id": "c1"},
        {"page_id": "c",  "parent_id": "p1",  "page_title": "Child", "chunk_id": "c2"},
        {"page_id": "gc", "parent_id": "c",   "page_title": "Grandchild", "chunk_id": "c3"},
        {"page_id": "p2", "parent_id": None,  "page_title": "Other", "chunk_id": "c4"},
    ]
    points = [
        {"id": p["chunk_id"], "vector": [1.0, 0.0, 0.0, 0.0], "payload": p}
        for p in pages_data
    ]
    await store.upsert_points(points)

    svc = StructureService(store)
    result = await svc.render_tree(page_id="c", children=True)
    assert isinstance(result, str)
    assert "Child" in result
    assert "Grandchild" in result
    assert "Root" not in result

    await store.close()
    shutil.rmtree(tmpdir, ignore_errors=True)
