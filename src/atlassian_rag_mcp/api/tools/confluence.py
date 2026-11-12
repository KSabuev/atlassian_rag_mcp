"""Confluence MCP tools (async).

``SearchResultUnion`` is the union of the base result and every registered
algorithm's result model — built from the registry, so adding an algorithm
requires no change here.
"""

from __future__ import annotations

from enum import Enum
from functools import reduce
from operator import or_
from typing import Annotated, Literal

from mcp.server.mcpserver import Context
from pydantic import Field

from atlassian_rag_mcp.api.mcp_server import _get_svc, mcp
from atlassian_rag_mcp.confluence.models import PageContent
from atlassian_rag_mcp.confluence.page_content import execute_page_content
from atlassian_rag_mcp.confluence.search import (
    execute_algorithm,
    get_algorithms,
)
from atlassian_rag_mcp.confluence.search.base import SearchResult, algorithms
from atlassian_rag_mcp.confluence.services.structure_service import StructureService
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.registry import (
    DICT_PARAMS_DESC,
    MethodDescriptor,
    error_message,
)

_algo_ids = [a.id for a in get_algorithms()]
AlgorithmName = Enum("AlgorithmName", {name: name for name in _algo_ids})

SearchResultUnion = reduce(
    or_, [SearchResult] + [algorithms[i].result_model for i in algorithms.ids()]
)


def _execute_description() -> str:
    """Compose the confluence_execute_algorithm tool description from the registry.

    Every algorithm whose ``routing`` attribute is set contributes one row to the
    "PICK BY QUERY TYPE" table (the algorithm id is appended from the registry, so
    it cannot drift from the actual algorithm name). Per-algorithm guidance lives
    ONLY here as routing rows — the descriptors returned by confluence_get_algorithms
    carry parameter semantics and result rules, never a repetition of the routing
    text (contract enforced in test_execute_description_composed_from_registry).
    """
    rows = [
        f"  {cls.routing} -> {a.id}"
        for a in get_algorithms()
        if (cls := algorithms.get(a.id)) is not None and getattr(cls, "routing", None)
    ]
    table = "\n".join(rows)
    return f"""Execute a Confluence search algorithm by ID.

PICK BY QUERY TYPE:
{table}

RULES
- If one algorithm misses, run the other two and merge: they fail on opposite
  query types.
"""


_CONFLUENCE_EXECUTE_DESCRIPTION = _execute_description()

@mcp.tool()
async def confluence_get_algorithms(
    ctx: Context | None = None,
) -> list[MethodDescriptor]:
    """List all available Confluence search algorithms with their parameter schemas."""
    return get_algorithms()


@mcp.tool(description=_CONFLUENCE_EXECUTE_DESCRIPTION)
async def confluence_execute_algorithm(
    algorithm_name: Annotated[
        AlgorithmName,
        Field(description="ID of the algorithm from confluence_get_algorithms (e.g. cql_fts_fusion)."),
    ],
    params: Annotated[
        dict,
        Field(
            description=(
                DICT_PARAMS_DESC
                + " Any algorithm also accepts an optional \"search_params\" "
                "object with per-request constants: {\"top_k\": int, \"rrf_k\": int}."
            )
        ),
    ],
    ctx: Context | None = None,
) -> SearchResultUnion:
    """Execute a Confluence search algorithm by ID (routing table: tool description)."""
    logger.debug(f"confluence_execute_algorithm: {algorithm_name.value}, params={params}")

    try:
        svc = _get_svc(ctx)
        cls = algorithms.get(algorithm_name.value)
        result = await execute_algorithm(
            algorithm_name.value, params, svc.search_deps,
            relevant_fn=svc.relevant_filter,
        )
        return (cls.result_model if cls is not None else SearchResult)(**result)

    except Exception as e:
        logger.error(f"confluence_execute_algorithm failed: {e}", exc_info=True)
        return SearchResult(sources=[], error=error_message(e))


@mcp.tool()
async def confluence_get_structure(
    page_id: Annotated[
        str | None,
        Field(
            description="Confluence page ID. If not set, returns full tree. "
                        "If set with children/parents, returns scoped subtree"
        ),
    ] = None,
    children: Annotated[
        bool,
        Field(description="Include all descendants of page_id. Requires page_id"),
    ] = False,
    parents: Annotated[
        bool,
        Field(description="Include all ancestors from root to page_id. Requires page_id"),
    ] = False,
    ctx: Context | None = None,
) -> str:
    """Get Confluence page tree structure as Markdown from indexed metadata."""
    logger.debug(
        f"confluence_get_structure: page_id={page_id}, "
        f"children={children}, parents={parents}"
    )

    try:
        svc = _get_svc(ctx)

        service = StructureService(svc.vector_db, space_key=svc.search_deps.confluence_space_key)
        return await service.render_tree(
            page_id=page_id,
            children=children,
            parents=parents,
        )

    except Exception as e:
        logger.error(f"confluence_get_structure failed: {e}", exc_info=True)
        return f"Failed to get structure: {error_message(e)}"


@mcp.tool()
async def confluence_get_page_content(
    page_id: Annotated[str, Field(description="Confluence page ID")],
    what: Annotated[
        Literal["info", "html", "chunks", "table", "files", "attachments"],
        Field(
            description=(
                "'info' — lightweight page metadata only (title, URL, version, "
                "last_updated; no content, no index access), 'html' — cleaned "
                "live HTML of the page, 'chunks' — page chunks from the index "
                "(optionally filtered by chunk_index), 'table' — all rows/parts "
                "of one block, 'files' — attached files with download links "
                "(lightweight list, use to discover file names), 'attachments' "
                "— attached files with extracted text (docx/pdf/xlsx/… "
                "converted; video/audio/images/archives are skipped)"
            )
        ),
    ] = "chunks",
    block_index: Annotated[
        int | None,
        Field(description="Block number on the page (1-based). Required when what='table'"),
    ] = None,
    chunk_index: Annotated[
        int | None,
        Field(description="Exact chunk number on the page (selects a single chunk)"),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Max chunks to return (chunks/table) / max attachments (files/attachments)"),
    ] = 1000,
    offset: Annotated[
        int,
        Field(description="Chunks to skip (table pagination) / attachments to skip (files/attachments)"),
    ] = 0,
    titles: Annotated[
        list[str] | None,
        Field(
            description=(
                "Exact attachment file names (case-insensitive). "
                "what='files' — filter the list; what='attachments' — convert only these "
                "files (missing names are reported with status 'not_found'). "
                "Use what='files' first to discover available names."
            )
        ),
    ] = None,
    ctx: Context | None = None,
    ) -> PageContent:
    """Get any part of a Confluence page: page metadata ('info'), indexed chunks,
    one table's rows, or attached files (list or extracted text)."""
    logger.debug(f"confluence_get_content: page_id={page_id}, what={what}, "
                f"block_index={block_index}, chunk_index={chunk_index}, "
                f"limit={limit}, offset={offset}, titles={titles}")

    try:
        svc = _get_svc(ctx)
        return await execute_page_content(
            what, page_id, svc,
            limit=limit,
            offset=offset,
            block_index=block_index,
            chunk_index=chunk_index,
            titles=titles,
        )

    except Exception as e:
        logger.error(f"confluence_get_content failed: {e}", exc_info=True)
        return PageContent(page_id=page_id, error=error_message(e))
