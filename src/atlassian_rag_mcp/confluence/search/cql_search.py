"""Pure CQL search (diagnostics / builder verification).

Runs ONLY the CQL branch of the cql_fts_fusion family: builds a CQL query from
structured parameters and/or raw ``cql`` via the CQL builder (same allow-list
validation), calls Confluence's CQL API once, and returns the pages in
Confluence's own order — no FTS branch, no score fusion, no relevance
filtering. Intended to verify CQL builder behavior end-to-end: the exact CQL
string the builder produced from the parameters is logged at debug level
(``cql_search: built CQL query: ...``).

NOT a relevance search: no scoring, no candidate union, no FTS recall. For
finding information use cql_fts_fusion / fulltext_search / vector_fts_fusion.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from atlassian_rag_mcp.confluence.cql_builder import (
    CQL_SYNTAX_HELP,
    CqlSearchParams,
    build_cql_query,
    cql_filter_params,
)
from atlassian_rag_mcp.confluence.models import Page
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.registry import MethodParameter

from .base import (
    CQL_RESULT_SHAPE,
    SEARCH_PARAMS,
    AlgorithmResult,
    MethodDescriptor,
    SearchDeps,
    SearchResult,
    build_page_url,
    cql_clean_excerpt,
    cql_page_id,
    cql_space_key,
    default_postprocess,
    register,
)

_NON_CQL_PARAMS = ("search_params",)


class CqlSearchSource(Page):
    """CqlSearch source: a page in Confluence's own CQL order (rank only, no score)."""

    excerpt: str | None = None
    cql_rank: int | None = None


class CqlSearchResult(SearchResult):
    """Result of the cql_search algorithm (pure CQL branch)."""

    sources: list[CqlSearchSource]


@register
class CqlSearch:
    skip_irrelevant: bool = True
    result_model: type[CqlSearchResult] = CqlSearchResult
    routing: str | None = None  # diagnostic — not part of the search routing table

    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="cql_search",
            description=(
                "DIAGNOSTIC: pure CQL search — runs ONLY the CQL branch (no FTS scoring, no "
                "score fusion, no relevance filtering): builds a CQL query from the "
                "structured parameters and/or raw `cql` via the CQL builder (same "
                "allow-list validation) and returns Confluence's own CQL results in "
                "Confluence's order (1-based cql_rank, no score). Use to verify CQL "
                "builder behavior — the exact CQL string built from the parameters is "
                "logged at debug level (\"cql_search: built CQL query: ...\"). NOT a "
                "relevance search: for finding information use cql_fts_fusion / "
                "fulltext_search / vector_fts_fusion.\n"
                + CQL_RESULT_SHAPE
            ),
            parameters=[
                MethodParameter(name="cql", type="string",
                                description="CQL query fragment or plain text. "
                                              + CQL_SYNTAX_HELP,
                                required=False),
                *cql_filter_params(space_note="Omit to search across all spaces."),
                SEARCH_PARAMS,
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], deps: SearchDeps) -> AlgorithmResult:
        try:
            cql_params = CqlSearchParams(
                **{k: v for k, v in params.items() if k not in _NON_CQL_PARAMS}
            )
        except ValidationError as e:
            return AlgorithmResult(results=[], error=f"Invalid cql parameters: {e}")

        cql_query = build_cql_query(cql_params)
        if not cql_query:
            return AlgorithmResult(
                results=[],
                error="No CQL query: provide cql or at least one filter parameter",
            )
        logger.debug(f"cql_search: built CQL query: {cql_query!r}")

        pages: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        batch = await deps.confluence_client.cql(
            cql=cql_query, start=0, limit=deps.top_k, expand="content.version",
        )
        for pos, r in enumerate(batch.results, start=1):
            pid = cql_page_id(r)
            if not pid or pid in pages:
                continue
            content = r.get("content", {})
            ver = content.get("version") or {}
            space_key = cql_space_key(r)
            pages[pid] = {
                "page_id": pid,
                "page_title": content.get("title", ""),
                "page_version": ver.get("number", 0),
                "updated_at": ver.get("when"),
                "page_url": build_page_url(pid, deps.confluence_url),
                "excerpt": cql_clean_excerpt(r.get("excerpt") or "")[:300] or None,
                "space_key": space_key,
                "cql_rank": pos,
            }
            order.append(pid)

        missing = [pid for pid in order if not pages[pid]["space_key"]]
        if missing:
            return AlgorithmResult(
                results=[],
                error=f"space_key not resolved for page(s): {', '.join(sorted(set(missing)))}",
            )

        return AlgorithmResult(results=[pages[pid] for pid in order])

    @classmethod
    async def postprocess(cls, result: AlgorithmResult, deps: SearchDeps) -> dict[str, Any]:
        """Re-inject the page-level ``cql_rank`` that _build_rag_result drops for
        non-chunk sources (excerpt is kept by the default postprocessing)."""
        out = await default_postprocess(result, deps, result_model=CqlSearchResult)
        rank_by_page: dict[str, int | None] = {}
        for item in result.results:
            pid = item.get("page_id")
            if pid is not None and pid not in rank_by_page:
                rank_by_page[pid] = item.get("cql_rank")
        for s in out.get("sources", []):
            s["cql_rank"] = rank_by_page.get(s.get("page_id"))
        return out
