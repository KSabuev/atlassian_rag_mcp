"""Full-text search algorithm — pure FTS (Tantivy) lexical search over indexed chunks."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from atlassian_rag_mcp.confluence.fts_builder import FtsSearchParams, build_fts_query
from atlassian_rag_mcp.confluence.models import Chunk, Page
from atlassian_rag_mcp.exceptions import FtsError
from atlassian_rag_mcp.registry import MethodParameter

from .base import (
    CHUNK_RESULT_SHAPE,
    FTS_QUERY_HELP,
    SEARCH_PARAMS,
    AlgorithmResult,
    MethodDescriptor,
    SearchDeps,
    SearchResult,
    default_postprocess,
    enrich_payload_results,
    register,
)


class FullTextSearchSource(Page):
    """FullTextSearch source: matched chunks ranked by FTS score (no vector branch)."""

    fts_score: float = 0.0
    chunks: list[Chunk] = []


class FullTextSearchResult(SearchResult):
    """Result of the fulltext_search algorithm."""

    sources: list[FullTextSearchSource]


@register
class FullTextSearch:
    skip_irrelevant: bool = False
    result_model: type[FullTextSearchResult] = FullTextSearchResult
    routing: str | None = (
        "pure lexical query: Cyrillic acronym or short domain word "
        "(3-4 letter acronyms)"
    )

    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="fulltext_search",
            description=(
                "Pure lexical search (FTS/Tantivy) over every indexed chunk — exact wording, no "
                "semantic layer.\n"
                "USE WHEN: the exact text must be present in the document — a quoted phrase, an "
                "error message, an API/field/macro name, a value inside a table. It is also the "
                "right choice for Cyrillic acronyms and short domain words, which the "
                "embedding model cannot represent. Table cells are indexed "
                "row by row, so this is the only algorithm that finds text living inside "
                "tables. Identifier codes (case codes, Jira IDs, release versions: ABC.DEF.01, "
                "PROJ-1002, release.2026.3.1) are indexed as exact pattern tokens and matched "
                "with a boosted exact-term branch, so an exact code is found even when its "
                "parts appear scattered in other documents or inside related codes "
                "(C_ABC.DEF.01).\n"
                "NOT FOR: paraphrases and conceptual questions — FTS has no notion of meaning. "
                "NOT for long generic titles: plain input is OR-joined, so frequent words "
                "('модель', 'миграция', 'расчёт') score highly on large pages and can push the "
                "page with the exact short title out of the result window — require all terms "
                "with `text:(a AND b)` in that case.\n"
                "HOW: plain text is tokenized and OR-joined into `text:(w1 OR w2 ...)`; a "
                "string containing : \" ^ ~ * ? ( ) or AND/OR/NOT is passed through as RAW "
                "Tantivy. Only the fields text/title/pattern/page/space exist; an unknown "
                "field prefix (chunk_id:x, foo:barBaz, http://...) or an unbalanced "
                "quote/paren is rejected with an error — a non-ASCII prefix "
                "('тема: развёртывание') is plain text. `spaces`/`page_ids` are AND-ed "
                "into the query string itself.\n"
                + CHUNK_RESULT_SHAPE
                + "\nEXAMPLE: params={\"fts\": \"\\\"приёмка передачи\\\"\", "
                "\"spaces\": [\"PROJ\"]} — exact phrase. params={\"fts\": "
                "\"text:(миграция AND гос)\"} — require both terms."
            ),
            parameters=[
                MethodParameter(name="fts", type="string",
                                   description="Full-text query. " + FTS_QUERY_HELP,
                                   required=True),
                MethodParameter(name="page_ids", type="array",
                                    description="Restrict to these exact page IDs"),
                MethodParameter(name="spaces", type="array",
                                     description="Restrict to ANY of these space keys "
                                                  "(OR-joined, e.g. {space_keys}). Omit to search "
                                                  "across all spaces."),
                SEARCH_PARAMS,
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], deps: SearchDeps) -> AlgorithmResult:
        """Run a single FTS query. Hits are pre-ranked, so exactly top_k are
        requested; the trailing slice is a defensive bound in case a backend
        ignores the limit (keeps the response within the MCP transport cap)."""
        top_k = deps.top_k
        try:
            fts_params = FtsSearchParams(
                **{k: v for k, v in params.items() if k != "search_params"}
            )
        except ValidationError as e:
            return AlgorithmResult(results=[], error=f"Invalid fulltext_search parameters: {e}")
        try:
            fts = build_fts_query(fts_params)
            if not fts:
                return AlgorithmResult(results=[], error="No query provided")
            raw = await deps.fts_index.search(query=fts, limit=top_k)
        except FtsError as e:
            return AlgorithmResult(results=[], error=str(e))
        raw = raw[:top_k]

        candidates = [
            {
                "id": r.get("id", ""),
                "payload": r.get("payload", {}),
                "_combined_score": r.get("score", 0),
                "_fts_score": r.get("score", 0),
            }
            for r in raw
        ]

        results = enrich_payload_results(
            candidates,
            confluence_url=deps.confluence_url,
        )
        return AlgorithmResult(results=results)

    @classmethod
    async def postprocess(cls, result: AlgorithmResult, deps: SearchDeps) -> dict[str, Any]:
        return await default_postprocess(result, deps, result_model=FullTextSearchResult)
