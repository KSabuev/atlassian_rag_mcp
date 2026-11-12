"""CQL + FTS search with true branch fusion.

Two independent page-level branches, fused by score instead of re-ranking one
pool:

  - CQL branch: pages from the Confluence CQL API in Confluence's own order
    (rank value = linear position value, already in [0, 1]).
  - FTS branch: local FTS (Tantivy) over the WHOLE query text (any tokens,
    not only codes) — chunk hits are aggregated to the best score per page.
    This is the branch that surfaces pages the CQL pool missed.

The candidate set is the UNION of both branches. Fusion:

  - ``fusion="rrf"``:      reciprocal rank fusion over the two page lists
    (k = rrf_k).
  - ``fusion="weighted"``: each branch's scores are min-max normalized to
    [0, 1] (the CQL rank values already are), then
    score(p) = (1 - w_kw) * cql(p) + w_kw * fts(p).

Every page carries ``score`` (fusion score), ``fts_score`` (raw max FTS, 0
when the FTS branch had no hit) and ``cql_rank`` (original 1-based CQL
position, None for FTS-only pages). Excerpts come from the CQL branch when
the page is in the CQL pool, otherwise from the best-scoring FTS chunk.
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
from atlassian_rag_mcp.confluence.fts_builder import build_fusion_query
from atlassian_rag_mcp.confluence.models import Page
from atlassian_rag_mcp.exceptions import FtsError
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
from .vector_fts_fusion import (
    DEFAULT_TITLE_BOOST,
    FUSION_MODES,
    FUSION_WEIGHTED,
    fuse_branches,
)

FTS_BRANCH_LIMIT = 100
DEFAULT_W_KW = 0.7

_NON_CQL_PARAMS = ("fusion", "w_kw", "title_boost", "search_params")


def _cql_rank_values(n: int) -> list[float]:
    """Linear 1-based rank values in [0, 1] for a CQL list of length n.

    Position 1 -> 1.0, position n -> 0.0 (single-item list -> [1.0]).
    Already min-max shaped, so fuse_branches' normalization is the identity.
    """
    if n <= 0:
        return []
    if n == 1:
        return [1.0]
    return [(n - pos) / (n - 1) for pos in range(1, n + 1)]


class CqlFtsFusionSource(Page):
    """CqlFtsFusion source: fused page with per-branch scores and excerpt."""

    score: float = 0.0
    excerpt: str | None = None
    fts_score: float = 0.0
    cql_rank: int | None = None


class CqlFtsFusionResult(SearchResult):
    """Result of the cql_fts_fusion algorithm."""

    sources: list[CqlFtsFusionSource]


@register
class CqlFtsFusionSearch:
    skip_irrelevant: bool = False
    result_model: type[CqlFtsFusionResult] = CqlFtsFusionResult
    routing: str | None = (
        "query mixes identifiers (document codes, acronyms) with natural language, or "
        "needs a structural filter (title/label/author/date/space)"
    )

    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="cql_fts_fusion",
            description=(
                "CQL page search + local FTS with SCORE FUSION of the two branches "
                "(the candidate set is the UNION of both — pages the CQL pool missed are "
                "still found by the FTS branch). CQL branch: pages in Confluence's own "
                "order. FTS branch: local FTS over the whole query text (any tokens), "
                'best chunk score per page. Fused by "rrf" (1/(rrf_k+rank) per branch) or '
                '"weighted" (min-max normalized branch scores, w_kw = FTS weight, default '
                "0.7). The fused pool is cut to top_k pages.\n"
                "USE WHEN: identifiers (document codes, acronyms) must be combined with "
                "Confluence's structural filters, or when a query needs both a defined "
                "candidate pool and local keyword recall. For pure lexical / acronym "
                "queries without filters use fulltext_search; for conceptual questions "
                "without filters use vector_fts_fusion.\n"
                "NOTES: Pages not in the CQL pool have `cql_rank` = null and an excerpt "
                "built from their best FTS chunk. An empty `cql` with no CQL filters runs "
                "the FTS branch only (degrades to a page-level fulltext search). "
                "Structured CQL filters (title/label/creator/contributor/created/lastmodified/"
                "id/ancestor/parent) apply to the CQL branch only; the FTS branch is built "
                "from `cql` and `space`.\n"
                 + CQL_RESULT_SHAPE
                + '\nEXAMPLE: params={"cql": "правила обработки персональных данных", '
                '"fusion": "weighted", "w_kw": 0.7}'
            ),
            parameters=[
                MethodParameter(
                    name="cql",
                    type="string",
                    description="CQL query fragment or plain text. Used both as the "
                    "CQL branch query and as the keyword source for "
                    "the FTS branch. " + CQL_SYNTAX_HELP,
                    required=False,
                ),
                *cql_filter_params(space_note="Applies to the CQL branch query "
                                              "and to the FTS branch. Omit to "
                                              "search across all spaces."),
                MethodParameter(
                    name="fusion",
                    type="string",
                    description='Branch fusion mode: "rrf" (reciprocal rank '
                    'fusion, k = rrf_k) or "weighted" (min-max '
                    "normalized branch scores, FTS weight w_kw). "
                    'Default: "weighted".',
                ),
                MethodParameter(
                    name="w_kw",
                    type="number",
                    description="FTS-branch weight in weighted mode (0..1); CQL "
                    "weight is 1 - w_kw. Default 0.7. Ignored in rrf "
                    "mode.",
                ),
                MethodParameter(
                    name="title_boost",
                    type="integer",
                    description="Exponent of the title-field boost in the FTS "
                    "query (default 3 — production behavior). "
                    "Ignored for raw CQL/Tantivy queries.",
                ),
                SEARCH_PARAMS,
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], deps: SearchDeps) -> AlgorithmResult:
        top_k = deps.top_k
        try:
            cql_params = CqlSearchParams(**{k: v for k, v in params.items() if k not in _NON_CQL_PARAMS})
        except ValidationError as e:
            return AlgorithmResult(results=[], error=f"Invalid cql parameters: {e}")

        fusion = str(params.get("fusion") or FUSION_WEIGHTED).lower()
        if fusion not in FUSION_MODES:
            return AlgorithmResult(
                results=[],
                error=f"Invalid fusion mode {fusion!r}: expected one of {list(FUSION_MODES)}",
            )
        try:
            w_kw = float(params.get("w_kw", DEFAULT_W_KW))
        except (TypeError, ValueError):
            return AlgorithmResult(results=[], error=f"Invalid w_kw: {params.get('w_kw')!r}")
        if not 0.0 <= w_kw <= 1.0:
            return AlgorithmResult(results=[], error="w_kw must be in [0, 1]")
        try:
            title_boost = int(params.get("title_boost", DEFAULT_TITLE_BOOST))
        except (TypeError, ValueError):
            return AlgorithmResult(results=[], error=f"Invalid title_boost: {params.get('title_boost')!r}")
        if not 1 <= title_boost <= 10:
            return AlgorithmResult(results=[], error="title_boost must be in [1, 10]")

        cql_pages: dict[str, dict[str, Any]] = {}
        cql_order: list[str] = []
        cql_query = build_cql_query(cql_params)
        if cql_query:
            batch = await deps.confluence_client.cql(
                cql=cql_query,
                start=0,
                limit=top_k,
                expand="content.version",
            )
            for pos, r in enumerate(batch.results, start=1):
                pid = cql_page_id(r)
                if not pid or pid in cql_pages:
                    continue
                content = r.get("content", {})
                ver = content.get("version") or {}
                space_key = cql_space_key(r)
                cql_pages[pid] = {
                    "page_id": pid,
                    "page_title": content.get("title", ""),
                    "page_version": ver.get("number", 0),
                    "updated_at": ver.get("when"),
                    "page_url": build_page_url(pid, deps.confluence_url),
                    "excerpt": cql_clean_excerpt(r.get("excerpt") or "")[:300] or None,
                    "space_key": space_key,
                    "cql_rank": pos,
                }
                cql_order.append(pid)

            missing = [page_id for page_id, p in cql_pages.items() if not p["space_key"]]
            if missing:
                return AlgorithmResult(
                    results=[],
                    error=f"space_key not resolved for page(s): {', '.join(sorted(set(missing)))}",
                )

        cql_ranked = [(pid, rv) for pid, rv in zip(cql_order, _cql_rank_values(len(cql_order)))]

        try:
            fts_query = build_fusion_query(
                cql_params.cql or "",
                title_boost=title_boost,
                spaces=cql_params.space,
            )
        except FtsError as e:
            return AlgorithmResult(results=[], error=str(e))
        fts_best: dict[str, float] = {}
        fts_excerpt: dict[str, str] = {}
        fts_meta: dict[str, dict[str, Any]] = {}
        if fts_query:
            try:
                raw_fts = await deps.fts_index.search(query=fts_query, limit=FTS_BRANCH_LIMIT)
            except FtsError as e:
                return AlgorithmResult(results=[], error=str(e))
            for r in raw_fts:
                payload = r.get("payload", {})
                pid = str(payload.get("page_id") or "")
                if not pid:
                    continue
                sc = float(r.get("score", 0.0))
                if sc > fts_best.get(pid, 0.0):
                    fts_best[pid] = sc
                    fts_excerpt[pid] = (payload.get("text") or "")[:300]
                    fts_meta[pid] = {
                        "page_title": payload.get("page_title", ""),
                        "page_version": payload.get("page_version", 0),
                        "updated_at": payload.get("updated_at"),
                        "space_key": payload.get("space_key"),
                    }
        fts_ranked = sorted(fts_best.items(), key=lambda kv: (-kv[1], kv[0]))

        fused = fuse_branches(
            cql_ranked,
            fts_ranked,
            mode=fusion,
            rrf_k=deps.rrf_k,
            w_kw=w_kw,
        )

        cql_rank_of = {pid: pos for pos, pid in enumerate(cql_order, start=1)}
        ranked = sorted(
            fused.items(),
            key=lambda kv: (-kv[1], cql_rank_of.get(kv[0], 10**9), kv[0]),
        )

        results_list: list[dict[str, Any]] = []
        for pid, score in ranked[:top_k]:
            cql_page = cql_pages.get(pid)
            if cql_page:
                item = dict(cql_page)
            else:
                meta = fts_meta.get(pid, {})
                item = {
                    "page_id": pid,
                    "page_title": meta.get("page_title", ""),
                    "page_version": meta.get("page_version", 0),
                    "updated_at": meta.get("updated_at"),
                    "page_url": build_page_url(pid, deps.confluence_url),
                    "excerpt": fts_excerpt.get(pid) or None,
                    "space_key": meta.get("space_key"),
                    "cql_rank": None,
                }
            item["score"] = score
            item["fts_score"] = fts_best.get(pid, 0.0)
            results_list.append(item)

        if not results_list:
            return AlgorithmResult(results=[])

        return AlgorithmResult(results=results_list)

    @classmethod
    async def postprocess(cls, result: AlgorithmResult, deps: SearchDeps) -> dict[str, Any]:
        """Assemble page-level sources and re-inject the page-level fields
        (score, cql_rank) that _build_rag_result drops for non-chunk sources."""
        out = await default_postprocess(result, deps, result_model=CqlFtsFusionResult)
        rank_by_page: dict[str, int | None] = {}
        score_by_page: dict[str, float] = {}
        for item in result.results:
            pid = item.get("page_id")
            if pid is not None and pid not in rank_by_page:
                rank_by_page[pid] = item.get("cql_rank")
                score_by_page[pid] = float(item.get("score", 0.0))
        for s in out.get("sources", []):
            s["cql_rank"] = rank_by_page.get(s.get("page_id"))
            s["score"] = score_by_page.get(s.get("page_id"), 0.0)
        return out
