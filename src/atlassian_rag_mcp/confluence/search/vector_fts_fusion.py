"""Vector + FTS search with true branch fusion.

Two independent branches (vector pool top_k*POOL_MULTIPLIER + the best
FTS_BRANCH_TOP FTS matches) fused by an explicit scoring function:

  - ``fusion="rrf"``:      score(c) = sum over branches of 1 / (rrf_k + rank)
    (reciprocal rank fusion).
  - ``fusion="weighted"``: each branch's raw scores are min-max normalized to
                           [0, 1], then score(c) = w_vec * v(c) + w_kw * f(c)
                           (a chunk missing from a branch contributes 0).

The FTS query carries a configurable title-field boost (``title_boost``,
default 3) so the keyword branch can be tuned without touching the shared
builder.

Default config (weighted, w_kw=0.7, title_boost=3) is the winner of the PROJ
matrix experiment (2026-10-16): score 0.673 vs 0.587 for the former
interleaved baseline, top1 0.70 vs 0.17.
"""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.confluence.fts_builder import (
    build_fusion_query,
)
from atlassian_rag_mcp.confluence.models import Chunk, Page
from atlassian_rag_mcp.exceptions import FtsError
from atlassian_rag_mcp.registry import MethodParameter
from atlassian_rag_mcp.storage.utils import build_filter

from .base import (
    CHUNK_RESULT_SHAPE,
    FTS_QUERY_HELP,
    POOL_MULTIPLIER,
    SEARCH_PARAMS,
    AlgorithmResult,
    MethodDescriptor,
    SearchDeps,
    SearchResult,
    default_postprocess,
    enrich_payload_results,
    register,
)

FUSION_RRF = "rrf"
FUSION_WEIGHTED = "weighted"
FUSION_MODES = (FUSION_RRF, FUSION_WEIGHTED)

FTS_BRANCH_TOP = 60

DEFAULT_TITLE_BOOST = 3
DEFAULT_W_KW = 0.7

VALID_PARAMS = (
    "vector", "fts", "fusion", "w_kw", "title_boost",
    "page_ids", "page_version", "spaces", "heading_text", "search_params",
)


def _min_max_normalize(scores: list[float]) -> list[float]:
    """Min-max normalize a branch score list to [0, 1].

    Degenerate lists (single value or all equal) map to 1.0 when the value is
    positive, 0.0 otherwise — a lone match is still a match.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        v = 1.0 if hi > 0 else 0.0
        return [v] * len(scores)
    span = hi - lo
    return [(s - lo) / span for s in scores]


def fuse_branches(
    vec_ranked: list[tuple[str, float]],
    fts_ranked: list[tuple[str, float]],
    *,
    mode: str = FUSION_WEIGHTED,
    rrf_k: int = 40,
    w_kw: float = DEFAULT_W_KW,
) -> dict[str, float]:
    """Fuse two ranked (chunk_id, raw_score) branch lists into chunk scores.

    mode="rrf":      reciprocal rank fusion, rank is 0-based per branch.
    mode="weighted": min-max normalized raw scores, linear combination with
                     weights w_vec = 1 - w_kw (vector) and w_kw (keyword).
    A chunk present in both branches accumulates both contributions; a chunk
    present in one branch only carries that branch's score.
    """
    if mode not in FUSION_MODES:
        raise ValueError(f"Unknown fusion mode: {mode!r} (expected one of {FUSION_MODES})")

    scores: dict[str, float] = {}
    if mode == FUSION_RRF:
        for branch in (vec_ranked, fts_ranked):
            for rank, (cid, _s) in enumerate(branch):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        return scores

    vec_norm = dict(zip((c for c, _ in vec_ranked), _min_max_normalize([s for _c, s in vec_ranked])))
    fts_norm = dict(zip((c for c, _ in fts_ranked), _min_max_normalize([s for _c, s in fts_ranked])))
    w_vec = 1.0 - w_kw
    for cid in set(vec_norm) | set(fts_norm):
        scores[cid] = w_vec * vec_norm.get(cid, 0.0) + w_kw * fts_norm.get(cid, 0.0)
    return scores


async def run_branches(
    deps: SearchDeps,
    vector: str,
    fts: str,
    *,
    query_filter: dict,
    pool_size: int,
    heading_text: str | None,
    page_version: int | None,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run both search branches for one query; return (vec_ranked, fts_ranked,
    vec_payloads, fts_payloads).

    vec_ranked: (chunk_id, cosine score) over the whole top_k*POOL_MULTIPLIER
    pool (heading_text pre-filtered).
    fts_ranked: (chunk_id, FTS score) over the best FTS_BRANCH_TOP matches
    (page_version post-filtered with overfetch).
    The payload maps carry the raw point payload per chunk_id (vector branch
    wins on collisions — it is the full pool).
    """
    embedding = await deps.embedding_client.get_embedding(vector)
    raw_vec = await deps.vector_db.search(
        vector=embedding,
        limit=pool_size,
        query_filter=query_filter,
    )

    vec_ranked: list[tuple[str, float]] = []
    vec_payloads: dict[str, dict[str, Any]] = {}
    for r in raw_vec:
        pay = r.get("payload", {})
        if heading_text is not None:
            hh = pay.get("heading_hierarchy") or []
            chunk_heading = hh[-1] if hh else ""
            if heading_text.lower() not in chunk_heading.lower():
                continue
        cid = str(r.get("id", ""))
        vec_ranked.append((cid, r.get("score", 0.0)))
        vec_payloads[cid] = pay

    fts_ranked: list[tuple[str, float]] = []
    fts_payloads: dict[str, dict[str, Any]] = {}
    if fts:
        fetch_limit = (
            FTS_BRANCH_TOP * 10 if page_version is not None else FTS_BRANCH_TOP
        )
        raw_fts = await deps.fts_index.search(query=fts, limit=fetch_limit)
        if page_version is not None:
            raw_fts = [
                r for r in raw_fts
                if r.get("payload", {}).get("page_version") == page_version
            ]
        for r in raw_fts[:FTS_BRANCH_TOP]:
            cid = str(r.get("id", ""))
            fts_ranked.append((cid, r.get("score", 0.0)))
            fts_payloads.setdefault(cid, r.get("payload", {}))

    return vec_ranked, fts_ranked, vec_payloads, fts_payloads


class VectorFtsFusionSource(Page):
    """VectorFtsFusion source: ranked chunks with per-branch scores."""

    vector_score: float = 0.0
    fts_score: float = 0.0
    chunks: list[Chunk] = []


class VectorFtsFusionResult(SearchResult):
    """Result of the vector_fts_fusion algorithm."""

    sources: list[VectorFtsFusionSource]


@register
class VectorFtsFusionSearch:
    skip_irrelevant: bool = False
    result_model: type[VectorFtsFusionResult] = VectorFtsFusionResult
    routing: str | None = (
        "conceptual question, no filter, no identifiers (safe default)"
    )

    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="vector_fts_fusion",
            description=(
                "Semantic + keyword search with SCORE FUSION of the two branches "
                "(reciprocal-rank or weighted normalized sum). Branches: vector pool "
                "top_k*10 over the whole index + the best 60 FTS matches; the fused "
                "pool is cut to top_k and grouped by page.\n"
                "USE WHEN: conceptual searches over natural language, possibly mixed "
                "with domain terms. For pure lexical / acronym queries prefer "
                "fulltext_search; for identifiers mixed with structural filters prefer "
                "cql_fts_fusion.\n"
                "NOT FOR: CQL or field syntax — if it lands in `vector` it is silently "
                "embedded, not parsed, and returns noise with NO error.\n"
                "HOW: `spaces`/`page_ids` restrict BOTH branches; `page_version` filters the "
                "vector branch and post-filters FTS hits. Omit `fts` and the "
                "keyword branch reuses `vector`.\n"
                "`score` is the fused branch score (RRF or weighted sum); the raw "
                "per-branch values are in `vector_score` (cosine) and `fts_score` "
                "(FTS).\n"
                + CHUNK_RESULT_SHAPE
                + "\nEXAMPLE: params={\"vector\": \"правила обработки персональных данных\", \"fusion\": \"weighted\", \"w_kw\": 0.7}"  # noqa: E501
            ),
            parameters=[
                MethodParameter(name="vector", type="string",
                                description="Plain natural language describing the CONCEPT you "
                                             "need. NEVER put CQL or field syntax here.",
                                required=True),
                MethodParameter(name="fts", type="string",
                                description=FTS_QUERY_HELP + " Omit to reuse vector "
                                             "as the keyword source (the usual case).",
                                required=False),
                MethodParameter(name="fusion", type="string",
                                description="Branch fusion mode: \"rrf\" (reciprocal rank "
                                             "fusion, k = rrf_k) or \"weighted\" (min-max "
                                             "normalized scores, keyword weight w_kw). "
                                             "Default: \"weighted\"."),
                MethodParameter(name="w_kw", type="number",
                                description="Keyword-branch weight in weighted mode (0..1); "
                                             "vector weight is 1 - w_kw. Default 0.7. "
                                             "Ignored in rrf mode."),
                MethodParameter(name="title_boost", type="integer",
                                description="Exponent of the title-field boost in the FTS "
                                             "query (default 3 — production behavior). "
                                             "Ignored for raw Tantivy queries."),
                MethodParameter(name="page_ids", type="array",
                                description="Restrict to these exact page IDs "
                                             "(applies to BOTH branches)"),
                MethodParameter(name="page_version", type="integer",
                                description="Filter by page version. Vector branch only; FTS "
                                             "hits are post-filtered on the payload."),
                MethodParameter(name="spaces", type="array",
                                description="Restrict to ANY of these space keys "
                                             "(OR-joined, e.g. {space_keys}). Applies to BOTH "
                                             "branches. Omit to search across all indexed "
                                             "spaces."),
                MethodParameter(name="heading_text", type="string",
                                description="Keep only chunks whose deepest heading contains this "
                                             "text (case-insensitive substring). Vector branch "
                                             "only — FTS hits bypass it."),
                SEARCH_PARAMS,
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], deps: SearchDeps) -> AlgorithmResult:
        """Run the vector and FTS branches, fuse them, and return the ranked
        candidate pool (capped at top_k*POOL_MULTIPLIER).

        Ordering is stable: fused score descending, then vector rank as a
        deterministic tie-break preferring the semantic branch. The FTS branch
        can push the fused pool past the vector pool size — the tail is cut.
        """
        unknown = sorted(set(params) - set(VALID_PARAMS))
        if unknown:
            return AlgorithmResult(
                results=[],
                error=f"Unknown parameter(s): {', '.join(unknown)}. Valid parameters: "
                      f"{', '.join(VALID_PARAMS)}.",
            )

        vector = params.get("vector")

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
            return AlgorithmResult(
                results=[], error=f"Invalid title_boost: {params.get('title_boost')!r}"
            )
        if not 1 <= title_boost <= 10:
            return AlgorithmResult(results=[], error="title_boost must be in [1, 10]")

        page_ids = params.get("page_ids")
        page_version = params.get("page_version")
        heading_text = params.get("heading_text")
        spaces = params.get("spaces")

        try:
            fts = build_fusion_query(
                params.get("fts") or vector,
                title_boost=title_boost,
                page_ids=page_ids,
                spaces=spaces,
            )
        except FtsError as e:
            return AlgorithmResult(results=[], error=str(e))

        query_filter = build_filter(
            page_ids=page_ids,
            page_version=page_version,
            spaces=spaces,
        )
        top_k = deps.top_k
        pool_size = top_k * POOL_MULTIPLIER

        try:
            vec_ranked, fts_ranked, vec_payloads, fts_payloads = await run_branches(
                deps,
                vector,
                fts,
                query_filter=query_filter,
                pool_size=pool_size,
                heading_text=heading_text,
                page_version=page_version,
            )
        except FtsError as e:
            return AlgorithmResult(results=[], error=str(e))

        fused = fuse_branches(
            vec_ranked,
            fts_ranked,
            mode=fusion,
            rrf_k=deps.rrf_k,
            w_kw=w_kw,
        )

        vec_rank_order = {cid: i for i, (cid, _s) in enumerate(vec_ranked)}
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], vec_rank_order.get(kv[0], 10**9)))

        candidates: list[dict[str, Any]] = []
        vec_score_by_id = dict(vec_ranked)
        fts_score_by_id = dict(fts_ranked)
        for cid, score in ranked:
            payload = vec_payloads.get(cid) or fts_payloads.get(cid) or {}
            candidates.append({
                "id": cid,
                "payload": payload,
                "_combined_score": score,
                "_vector_score": vec_score_by_id.get(cid, 0.0),
                "_fts_score": fts_score_by_id.get(cid, 0.0),
            })

        candidates = candidates[:pool_size]

        results = enrich_payload_results(candidates, confluence_url=deps.confluence_url)
        return AlgorithmResult(results=results)

    @classmethod
    async def postprocess(cls, result: AlgorithmResult, deps: SearchDeps) -> dict[str, Any]:
        return await default_postprocess(result, deps, result_model=VectorFtsFusionResult)
