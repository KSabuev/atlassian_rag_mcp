"""Base — SearchDeps, registry, helpers, enrichment, public API."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any

from pydantic import BaseModel

from atlassian_rag_mcp.config.settings import settings
from atlassian_rag_mcp.embedding.base import EmbeddingProvider
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.registry import (
    MethodDescriptor,
    MethodParameter,
    Registry,
    missing_required,
    substitute_placeholders,
)
from atlassian_rag_mcp.storage.base import VectorDB
from atlassian_rag_mcp.storage.fts.base import FTSIndex
from atlassian_rag_mcp.storage.utils import dedupe_chunks_by_id

from .. import ConfluenceClient
from ..fts_builder import FTS_QUERY_HELP  # noqa: F401  (re-export: algorithms import it from here)
from ..models import Chunk, ChunkPayload, Page, make_payload
from .result_shape import CHUNK_RESULT_SHAPE, CQL_RESULT_SHAPE  # noqa: F401  (same)

RRF_K = 40
POOL_MULTIPLIER = 10


@dataclass
class SearchDeps:
    vector_db: VectorDB
    embedding_client: EmbeddingProvider
    confluence_client: ConfluenceClient

    fts_index: FTSIndex
    top_k: int = 20
    rrf_k: int = RRF_K
    confluence_space_key: str = ""
    confluence_url: str = ""
    get_in_work_versions: Any = None


class AlgorithmResult(BaseModel):
    results: list[Any]
    error: str | None = None


class SearchResult(BaseModel):
    """Common search-result envelope; each algorithm narrows ``sources`` in its own subclass."""

    sources: list[Page]
    error: str | None = None


class RankedResultEntry(BaseModel):
    """A single ranked chunk item consumed by the enrichment pipeline."""

    score: float = 0.0
    vector_score: float = 0.0
    fts_score: float = 0.0
    page_id: str | None = None
    page_title: str = ""
    page_url: str | None = None
    page_version: int | None = None
    parent_id: str | None = None
    updated_at: str | None = None
    space_key: str | None = None
    chunk: dict[str, Any] = {}


algorithms = Registry("algorithms")
register = algorithms.register


def _space_keys_example() -> str:
    """Example space-key list built from the configured CONFLUENCE_SPACE_KEY."""
    keys = settings.confluence_space_keys
    return "[" + ", ".join(f'"{k}"' for k in keys) + "]"


def get_algorithms() -> list[MethodDescriptor]:
    """Return list of registered MethodDescriptors (space-key examples from env)."""
    placeholders = {"{space_keys}": _space_keys_example()}
    methods = algorithms.get_methods()
    for m in methods:
        m.description = substitute_placeholders(m.description, placeholders)
        for p in m.parameters:
            p.description = substitute_placeholders(p.description, placeholders)
    return methods


def build_page_url(page_id: str | None, base_url: str = "") -> str | None:
    """Build Confluence page URL from page_id and base_url."""
    if page_id and base_url:
        base = base_url.rstrip("/")
        return f"{base}/pages/viewpage.action?pageId={page_id}"
    return None


_HL_RE = re.compile(r"@@@(?:hl|endhl)@@@")


def cql_clean_excerpt(excerpt: str) -> str:
    """Strip Confluence's internal ``@@@hl@@@...@@@endhl@@@`` highlight markers."""
    return _HL_RE.sub("", excerpt or "").strip()


def cql_page_id(r: dict[str, Any]) -> str:
    """Extract the page id from a CQL search result item."""
    return r.get("content", {}).get("id", "")


def cql_space_key(r: dict[str, Any]) -> str:
    """Extract the space key from the API response's ``_expandable.space`` URL.

    The CQL search result does not carry the space object itself, but
    ``content._expandable.space`` holds the endpoint ``/rest/api/space/{key}``
    for every result (with or without a ``space`` CQL filter).
    """
    content = r.get("content") or {}
    url = (content.get("_expandable") or {}).get("space") or ""
    return url.rsplit("/", 1)[-1]


def _payload_to_chunk_dict(payload: ChunkPayload) -> dict[str, Any]:
    return payload.model_dump(
        include={
            "chunk_id",
            "text",
            "chunk_type",
            "chunk_index",
            "block_index",
            "block_part_id",
            "heading_hierarchy",
            "space_key",
        }
    )


HEADER_ROWS = 2


def _iter_table_headers(
    results_list: list[dict[str, Any]],
) -> set[tuple[str, int, int]]:
    """Collect ``(page_id, page_version, block_index)`` of tables that produced
    matched ``table_row`` chunks. Whole-table ``table`` chunks are skipped —
    they already contain every row including the header.
    """
    needs: set[tuple[str, int, int]] = set()
    for item in results_list:
        chunk = item.get("chunk") or {}
        if chunk.get("chunk_type") != "table_row":
            continue
        pid = item.get("page_id")
        ver = item.get("page_version")
        block = chunk.get("block_index")
        if pid and ver is not None and block is not None:
            needs.add((pid, int(ver), int(block)))
    return needs


async def augment_table_header_rows(
    deps: SearchDeps,
    results_list: list[dict[str, Any]],
    in_work_versions: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Fetch the first two rows (block_part_id 1 and 2) of every table that
    already has a matched ``table_row`` chunk and return them as extra entries.

    Scores are 0 so the extra chunks never affect ranking. A row that already
    matched keeps the same ``chunk_id`` as the fetched copy, so the existing
    ``dedupe_chunks_by_id`` collapses duplicates — no new dedup logic needed.
    """
    in_work_versions = in_work_versions or {}
    needs = _iter_table_headers(results_list)
    if not needs:
        return []

    by_page: dict[tuple[str, int], set[int]] = defaultdict(set)
    for pid, ver, block in needs:
        if in_work_versions.get(pid) is not None and int(in_work_versions[pid]) == ver:
            continue
        by_page[(pid, ver)].add(block)

    extra: list[dict[str, Any]] = []
    for (pid, ver), blocks in by_page.items():
        try:
            points, _ = await deps.vector_db.scroll_points(
                filter_condition={
                    "must": [
                        {"key": "page_id", "match": {"value": pid}},
                        {"key": "page_version", "match": {"value": ver}},
                        {"key": "chunk_type", "match": {"value": "table_row"}},
                        {"key": "block_index", "match": {"any": sorted(blocks)}},
                        {"key": "block_part_id", "match": {"any": [1, HEADER_ROWS]}},
                    ],
                },
                limit=len(blocks) * HEADER_ROWS,
            )
        except Exception as e:
            logger.warning(
                f"Failed to fetch header rows for page {pid} v{ver}: {e}",
                exc_info=True,
            )
            continue

        seen_keys: set[tuple[int, int]] = set()
        for p in points:
            pay = p.get("payload", {})
            part = pay.get("block_part_id")
            if not isinstance(part, int) or part not in (1, HEADER_ROWS):
                continue
            block = pay.get("block_index")
            if not isinstance(block, int) or block not in blocks:
                continue
            if (block, part) in seen_keys:
                continue
            seen_keys.add((block, part))

            payload = make_payload(pay, p.get("id"))
            extra.append(
                RankedResultEntry(
                    score=0.0,
                    vector_score=0.0,
                    fts_score=0.0,
                    page_id=pid,
                    page_title=payload.page_title,
                    page_url=build_page_url(payload.page_id, deps.confluence_url),
                    page_version=payload.page_version,
                    parent_id=payload.parent_id,
                    updated_at=payload.updated_at,
                    space_key=payload.space_key,
                    chunk=_payload_to_chunk_dict(payload),
                ).model_dump()
            )
    return extra


def enrich_payload_results(
    results: list[dict[str, Any]],
    *,
    score_field: str = "_combined_score",
    vector_score_field: str = "_vector_score",
    fts_score_field: str | None = "_fts_score",
    confluence_url: str = "",
) -> list[dict[str, Any]]:
    """Enrich raw search results with scores and page info, sorted by score."""
    enriched = []
    for r in results:
        payload = make_payload(r.get("payload", {}), r.get("id"))
        entry = RankedResultEntry(
            score=r.get(score_field, r.get("score", 0)),
            vector_score=r.get(vector_score_field, 0),
            fts_score=r.get(fts_score_field, 0.0) if fts_score_field is not None else 0.0,
            page_id=payload.page_id,
            page_title=payload.page_title,
            page_url=build_page_url(payload.page_id, confluence_url),
            page_version=payload.page_version,
            parent_id=payload.parent_id,
            updated_at=payload.updated_at,
            space_key=payload.space_key,
            chunk=_payload_to_chunk_dict(payload),
        ).model_dump()
        enriched.append(entry)
    enriched.sort(key=lambda x: x["score"], reverse=True)
    return enriched


def _build_rag_result(
    algo_result: AlgorithmResult,
    in_work_versions: dict[str, int] | None = None,
    *,
    result_model: type[SearchResult] = SearchResult,
) -> dict[str, Any]:
    """Group ranked items into per-page sources validated into ``result_model``.

    Chunks of a page version currently "in work" (being re-indexed) are dropped:
    the page is served using only its last completed version. Sources keep the
    first-appearance rank order of the incoming items — algorithms deliver
    results pre-ranked (fusion — by fused chunk score, cql_fts_fusion — by
    fused page score).

    Each source dict is a superset of all possible fields; the model keeps only
    its own fields (pydantic ``extra='ignore'``), so chunk-based sources never
    carry ``excerpt`` and excerpt-based sources never carry ``chunks``/scores.
    """
    results_list = algo_result.results
    if not results_list:
        return result_model(sources=[], error=algo_result.error).model_dump()

    in_work_versions = in_work_versions or {}

    def _is_hidden(item: dict) -> bool:
        pid = item.get("page_id")
        ver = item.get("page_version")
        if pid is None or ver is None:
            return False
        in_work = in_work_versions.get(pid)
        return in_work is not None and int(ver) == int(in_work)

    visible = [item for item in results_list if not _is_hidden(item)]

    page_groups: dict[str, list[dict]] = defaultdict(list)
    for item in visible:
        pid = item.get("page_id")
        if pid:
            page_groups[pid].append(item)

    sources = []
    for pid, items in page_groups.items():
        first = items[0]
        vector_score = 0.0
        fts_score = 0.0
        chunks = []
        for item in items:
            vector_score = max(vector_score, item.get("vector_score", 0.0))
            fts_score = max(fts_score, item.get("fts_score", 0.0))
            if "chunk" in item:
                chunks.append(item["chunk"])

        chunks = dedupe_chunks_by_id(chunks)
        chunk_models = [Chunk.model_validate(c).model_dump() for c in chunks]

        sources.append(
            {
                "page_id": pid,
                "page_title": first.get("page_title", ""),
                "page_url": first.get("page_url", ""),
                "version": first.get("page_version"),
                "last_updated": first.get("updated_at"),
                "parent_id": first.get("parent_id", ""),
                "space_key": first.get("space_key"),
                "vector_score": vector_score,
                "fts_score": fts_score,
                "excerpt": (first.get("excerpt") or None),
                "chunks": chunk_models,
            }
        )

    return result_model(sources=sources, error=algo_result.error).model_dump()


async def _resolve_in_work_versions(deps: SearchDeps) -> dict[str, int]:
    fn = getattr(deps, "get_in_work_versions", None)
    if not callable(fn):
        return {}
    try:
        return await fn() or {}
    except Exception:
        return {}


async def default_postprocess(
    result: AlgorithmResult,
    deps: SearchDeps,
    *,
    result_model: type[SearchResult] = SearchResult,
) -> dict[str, Any]:
    """Default postprocessing: resolve in-work versions, append table header
    rows for matched table_row chunks, enrich and assemble the result."""
    in_work = await _resolve_in_work_versions(deps)
    results_list = result.results
    extra = await augment_table_header_rows(deps, results_list, in_work_versions=in_work)
    if extra:
        results_list = [*results_list, *extra]
    merged = AlgorithmResult(results=results_list, error=result.error)
    return _build_rag_result(merged, in_work_versions=in_work, result_model=result_model)


SEARCH_PARAMS = MethodParameter(
    name="search_params",
    type="object",
    description="Optional per-request constants object overriding server defaults "
    '(e.g. {"top_k": 10, "rrf_k": 40}).\n'
    "Keys: top_k (int >= 1, max results returned — the algorithms build a "
    "top_k*10 candidate pool from their branch/fusion and cut it to top_k), "
    "rrf_k (int >= 1, RRF fusion constant).\n"
    "Unknown keys or invalid values return an error. Precedence: search_params > "
    "server settings.",
    required=False,
)

SEARCH_PARAMS_KEYS = ("top_k", "rrf_k")


def _coerce_positive_int(value: Any) -> int:
    """Coerce an MCP value to an int >= 1 (accepts int and numeric strings)."""
    if isinstance(value, bool):
        raise ValueError("must be an integer, got bool")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"must be an integer, got {value!r}")
        value = int(value)
    if isinstance(value, int):
        n = value
    elif isinstance(value, str):
        try:
            n = int(value.strip())
        except ValueError:
            raise ValueError(f"must be an integer, got {value!r}")
    else:
        raise ValueError(f"must be an integer, got {value!r}")
    if n < 1:
        raise ValueError(f"must be >= 1, got {n}")
    return n


def resolve_search_params(params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Parse the optional per-request ``search_params`` constants object.

    Returns ``(overrides, error)``: ``overrides`` holds only the keys present in
    the request (coerced and validated), ``error`` is non-None when the object is
    malformed (not a dict, unknown key, wrong type or out-of-range value).
    """
    raw = params.get("search_params")
    if raw is None:
        return {}, None
    if not isinstance(raw, dict):
        return {}, "search_params must be an object"
    if not raw:
        return {}, None

    overrides: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in SEARCH_PARAMS_KEYS:
            return {}, (
                f"unknown search_params key: {key!r} "
                "(allowed: top_k, rrf_k)"
            )
        if value is None:
            continue
        try:
            overrides[key] = _coerce_positive_int(value)
        except ValueError as e:
            return {}, f"invalid search_params.{key}: {e}"
    return overrides, None


def _response_bytes(result_dict: dict[str, Any]) -> int:
    """Size of the response as sent over the MCP transport (the SSE body).

    FastMCP delivers the tool result TWICE: pretty-printed JSON (indent=2) in
    ``result.content[0].text`` (indentation on top of quote/backslash escaping)
    and the compact object in ``result.structuredContent``. Measure both copies
    plus a margin for the JSON-RPC/SSE envelope, so the limit bounds the actual
    body — keeping it under the ~1 MB SSE event cap of MCP clients.
    """
    pretty = json.dumps(result_dict, ensure_ascii=False, indent=2)
    compact = json.dumps(result_dict, ensure_ascii=False)
    return (
        len(json.dumps(pretty).encode("utf-8"))
        + len(compact.encode("utf-8"))
        + 65536
    )


OMITTED_TEXT_PREFIX = "[omitted] full text: "


def _omitted_chunk_text(page_id: Any, chunk_index: Any) -> str:
    """Ready-to-run re-fetch call stored in an omitted chunk's ``text``."""
    pid = str(page_id) if page_id else ""
    if not pid:
        return OMITTED_TEXT_PREFIX + "unavailable (page_id missing in this response)"
    call = f"confluence_get_page_content(what='chunks', page_id='{pid}'"
    if chunk_index is not None:
        call += f", chunk_index={chunk_index}"
    call += ")"
    return OMITTED_TEXT_PREFIX + call


def apply_response_size_guard(
    result_dict: dict[str, Any],
    *,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Keep the serialized search response within the transport limit.

    MCP clients cap a single SSE event at ~1 MB. Chunk texts are full
    (untruncated) by default; when the serialized response would exceed
    ``max_bytes`` (default ``settings.max_response_bytes``), chunk texts are
    replaced from the TAIL of the result forward with a ready-to-run re-fetch
    call (``[omitted] full text: confluence_get_page_content(what='chunks',
    page_id=..., chunk_index=...)``) — nothing is ever cut silently: the full
    texts always remain in the index and the call in ``text`` is executable
    verbatim.
    """
    limit = max_bytes if max_bytes is not None else settings.max_response_bytes
    if _response_bytes(result_dict) <= limit:
        return result_dict

    done = False
    for source in reversed(result_dict.get("sources", [])):
        chunks = source.get("chunks") or []
        for chunk in reversed(chunks):
            text = chunk.get("text")
            if not text or text.startswith(OMITTED_TEXT_PREFIX):
                continue
            chunk["text"] = _omitted_chunk_text(source.get("page_id"), chunk.get("chunk_index"))
            if _response_bytes(result_dict) <= limit:
                done = True
                break
        if done:
            break
    return result_dict


async def execute_algorithm(
    algorithm_name: str,
    params: dict[str, Any],
    deps: SearchDeps,
    *,
    relevant_fn: Any = None,
) -> dict[str, Any]:
    """Execute a search algorithm and return ranked results.

    search_params overrides (top_k/rrf_k) apply to a copy of ``deps`` — the
    shared server-wide singleton is never mutated. Pool-based algorithms (the
    fusion family) return a top_k*POOL_MULTIPLIER candidate pool; it is cut to
    the effective top_k in incoming (branch/fusion) order — a no-op for
    self-trimming algorithms that already return <= top_k items.

    relevant_fn: optional async callable ``(list[dict]) -> list[dict]`` applied
              to the ``sources`` list after postprocessing when the algorithm class
              has ``skip_irrelevant = False``.
    """
    cls = algorithms.get(algorithm_name)
    if cls is None:
        return SearchResult(
            sources=[],
            error=f"Unknown algorithm: {algorithm_name}. Available: {', '.join(algorithms.ids())}",
        ).model_dump()

    missing = missing_required(cls.descriptor(), params)
    if missing:
        return SearchResult(
            sources=[],
            error=f"Missing required parameter(s): {', '.join(missing)}",
        ).model_dump()

    overrides, sp_error = resolve_search_params(params)
    if sp_error:
        return SearchResult(sources=[], error=sp_error).model_dump()

    if overrides:
        deps = replace(
            deps,
            **{k: overrides[k] for k in ("top_k", "rrf_k") if k in overrides},
        )

    result = await algorithms.invoke(cls, params, deps)

    results = result.results
    result = AlgorithmResult(
        results=results[:deps.top_k] if len(results) > deps.top_k else results,
        error=result.error,
    )

    fn = getattr(cls, "postprocess", None)
    if fn is not None:
        postprocessed = await fn(result, deps)
    else:
        postprocessed = await default_postprocess(
            result,
            deps,
            result_model=getattr(cls, "result_model", SearchResult),
        )

    if relevant_fn is not None and not getattr(cls, "skip_irrelevant", False):
        sources = await relevant_fn(postprocessed.get("sources", []))
        postprocessed = {**postprocessed, "sources": sources}

    return apply_response_size_guard(postprocessed)
