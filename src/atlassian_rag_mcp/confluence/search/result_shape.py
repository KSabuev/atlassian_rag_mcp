"""LLM-facing result-format texts for the search algorithms.

Kept apart from ``base.py`` (execution core) so the prose does not dilute the
code; re-exported from ``base`` for backwards-compatible imports.
"""

CQL_RESULT_SHAPE = (
    "RETURNS: up to top_k PAGES in Confluence's own relevance order, each with a plain-text "
    "`excerpt` (markers stripped, truncated to 300 chars) and NO chunks. If the space of ANY "
    "hit cannot be resolved the whole call returns an `error` — an empty `sources` with a "
    "non-empty `error` means malformed input, not 'nothing found'. Pages whose indexed version "
    "differs from the live version are dropped (from the top 10) and queued for re-index. "
    "Nothing filters semantically irrelevant hits, but CQL itself returns nothing rather than "
    "noise when the query is out of domain."
)

CHUNK_RESULT_SHAPE = (
    "RETURNS: up to top_k CHUNKS grouped by page, so normally FEWER than top_k sources; each source "
    "carries `chunks[]` with the FULL chunk text (no size cap) plus chunk_type, heading_hierarchy, "
    "chunk_index — and NO excerpt. Call confluence_get_page_content(what='chunks') for the whole "
    "page body. If the response would exceed the transport limit, the server replaces chunk texts "
    "from the tail with a ready-to-run call instead of cutting them: each omitted chunk's `text` "
    "becomes `[omitted] full text: confluence_get_page_content(what='chunks', page_id=..., "
    "chunk_index=...)` — execute it verbatim to fetch the full text. `score` is this algorithm's "
    "own scale and is NOT "
    "comparable between algorithms. A matched table_row pulls in its table's first two rows at "
    "score 0. Nothing filters semantically irrelevant hits: an out-of-domain query returns "
    "low-scoring noise instead of an empty list, so treat a low score as 'not found' and switch "
    "algorithm."
)
