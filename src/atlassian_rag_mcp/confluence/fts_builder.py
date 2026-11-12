"""FTS (Tantivy query string) query builder — sanitization, fluent API.

Counterpart of CQLBuilder/JQLBuilder: sanitize_fts_value(),
fts_looks_advanced(), module-level build_fts_query(), chainable FtsBuilder.

- Raw recognition (``fts_looks_advanced``): only explicit Tantivy syntax
  counts — ``field:`` prefixes over the index fields, quoted phrases,
  whole-word uppercase AND/OR/NOT and ``*`` wildcards. Boost suffixes
  (``тест^2``) are stripped before detection (the builder handles ``^N``);
  a boost attached to raw syntax (``text:abc^2``) passes through raw. Every
  form is verified against the live index's ``parse_query``.
- Strict validation (``validate_fts_query``): an ASCII ``field:`` prefix must
  be one of the allowed fields (text/title/pattern/page/space; ``chunk_id:``
  and any other identifier are rejected with FtsError), as are an empty field
  value and unbalanced quotes/parentheses. A non-ASCII prefix (``тема:``) is
  plain text, not a field reference.
- Plain keywords are normalized like the index (storage/fts/normalize):
  Russian-stemmed, stopwords dropped; the clause is emitted with a title boost,
  ``(title:(...))^3 OR text:(...)``. PATTERNS-whitelisted codes additionally
  get an exact-match branch on ``pattern`` (one indexed term, outranks pages
  that only list related codes).
- ``FTS_QUERY_HELP`` is the LLM-facing semantics embedded verbatim in the
  algorithm descriptors (via search/base) — it lives here so it cannot drift
  from the builder that interprets these rules.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from atlassian_rag_mcp.exceptions import FtsError
from atlassian_rag_mcp.storage.fts.normalize import (
    RU_STOPWORDS,
    WORD_PART_RE,
    extract_pattern_tokens,
    stem_query_string,
    stem_token,
)
from atlassian_rag_mcp.utils.query import (
    OptionalText,
    OptionalTextList,
)

_INDEX_FIELDS = ("text", "title", "pattern", "chunk_id", "page", "space")

FTS_QUERY_HELP = (
    "Lexical keyword query for the FTS (Tantivy) branch. Auto-detection:\n"
    "  - Plain text (no special chars) → (title:(w1 OR w2 ...))^3 OR "
    "text:(w1 OR w2 ...) — keywords OR-joined (Russian-stemmed, stopwords "
    "dropped), title matches boosted, FTS ranks; single word → (title:(w))^3 "
    "OR text:w. Per-term boost term^N (N=1..10, clamped): a PATTERNS-whitelisted "
    "code → exact branch pattern:(CODE)^N only (no title/text branches); any "
    "other term → (title:(...))^N. The N digits are never searched as text.\n"
    '  - Raw Tantivy query when the string contains : " ^ ~ * ? ( ) or AND/OR/NOT:\n'
    '      phrase "a b"~2, fuzzy term~1, wildcard te*t, boost term^3,\n'
    "      field text:/title:/page:/space:/pattern:, require-all "
    "text:(a AND b) — Cyrillic terms are stemmed to match the index.\n"
    "  - Strict validation: only the fields text/title/pattern/page/space exist; "
    "an unknown field prefix (chunk_id:x, pmp:x, http://...), an empty field "
    "value (text:) or unbalanced quotes/parens are rejected with an error.\n"
    "Omit to fall back to vector as the keyword source."
)
_ALLOWED_FIELDS = tuple(f for f in _INDEX_FIELDS if f != "chunk_id")
_FIELD_REF_RE = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*):(?=\S)")
_QUOTED_RE = re.compile(r'"[^"]*"')
_RAW_PATTERNS = (
    re.compile(rf"(?<![\w.])(?:{'|'.join(_INDEX_FIELDS)}):(?=\S)"),
    re.compile(r'"[^"]*"'),
    re.compile(r"\S\^\d+"),
    re.compile(r"(?<=\s)(?:AND|OR|NOT)(?=\s)"),
    re.compile(r"\w\*"),
)
_UNSAFE_TOKEN_RE = re.compile(r"[^\w.\-]", re.UNICODE)
_WORD_CHAR_RE = re.compile(r"\w", re.UNICODE)
_RESERVED_TERMS = frozenset({"AND", "OR", "NOT"})


def _expand_dotted_token(token: str) -> list[str]:
    """Return word parts for a dotted code token, or [token] otherwise.

    Reserved uppercase query keywords (AND/OR/NOT) are quoted so they stay
    literal terms inside the AND group.
    """
    if "." in token:
        parts = re.findall(r"\w+", token)
        if len(parts) > 1:
            return [f'"{p}"' if p.upper() in _RESERVED_TERMS else p for p in parts]
    return [token]


def sanitize_fts_value(value: str) -> str:
    """Sanitize a value for safe use inside a Tantivy double-quoted string."""
    if not value:
        return value
    return value.replace("\\", "\\\\").replace('"', '\\"')


def validate_fts_query(query: str) -> None:
    """Strict structural validation of a user FTS query (no index access).

    Raises ``FtsError`` on:
      * an ASCII ``field:`` prefix that is not an allowed field — allowed:
        text, title, pattern, page, space (so ``chunk_id:c1``, ``pmp:x``,
        ``http://...`` are rejected);
      * an allowed field with an empty value (``text:``);
      * unbalanced double quotes or parentheses (quoted-phrase content is
        ignored: ``text:"a:b"`` is fine).

    A non-ASCII prefix (``тема: ...``) is not a field reference — such text
    stays plain.
    """
    q = (query or "").strip()
    if not q:
        return
    if len(_QUOTED_RE.findall(q)) * 2 != q.count('"'):
        raise FtsError(f"Invalid FTS query: unbalanced double quotes in {q!r}")
    stripped = _QUOTED_RE.sub('""', q)
    for m in _FIELD_REF_RE.finditer(stripped):
        name = m.group(1)
        if name not in _ALLOWED_FIELDS:
            raise FtsError(
                f"Invalid FTS query: unknown field '{name}' in {q!r} — "
                f"allowed fields: {', '.join(_ALLOWED_FIELDS)}"
            )
    if re.search(rf"(?<![\w.])({'|'.join(_ALLOWED_FIELDS)}):\s*$", stripped):
        raise FtsError(f"Invalid FTS query: empty value for a field in {q!r}")
    if stripped.count("(") != stripped.count(")"):
        raise FtsError(f"Invalid FTS query: unbalanced parentheses in {q!r}")


def fts_looks_advanced(query: str) -> bool:
    """Heuristic: does the text look like raw Tantivy query language (not plain text)?

    Only explicit query syntax counts: known-index ``field:`` prefixes, quoted
    phrases, whole-word uppercase AND/OR/NOT and ``*`` wildcards. Per-term
    boost suffixes are stripped before detection: a plain keyword query with
    boosts (``тест^2 ABC.DEF.01^4``) stays plain — the builder
    handles ``^N`` itself — while ``text:abc^2`` is still raw. Everything else
    is plain keyword text.
    """
    return any(p.search(_strip_term_boosts(query)) for p in _RAW_PATTERNS)


def _quote_term(term: str) -> str:
    """Wrap a term in double quotes when it is a query operator or breaks an unquoted term."""
    if term.upper() in _RESERVED_TERMS or _UNSAFE_TOKEN_RE.search(term):
        return f'"{sanitize_fts_value(term)}"'
    return sanitize_fts_value(term)


def pattern_query_terms(raw: str) -> list[str]:
    """Exact pattern-token terms for plain keyword text (deduplicated).

    Every identifier matching the PATTERNS whitelist (ABC.DEF.01,
    release.2026.3.1, PROJ-1002) yields its exact ``pattern``-field
    term — a single indexed term distinct from related codes, so the query
    gets a unique boosted exact-match branch.
    """
    seen: set[str] = set()
    out: list[str] = []
    for t in extract_pattern_tokens(raw or ""):
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def prepare_query_terms(raw: str) -> list[str]:
    """Plain keyword text -> safe clause parts for the FTS index.

    Tokens are whitespace-split, then per token: dropped when it carries no
    word characters (or is a bare Russian stopword); identifier tokens
    (dotted/underscored/hyphenated: ``release.2026.3.1``, ``PMP_Ролевая_модель``)
    are decomposed into word parts ANDed together — the index tokenizer
    splits on dots/underscores/hyphens, so the whole raw term would otherwise
    read as an adjacent phrase and miss scattered mentions; every part is
    Russian-stemmed (codes/Latin/numbers pass through) to match the index
    normalization (``normalize_index_text``). Duplicates are removed, order
    is preserved. Returns ready-to-join parts (groups pre-parenthesized).
    """
    parts: list[str] = []
    seen: set[str] = set()
    for tok in (raw or "").split():
        word_parts = WORD_PART_RE.findall(tok)
        if not word_parts:
            continue
        if len(word_parts) == 1 and word_parts[0].lower() in RU_STOPWORDS:
            continue
        out: list[str] = []
        for p in word_parts:
            s = stem_token(p)
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(_quote_term(s))
        if not out:
            continue
        parts.append(f"({' AND '.join(out)})" if len(out) > 1 else out[0])
    return parts





_TERM_BOOST_RE = re.compile(r"\^(\d+)\Z")


def _split_term_boost(token: str) -> tuple[str, int | None]:
    """Split a trailing ``^N`` term boost: ``CODE.01^4`` -> (``CODE.01``, 4).

    Only trailing pure digits count as a boost suffix; N is clamped to 1..10.
    Anything else (no ``^N``, ``a^2b``, ``a^abc``) leaves the token unchanged.
    """
    m = _TERM_BOOST_RE.search(token) if token else None
    if not m:
        return token, None
    return token[: m.start()], min(max(int(m.group(1)), 1), 10)


def _strip_term_boosts(query: str) -> str:
    """Query text with all per-token ``^N`` boost suffixes removed.

    Used for advanced-syntax detection: ``тест^2 abc`` must not register as a
    raw Tantivy query just because a term carries a boost suffix.
    """
    return " ".join(
        clean
        for clean, _boost in (_split_term_boost(tok) for tok in (query or "").split())
    ).strip()


def split_boosted_tokens(query: str) -> tuple[list[str], list[tuple[str, int]]]:
    """Split plain query text into default tokens and ``(token, boost)`` pairs.

    Boost suffixes are stripped here, so the literal ``N`` digits never reach
    the index as search terms.
    """
    plain: list[str] = []
    boosted: list[tuple[str, int]] = []
    for tok in (query or "").split():
        clean, boost = _split_term_boost(tok)
        clean = clean.strip()
        if not clean:
            continue
        if boost is not None:
            boosted.append((clean, boost))
        else:
            plain.append(clean)
    return plain, boosted


def boosted_token_branches(token: str, boost: int) -> list[str]:
    """Query branches for a boosted token ``token^N`` — no default branches.

    A PATTERNS-whitelisted identifier (ABC.DEF.01, PROJ-1002,
    release.2026.3.1) goes into its exact branch only: ``pattern:(CODE)^N``.
    Any other token falls back to the title branch: ``(title:(...))^N``
    (word-part groups as :func:`prepare_query_terms`). Neither branch is the
    default ``^3`` keyword clause, so the suffix weight applies exactly here.
    """
    if extract_pattern_tokens(token) == [token]:
        return [f"pattern:({token})^{boost}"]
    return [f"(title:({g}))^{boost}" for g in prepare_query_terms(token)]


def kw_branches(
    parts: list[str],
    pattern_terms: list[str] | None = None,
    title_boost: int = 3,
) -> list[str]:
    """Branches of the default keyword clause: boosted title, text, patterns."""
    out: list[str] = []
    if parts:
        inner = " OR ".join(parts)
        out.append(f"(title:({inner}))^{title_boost}")
        out.append(f"text:({inner})")
    if pattern_terms:
        pattern_inner = " OR ".join(
            f'"{sanitize_fts_value(t)}"^{title_boost}' for t in pattern_terms
        )
        out.append(f"pattern:({pattern_inner})")
    return out


class FtsSearchParams(BaseModel):
    """Typed input for ``build_fts_query`` — only the FTS builder's fields.

    Same contract as the JQL/CQL/TQL builders' params models: all fields
    optional (an empty input builds an empty query — the caller decides
    whether a missing query is an error), unknown parameters (e.g.
    ``vector``) are rejected with a validation error naming the field.
    ``fts`` is the keyword source; ``spaces`` / ``page_ids`` restrict the
    FTS query. Text fields accept a string or a number (coerced to str);
    list fields accept a bare scalar or a list of scalars.
    """

    model_config = ConfigDict(extra="forbid")

    fts: OptionalText = None
    spaces: OptionalTextList = None
    page_ids: OptionalTextList = None


def build_fusion_query(
    query: str,
    *,
    title_boost: int = 3,
    spaces: list[str] | None = None,
    page_ids: list[str] | None = None,
) -> str:
    """Build an FTS query with a configurable title boost (thin wrapper).

    Single implementation is :func:`build_fts_query` — this is a convenience
    adapter used by the fusion algorithms that take ``title_boost`` as a
    parameter and already resolved the keyword source (``fts``/``vector``).
    """
    return build_fts_query(
        FtsSearchParams(fts=query, spaces=spaces, page_ids=page_ids),
        title_boost=title_boost,
    )


def build_fts_query(params: FtsSearchParams, *, title_boost: int = 3) -> str:
    """Build the FTS query string from :class:`FtsSearchParams`.

    Plain keyword text is normalized (stemmed, stopwords dropped) and joined
    into an OR clause over ``title`` (boosted), ``text`` and ``pattern``;
    PATTERNS-whitelisted codes get the exact-match ``pattern`` branch so the
    exact code outranks pages that only list related codes. A per-term boost
    suffix (``ABC.DEF.01^4``) replaces the default branches for that term;
    ``title_boost`` overrides the default exponent (3). Raw Tantivy syntax is
    passed through with Cyrillic tokens stemmed. ``spaces`` / ``page_ids``
    are embedded as ``AND (space:a OR ...)`` clauses. Strictly validated
    first (:func:`validate_fts_query`). Returns "" when nothing is set.
    """
    query = (params.fts or "").strip()
    space_keys = params.spaces or []
    page_ids = params.page_ids or []
    if not query and not space_keys and not page_ids:
        return ""
    builder = FtsBuilder()
    if query:
        validate_fts_query(query)
        if fts_looks_advanced(query):
            builder.raw(stem_query_string(query))
        else:
            plain, boosted = split_boosted_tokens(query)
            parts = prepare_query_terms(" ".join(plain))
            patterns = pattern_query_terms(" ".join(plain))
            branches = kw_branches(parts, patterns, title_boost=title_boost)
            for token, boost in boosted:
                branches.extend(boosted_token_branches(token, boost))
            if branches:
                builder.raw(f"({' OR '.join(branches)})")
    if space_keys:
        builder.space_keys(*space_keys)
    if page_ids:
        builder.page_ids(*page_ids)
    return builder.build()





class FtsBuilder:
    """Fluent Tantivy query builder with chainable clauses.

    Usage::

        q = FtsBuilder() \\
            .terms(["api", "docs"], conjunction=True) \\
            .phrase("exact phrase") \\
            .page_ids("p1", "p2") \\
            .space_keys("DEV") \\
            .build()

        # → text:(api AND docs) AND text:"exact phrase" AND
        #   (page:p1 OR page:p2) AND space:DEV

    Clauses are AND-combined. Empty/whitespace values are dropped silently.
    """

    def __init__(self) -> None:
        self._clauses: list[str] = []
        self._combiner = "AND"

    def terms(self, tokens: list[str], conjunction: bool = False) -> "FtsBuilder":
        """Keyword clause: ``text:(a OR b)`` (conjunction=True → ``text:(a AND b)``).

        Tokens containing characters that would break an unquoted Tantivy term
        (``:``/``/``/``{``/quotes/...) are wrapped in double quotes and matched
        by their analyzed tokens; tokens without word characters are dropped.
        Dotted code/version tokens (``a.b.10``, ``release.2026.3.1.1028``)
        are decomposed into word parts ANDed in a group: the index tokenizer
        splits on dots/underscores/hyphens, so the whole raw term would read
        as an adjacent phrase and miss scattered mentions (table cells, logs).
        """
        clean = [t for t in tokens if t and t.strip()]
        if not clean:
            return self
        safe = []
        for t in clean:
            t = t.strip()
            if not _WORD_CHAR_RE.search(t):
                continue
            parts = _expand_dotted_token(t)
            if len(parts) > 1:
                safe.append(f"({' AND '.join(parts)})")
            elif _UNSAFE_TOKEN_RE.search(t):
                safe.append(f'"{sanitize_fts_value(t)}"')
            else:
                safe.append(sanitize_fts_value(t))
        if not safe:
            return self
        inner = f" { 'AND' if conjunction else 'OR'} ".join(safe)
        clause = f"text:({inner})" if len(safe) > 1 else f"text:{safe[0]}"
        self._clauses.append(clause)
        return self

    def phrase(self, phrase: str) -> "FtsBuilder":
        """Phrase clause: ``text:"exact phrase"``."""
        value = phrase.strip() if phrase else ""
        if not value:
            return self
        self._clauses.append(f'text:"{sanitize_fts_value(value)}"')
        return self

    def page_id(self, page_id: str) -> "FtsBuilder":
        """Single value clause: ``page:<id>``."""
        value = page_id.strip() if page_id else ""
        if not value:
            return self
        self._clauses.append(f"page:{value}")
        return self

    def page_ids(self, *page_ids: str) -> "FtsBuilder":
        """Multi-value clause: ``(page:a OR page:b)``."""
        return self._multi("page", page_ids)

    def space_key(self, key: str) -> "FtsBuilder":
        """Single value clause: ``space:<key>``."""
        value = key.strip() if key else ""
        if not value:
            return self
        self._clauses.append(f"space:{value}")
        return self

    def space_keys(self, *keys: str) -> "FtsBuilder":
        """Multi-value clause: ``(space:a OR space:b)``."""
        return self._multi("space", keys)

    def raw(self, fragment: str) -> "FtsBuilder":
        """Raw fragment clause appended as-is (no auto-grouping)."""
        value = fragment.strip() if fragment else ""
        if not value:
            return self
        self._clauses.append(value)
        return self

    def _multi(self, field: str, values: tuple[str, ...]) -> "FtsBuilder":
        clean = []
        for v in values:
            value = v.strip() if v else ""
            if value:
                clean.append(f"{field}:{value}")
        if not clean:
            return self
        clause = " OR ".join(clean)
        self._clauses.append(f"({clause})" if len(clean) > 1 else clause)
        return self

    def build(self) -> str:
        """Assemble and return the final Tantivy query string (empty string if nothing set)."""
        return f" {self._combiner} ".join(self._clauses)
