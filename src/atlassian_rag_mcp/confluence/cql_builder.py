"""CQL query builder — sanitization, fluent API, strict field validation.

Shares the builder pattern with FtsBuilder/JQLBuilder: sanitize_cql_value(),
cql_looks_advanced(), the module-level build_cql_query() and a chainable
CQLBuilder. Instance: Confluence SERVER 7.3, classic Server/DC CQL
(fields live-verified, see docs/cql_fields_reference.md).

Live-verified contract encoded in the constants below:
- Raw CQL may use ONLY the allow-list (``_CQL_SIMPLE_FIELDS`` + dotted
  subfields in ``_CQL_SUBFIELDS`` + the triple pattern
  ``languages.<lang>.title``). Anything else is rejected client-side
  (``_validate_cql_fields``) with an error listing all allowed fields
  (``_ALLOWED_CQL_FIELDS``); the query never reaches the server.
- ``space.category`` passes the allow-list but the server answers HTTP 500
  for any query using it — surfaced as an error result, never swallowed.
- There is no IS operator: a bare ``is`` is rejected client-side (the server
  would answer HTTP 400 anyway).
- Validation is case-insensitive (tokens lowercased); the query is sent
  verbatim, in its original case.
- ``cql_looks_advanced``: an identifier or dotted subfield followed by a CQL
  operator (``= != ~ !~ < <= > >= IN ( NOT IN (``) or ``currentUser()`` marks
  raw CQL. Bare ``IN`` without a parenthesis is NOT an operator ("PMP IN
  PROGRESS" is plain text); an operator between two words ("a = b") is raw
  CQL too, so real CQL is never silently wrapped as text. ``title:x`` colon
  syntax is not supported — space-separated only.
- ``CQL_SYNTAX_HELP`` is the LLM-facing semantics used verbatim in the
  cql_search / cql_fts_fusion descriptors — it lives here so it cannot drift
  from the builder that interprets these rules.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from atlassian_rag_mcp.registry import MethodParameter
from atlassian_rag_mcp.utils.query import (
    OptionalText,
    OptionalTextList,
    as_list,
    build_date_condition,
    split_top_level_order_by,
    validate_order_by,
)

_CQL_SIMPLE_FIELDS = (
    "ancestor",
    "container",
    "content",
    "created",
    "contributor",
    "creator",
    "favorite",
    "favourite",
    "id",
    "label",
    "lastmodified",
    "languages",
    "macro",
    "mention",
    "parent",
    "refinedCategory",
    "refinedSite",
    "siteSearch",
    "space",
    "termGlobal",
    "termIdioms",
    "text",
    "title",
    "type",
    "user",
    "watcher",
)

_CQL_SIMPLE_FIELDS_LC = frozenset(f.lower() for f in _CQL_SIMPLE_FIELDS)

_CQL_SUBFIELDS: dict[str, tuple[str, ...]] = {
    "parent": ("title",),
    "space": ("category", "key", "title", "type"),
    "user": ("fullname", "userkey"),
    "watcher": ("fullname", "userkey"),
}

_CQL_TRIPLE_BASES = ("languages",)

_ALLOWED_CQL_FIELDS = ", ".join(
    list(_CQL_SIMPLE_FIELDS)
    + [f"{field}.{sub}" for field in sorted(_CQL_SUBFIELDS) for sub in _CQL_SUBFIELDS[field]]
    + ["languages.<lang>.title"]
)

_CQL_KEYWORDS = frozenset({"and", "or", "not", "in", "by", "asc", "desc", "order"})

_CQL_OPERATOR = r"(?:!~|~|>=|<=|!=|=|>|<|\bNOT IN\s*\(|\bIN\s*\()"
_CQL_FIELD_OP_RE = re.compile(
    rf"(?<![\w.])[A-Za-z_][A-Za-z0-9]*\s*{_CQL_OPERATOR}",
    re.IGNORECASE,
)
_CQL_SUBFIELD_OP_RE = re.compile(
    rf"(?<![\w.])[A-Za-z_][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)+\s*{_CQL_OPERATOR}",
    re.IGNORECASE,
)
_CURRENT_USER_RE = re.compile(r"currentUser\(\)", re.IGNORECASE)

_CQL_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)*")
_CQL_QUOTED_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_CQL_VALUE_AFTER_OP_RE = re.compile(
    r"(?:!~|~|>=|<=|!=|=|>|<|(?:\bNOT\s+)?\bIN\s*\()\s*$",
    re.IGNORECASE,
)

CQL_SYNTAX_HELP = (
    "Supports full CQL syntax. The instance is Confluence SERVER 7.3 (classic "
    "Server/DC CQL). Raw CQL in `cql` supports the full instance field set — the "
    "error on an unknown field lists all allowed fields. Field names are "
    "CASE-INSENSITIVE (the query is sent as-is).\n"
    '  - Fuzzy text: text ~ "ABC". IMPORTANT: a multi-token phrase matches ALL its '
    "tokens (AND semantics) — a natural multi-word question returns nothing; "
    "use ONE distinctive term, or cql_fts_fusion for wording questions.\n"
    "  - Operator limits (verified): created / lastmodified (dates) support ONLY "
    "<, <=, >, >=, =, != (no ~ / !~); title supports =, !=, ~, !~ and IN; label "
    "supports =, !=, IN, NOT IN; space supports =, IN, NOT IN; id / ancestor / "
    "parent take numeric page ids (= / IN).\n"
    '  - Negation: text !~ "...", title !~ "..."; NOT as "A AND NOT B" (leading '
    'NOT and bare "A NOT B" are not supported).\n'
    '  - Wildcards: text ~ "print*", "М?".\n'
    "  - Boolean: AND, OR, NOT.\n"
    "  - Colon syntax like title:x is NOT supported — space-separated only.\n"
    '  - Dates: created >= "2025-01-01" (ISO, quoted or not); functions now(), '
    'now("-7d"), now("-4w"), startOfYear(), endOfMonth() etc. work '
    "(- = the last N days; relative dates work ONLY inside now() — a bare -7d as a "
    "date value is rejected by this instance). today() is NOT supported (now() takes "
    "only relative dates, e.g. now(\"-4w\")).\n"
    "  - Functions: currentUser() — creator = currentUser(), "
    "contributor != currentUser().\n"
    "  - ORDER BY <field> ASC|DESC.\n"
    "Rejected client-side (the query never reaches the server): fields outside "
    "the instance field set (e.g. Cloud-only status, version.*, restriction.*, "
    "pageStatus) — the error names the unknown field and lists all allowed ones.\n"
    "Auto-detection: only explicit CQL (an identifier or dotted field + operator, "
    "or currentUser()) is validated and sent VERBATIM as raw; anything else is "
    'wrapped as text ~ "query".\n'
    "Optional — the query may be built from structured filters alone."
)

CQL_DATE_HELP = (
    "Natural-language dates (today, 2025-01-01, 31.12.2025) are parsed to ISO; "
    "CQL date functions (now(), now(\"-4w\"), startOfYear(), endOfMonth()) are passed "
    "through unquoted. Sign (verified live): - = the last N days — relative dates "
    "work ONLY inside now(): now(\"-7d\") = the last 7 days; a bare -7d as "
    "a date value is rejected by this instance (400). today() is NOT "
    "supported by this instance."
)

CQL_SPACE_FILTER_BASE = (
    "CQL space filter: restrict the search to these "
    "space keys (e.g. {space_keys}) — space = \"KEY\" / "
    "space IN (...)."
)


def cql_filter_params(space_note: str = "") -> list[MethodParameter]:
    """Structured CQL filter parameters shared by the CQL search algorithms.

    Single source for the title/label/creator/contributor/space/created/
    lastmodified/id/ancestor/parent descriptions (cql_search and
    cql_fts_fusion). ``space_note`` is an optional per-algorithm suffix for the
    ``space`` parameter (e.g. the branch-scope note for the fusion algorithm).
    """
    space_desc = CQL_SPACE_FILTER_BASE + ((" " + space_note) if space_note else "")
    return [
        MethodParameter(
            name="title",
            type="string",
            description='CQL filter: fuzzy title match `title ~ "..."` (substring, NOT exact phrase).',
        ),
        MethodParameter(name="label", type="array", description="CQL filter by labels: label IN (...)"),
        MethodParameter(
            name="creator",
            type="string",
            description="CQL filter by creator (username or 'me'/currentUser()): creator = \"...\"",
        ),
        MethodParameter(
            name="contributor",
            type="string",
            description="CQL filter by contributor (username or 'me'/currentUser()): contributor = \"...\"",
        ),
        MethodParameter(
            name="space",
            type="array",
            description=space_desc,
        ),
        MethodParameter(
            name="created",
            type="string",
            description=f"CQL filter created >= date. {CQL_DATE_HELP}",
        ),
        MethodParameter(
            name="lastmodified",
            type="string",
            description=f"CQL filter lastmodified >= date. {CQL_DATE_HELP}",
        ),
        MethodParameter(
            name="id",
            type="array",
            description="CQL filter by page id: id IN (123, 456) — batch "
            "selection / scoping to exact pages (numeric page ids).",
        ),
        MethodParameter(
            name="ancestor",
            type="string",
            description="CQL filter by ancestor page id: ancestor = <id> — "
            "restrict to the whole subtree (all descendants) of the page.",
        ),
        MethodParameter(
            name="parent",
            type="string",
            description="CQL filter by parent page id: parent = <id> — "
            "restrict to the direct children of the page.",
        ),
    ]


_CURRENT_USER_ALIASES = ("me", "currentuser", "current_user", "currentuser()")


def sanitize_cql_value(value: str) -> str:
    """Sanitize a value for safe use inside a CQL double-quoted string.

    Double quotes are escaped with a backslash — Confluence CQL rejects the
    doubled-quote form ("a""b" -> "The query cannot be parsed").
    """
    if not value:
        return value

    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    return value


def build_space_cql(keys: list[str] | None) -> str:
    """Build a CQL ``space`` condition from a list of space keys.

    Empty/None list → ``""`` (no space restriction, search all spaces).
    Single key → ``space = "KEY"``; multiple → ``space IN ("A", "B")``.
    """
    keys = [k for k in as_list(keys) if k and k.strip()]
    if not keys:
        return ""
    keys = [k.strip() for k in keys]
    if len(keys) == 1:
        return f'space = "{sanitize_cql_value(keys[0])}"'
    quoted = '", "'.join(sanitize_cql_value(k) for k in keys)
    return f'space IN ("{quoted}")'


class CQLBuilder:
    """Fluent CQL builder with chainable filters.

    Usage::

        cql = CQLBuilder() \\
            .space("PROJ") \\
            .text_contains("API documentation") \\
            .order_by("lastmodified", "DESC") \\
            .build()

        # → space = "PROJ" AND text ~ "API documentation" ORDER BY lastmodified DESC
    """

    def __init__(self) -> None:
        self._conditions: list[str] = []
        self._order: str = ""

    def type(self, content_type: str) -> "CQLBuilder":
        """Content type clause: ``type = "page"`` (e.g. page, blogpost)."""
        safe = sanitize_cql_value(content_type)
        self._conditions.append(f'type = "{safe}"')
        return self

    def space(self, key: str) -> "CQLBuilder":
        """Single space clause: ``space = "KEY"``."""
        safe = sanitize_cql_value(key)
        self._conditions.append(f'space = "{safe}"')
        return self

    def spaces(self, *keys: str) -> "CQLBuilder":
        """Space clause: ``space = "KEY"`` or ``space IN ("A", "B")``.

        No keys → no condition added (search all spaces).
        """
        cond = build_space_cql(list(keys))
        if cond:
            self._conditions.append(cond)
        return self

    def text_contains(self, text: str) -> "CQLBuilder":
        """Text search clause: ``text ~ "value"``."""
        safe = sanitize_cql_value(text)
        self._conditions.append(f'text ~ "{safe}"')
        return self

    def title(self, title: str) -> "CQLBuilder":
        """Title clause: ``title ~ "value"``."""
        safe = sanitize_cql_value(title)
        self._conditions.append(f'title ~ "{safe}"')
        return self

    def label(self, *labels: str) -> "CQLBuilder":
        """Multi-value label clause: ``label IN ("a", "b")``."""
        safe_list = [sanitize_cql_value(label) for label in labels]
        quoted = '", "'.join(safe_list)
        self._conditions.append(f'label IN ("{quoted}")')
        return self

    def id_in(self, *ids: str) -> "CQLBuilder":
        """Multi-value clause: ``id IN (id1, id2)``."""
        self._conditions.append(f"id IN ({','.join(ids)})")
        return self

    def ancestor(self, page_id: str) -> "CQLBuilder":
        """Descendants clause: ``ancestor = <page id>`` (unquoted numeric id)."""
        self._conditions.append(f"ancestor = {page_id}")
        return self

    def parent(self, page_id: str) -> "CQLBuilder":
        """Direct-parent clause: ``parent = <page id>`` (unquoted numeric id)."""
        self._conditions.append(f"parent = {page_id}")
        return self

    def raw(self, fragment: str) -> "CQLBuilder":
        """Append an arbitrary CQL fragment as-is (no auto-grouping)."""
        self._conditions.append(fragment)
        return self

    @staticmethod
    def _user_condition(field: str, username: str) -> str:
        """Build a user-valued condition, mapping 'me'/'currentUser' to the CQL function."""
        if username.strip().lower() in _CURRENT_USER_ALIASES:
            return f"{field} = currentUser()"
        safe = sanitize_cql_value(username)
        return f'{field} = "{safe}"'

    def creator(self, username: str) -> "CQLBuilder":
        """Creator clause: ``creator = "alice"``. Pass ``"me"``/``"currentUser"`` for currentUser()."""
        self._conditions.append(self._user_condition("creator", username))
        return self

    def contributor(self, username: str) -> "CQLBuilder":
        """Contributor clause: ``contributor = "alice"``. Pass ``"me"``/``"currentUser"`` for currentUser()."""
        self._conditions.append(self._user_condition("contributor", username))
        return self

    def created(self, date_str: str) -> "CQLBuilder":
        """Date clause: ``created >= "YYYY-MM-DD"`` (accepts natural language).

        CQL date functions (``now()``/``now("-4w")``/``startOfYear()``/...) are
        emitted unquoted — inside quotes the server cannot parse them.
        """
        self._conditions.append(build_date_condition("created", date_str, sanitize_cql_value))
        return self

    def lastmodified(self, date_str: str) -> "CQLBuilder":
        """Date clause: ``lastmodified >= "YYYY-MM-DD"`` (accepts natural language).

        CQL date functions (``now()``/``now("-4w")``/``startOfYear()``/...) are
        emitted unquoted — inside quotes the server cannot parse them.
        """
        self._conditions.append(build_date_condition("lastmodified", date_str, sanitize_cql_value))
        return self

    def order_by(self, field: str, direction: str = "DESC") -> "CQLBuilder":
        """Ordering clause: ``ORDER BY field DESC``."""
        validate_order_by(field)
        safe_dir = direction.strip().upper()
        if safe_dir not in ("ASC", "DESC"):
            safe_dir = "DESC"
        self._order = f"ORDER BY {field.strip()} {safe_dir}"
        return self

    def build(self) -> str:
        """Assemble and return the final CQL string.

        Any real top-level ``ORDER BY`` clause embedded in a raw fragment is hoisted
        to the very end so later ``AND`` conditions keep the query valid.
        """
        order = self._order
        conditions: list[str] = []
        for cond in self._conditions:
            head, found = split_top_level_order_by(cond)
            if found:
                order = found
            if head:
                conditions.append(head)
        parts = " AND ".join(conditions)
        if order:
            parts = f"{parts} {order}" if parts else order
        return parts


def cql_looks_advanced(text: str) -> bool:
    """Heuristic: does the text look like a raw CQL fragment (not plain text)?

    Any identifier or dotted subfield followed by a CQL operator (``= != ~ !~
    < <= > >= IN ( NOT IN (``) or ``currentUser()``. Bare ``IN`` without a
    parenthesis is NOT an operator — "PMP IN PROGRESS" is plain text. Natural
    text with a bare ``=``/``~``/``>``/``<`` between two words (e.g. "a = b") IS
    detected as raw CQL and then rejected by ``_validate_cql_fields`` if the
    field is not allowed (conservative: real CQL is never silently wrapped as
    text).
    """
    return bool(_CQL_FIELD_OP_RE.search(text) or _CQL_SUBFIELD_OP_RE.search(text) or _CURRENT_USER_RE.search(text))


def _validate_cql_fields(cql: str) -> None:
    """Validate that a raw CQL fragment uses only allow-listed CQL fields.

    Field names are compared case-insensitively (lowercased); the original
    string is what gets sent to the server. Quoted values, CQL keywords and
    function calls (``now()``, ``currentUser()``, ...) are not fields. Values
    that immediately follow an operator (unquoted values: ``type = page``,
    ``space IN (PROJ)``, ``created >= 2025-01-01``) are skipped too — the
    server accepts unquoted values (live-verified) and only the tokens used
    as field names are validated. Raises ValueError on the first unknown
    field, listing all allowed fields.
    """
    unquoted = _CQL_QUOTED_RE.sub("", cql)
    for match in _CQL_TOKEN_RE.finditer(unquoted):
        if unquoted[match.end() : match.end() + 1] == "(":
            continue  # function call, not a field
        if _CQL_VALUE_AFTER_OP_RE.search(unquoted, 0, match.start()):
            continue  # value after operator, not a field
        token = match.group()
        lowered = token.lower()
        if lowered in _CQL_KEYWORDS:
            continue
        parts = lowered.split(".")
        if len(parts) == 1 and parts[0] in _CQL_SIMPLE_FIELDS_LC:
            continue
        if len(parts) == 2 and parts[1] in _CQL_SUBFIELDS.get(parts[0], ()):
            continue
        if len(parts) == 3 and parts[0] in _CQL_TRIPLE_BASES and parts[2] == "title":
            continue
        raise ValueError(f"Unknown CQL field: '{token}'. Allowed CQL fields: {_ALLOWED_CQL_FIELDS}")


class CqlSearchParams(BaseModel):
    """Typed input for ``build_cql_query`` (same contract as JQL/FTS builders).

    Parameter names are exactly the CQL query field names (``space``, ``title``,
    ``label``, ``creator``, ``contributor``, ``created``, ``lastmodified``,
    ``id``, ``ancestor``, ``parent``) plus the raw query text ``cql``. All
    fields are optional; unknown parameters are rejected with a validation
    error naming the field. Text fields accept a string or a number (coerced
    to str) and reject bool/list/dict values; list fields accept a bare scalar
    (single-item list) or a list of scalars. Date fields (``created`` /
    ``lastmodified``) accept natural-language dates (``today``,
    ``2025-01-01``) or CQL date functions (``now("-7d")``, ``startOfYear()``),
    which are parsed to ISO dates or passed through for the server to handle.
    Id fields (``id``,
    ``ancestor``, ``parent``) accept numeric page ids only.
    """

    model_config = ConfigDict(extra="forbid")

    cql: OptionalText = None
    space: OptionalTextList = None
    title: OptionalText = None
    label: OptionalTextList = None
    creator: OptionalText = None
    contributor: OptionalText = None
    created: OptionalText = None
    lastmodified: OptionalText = None
    id: OptionalTextList = None
    ancestor: OptionalText = None
    parent: OptionalText = None


def _validate_page_id(value: str, name: str) -> str:
    """Page ids are numeric; reject anything else before it reaches the query."""
    value = str(value).strip()
    if not value.isdigit():
        raise ValueError(f"{name} must be a numeric page id, got: {value!r}")
    return value


def _apply_filters(builder: CQLBuilder, params: CqlSearchParams) -> None:
    """Apply structured CQL filters to the builder (AND-combined)."""
    if params.title:
        builder.title(params.title)
    if params.label:
        builder.label(*params.label)
    if params.creator:
        builder.creator(params.creator)
    if params.contributor:
        builder.contributor(params.contributor)
    if params.created:
        builder.created(params.created)
    if params.lastmodified:
        builder.lastmodified(params.lastmodified)
    if params.id:
        builder.id_in(*(_validate_page_id(i, "id") for i in params.id))
    if params.ancestor:
        builder.ancestor(_validate_page_id(params.ancestor, "ancestor"))
    if params.parent:
        builder.parent(_validate_page_id(params.parent, "parent"))


def build_cql_query(params: CqlSearchParams) -> str:
    """Build a CQL query string from :class:`CqlSearchParams` (cql, space, filters).

    Plain text (no CQL operators) is wrapped as ``text ~ "query"``; raw CQL
    fragments are validated against the CQL field allow-list (case-insensitive)
    and appended verbatim. Returns "" when nothing is set.
    """
    text = params.cql or ""
    if (
        not text
        and not params.space
        and not any(
            (
                params.title,
                params.label,
                params.creator,
                params.contributor,
                params.created,
                params.lastmodified,
                params.id,
                params.ancestor,
                params.parent,
            )
        )
    ):
        return ""
    builder = CQLBuilder().type("page").spaces(*(params.space or []))
    _apply_filters(builder, params)
    if text:
        if cql_looks_advanced(text):
            _validate_cql_fields(text)
            builder.raw(text)
        else:
            builder.text_contains(text)
    return builder.build()
