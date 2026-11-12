"""Zephyr Scale TQL contract, query builder and shared method-parameter builders.

Single source for the TQL facts that agent-facing descriptions need — the
Zephyr counterpart of JQL_SCHEMA_HELP (jira/methods.py) and CQL_SYNTAX_HELP
(confluence/cql_builder.py). The three search methods embed TQL_HELP verbatim
so the search surfaces can never drift apart.

Query building follows the JQLBuilder/CQLBuilder pattern: TqlSearchParams
(typed, extra=forbid), tql_looks_advanced() and build_tql_query() — explicit
TQL syntax is validated and passed through verbatim; plain text is wrapped as
``<entity>.name = "..."`` (the only text-searchable condition); with no query
the condition is auto-built from project_id / folder_id. Unknown fields,
unsupported operators and function calls are rejected client-side (the query
never reaches the server) with an error naming the field and listing the
allowed ones.

TQL has no date conditions (createdOn/updatedOn filters and now() return 400,
live-verified) — the CQL/JQL date utilities are intentionally not used; the
only date syntax is ORDER BY testPlan.createdOn / updatedOn (a bare field
name).

The field contract is live-verified on the instance (see
docs/zephyr_tql_field_report.md). Constants below encode it:
- ``TQL_HELP`` — the L0 contract embedded verbatim into the search method
  descriptions; ``TQL_*_EXTRA`` — the per-entity addenda after it;
- ``PROJECT_KEY_DESC`` / ``QUERY_DESC`` / ... — shared parameter descriptions
  so the search methods and the project-scoped metadata methods never drift;
- ``_TQL_FIELDS`` — per-entity field/operator allow-list (cases also accept
  ``id IN`` and ``estimatedTime`` comparisons; ORDER BY: cases/runs — name
  only, plans — five fields);
- ``_TQL_ORDER_BY_RE`` — bare ORDER BY keyword marks raw (quoted values are
  stripped before detection, so ``name = "order by x"`` is not raw);
- ``_TQL_FUNC_RE`` — TQL has no functions; anything that looks like one stays
  on the raw path so validation rejects it instead of wrapping it as a name
  value.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from atlassian_rag_mcp.registry import MethodParameter
from atlassian_rag_mcp.utils.query import (
    OptionalText,
    split_top_level_order_by,
    validate_order_by,
)

TQL_HELP = (
    "TQL (Zephyr Scale query language) — live-verified on this instance:\n"
    "  - The `tql` parameter is used verbatim when it contains explicit TQL "
    "syntax (a field operator or ORDER BY, see the detail below) and overrides "
    "project_key/project_id/folder_id. Plain text without TQL syntax is wrapped "
    'as <entity>.name = "..." (exact-name wrap).\n'
    "  - Pagination: {results, total, startAt, maxResults} — total is the FULL "
    "match count (can be far larger than one page); large pages are accepted "
    "(verified up to 500 items); use offset to paginate.\n"
    "  - Results are sorted by name ascending by default; an explicit "
    "ORDER BY <entity>.name works (Russian-locale collation, not pure Unicode).\n"
    "  - Supported by all three entities: <entity>.projectId IN (…) / = / NOT IN, "
    "<entity>.folderTreeId IN (folder id from get_folders), <entity>.key IN "
    "(\"PROJ-123\"), <entity>.name = \"...\" (exact match only, no ~).\n"
    "  - Raw queries are validated client-side against the per-entity field "
    "allow-list (the query never reaches the server): unknown fields, "
    "unsupported operators (e.g. name ~) and function calls (now() etc.) are "
    "rejected with an error naming the field and listing the allowed ones.\n"
    "  - NOT supported (HTTP 400): status, statusId, createdOn, updatedOn, "
    "cf.*, customFieldValues() — filter those on the client side."
)

TQL_TESTCASE_EXTRA = (
    "testCase specifics: additionally supported testCase.id IN (…), "
    "testCase.estimatedTime > / >=; additionally NOT supported: priority, "
    "priorityId, labels, owner, folderId, archived, majorVersion."
)
TQL_TESTRUN_EXTRA = (
    "testRun specifics: additionally NOT supported: id, start, finish."
)
TQL_TESTPLAN_EXTRA = (
    "testPlan specifics: ORDER BY testPlan.name|id|key|createdOn|updatedOn "
    "ASC|DESC works (verified); additionally NOT supported: id."
)

PROJECT_KEY_DESC = (
    "Jira project key (accepts any of {project_keys}). "
    "Defaults to the first project in the list."
)
PROJECT_ID_DESC = "Zephyr project id (e.g., 12345, from get_projects). Overrides project_key."
QUERY_DESC = (
    "Custom TQL query. A query with explicit TQL syntax (a field operator or "
    "ORDER BY) overrides project_key/project_id/folder_id and is validated "
    "client-side (unknown fields are rejected with an error listing the "
    "allowed ones). Plain text without TQL syntax is wrapped as "
    '<entity>.name = "..." (exact match).'
)
LIMIT_DESC = (
    "Page size (default 25). Large pages are accepted — verified up to 500 "
    "items per page."
)
OFFSET_DESC = "Page offset (default 0). Use to paginate large result sets."
FIELDS_OVERRIDE_DESC = "Override the default TQL field list (rarely needed)."


def project_key_param() -> MethodParameter:
    """Fresh project_key parameter (get_methods() substitutes {project_keys}
    in place on every call, so descriptors must not alias shared objects)."""
    return MethodParameter(
        name="project_key",
        type="string",
        description=PROJECT_KEY_DESC,
    )


def zephyr_search_params(folder_note: str) -> list[MethodParameter]:
    """Shared search-method parameters: project_key, project_id, folder_id, tql.

    Single source for search_test_cases / search_test_runs /
    search_test_plans. ``folder_note`` is the per-entity folder_id
    description. Fresh copies per call.
    """
    return [
        project_key_param(),
        MethodParameter(name="project_id", type="integer", description=PROJECT_ID_DESC),
        MethodParameter(name="folder_id", type="integer", description=folder_note),
        MethodParameter(name="tql", type="string", description=QUERY_DESC),
    ]


def zephyr_page_params() -> list[MethodParameter]:
    """Shared limit/offset parameters (fresh copies per call)."""
    return [
        MethodParameter(name="limit", type="integer", description=LIMIT_DESC),
        MethodParameter(name="offset", type="integer", description=OFFSET_DESC),
    ]


_KEY_SOURCE = {
    "case": "search_test_cases",
    "run": "search_test_runs",
    "plan": "search_test_plans",
}


def key_param(kind: str) -> MethodParameter:
    """Test <kind> key parameter (kind: "case" / "run" / "plan"), required."""
    return MethodParameter(
        name=f"test_{kind}_key",
        type="string",
        description=(
            f"Test {kind} key (e.g., \"{{project_key}}-123\"; "
            f"project from {{project_keys}}), from {_KEY_SOURCE[kind]}."
        ),
        required=True,
    )


def fields_override_param() -> MethodParameter:
    """Shared fields-override parameter (fresh copy per call)."""
    return MethodParameter(name="fields", type="string", description=FIELDS_OVERRIDE_DESC)


TQL_ENTITIES: tuple[str, ...] = ("testCase", "testRun", "testPlan")

_TQL_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "testCase": {
        "projectId": ("IN", "NOT IN", "="),
        "folderTreeId": ("IN",),
        "key": ("IN",),
        "name": ("=",),
        "id": ("IN",),
        "estimatedTime": ("=", ">", ">=", "<", "<="),
    },
    "testRun": {
        "projectId": ("IN", "NOT IN", "="),
        "folderTreeId": ("IN",),
        "key": ("IN",),
        "name": ("=",),
    },
    "testPlan": {
        "projectId": ("IN", "NOT IN", "="),
        "folderTreeId": ("IN",),
        "key": ("IN",),
        "name": ("=",),
    },
}

_TQL_ORDER_FIELDS: dict[str, tuple[str, ...]] = {
    "testCase": ("name",),
    "testRun": ("name",),
    "testPlan": ("name", "id", "key", "createdOn", "updatedOn"),
}

_TQL_FIELDS_LC: dict[str, dict[str, str]] = {
    entity: {f.lower(): f for f in fields} for entity, fields in _TQL_FIELDS.items()
}
_TQL_ORDER_FIELDS_LC: dict[str, frozenset[str]] = {
    entity: frozenset(f.lower() for f in fields)
    for entity, fields in _TQL_ORDER_FIELDS.items()
}
_TQL_ORDER_FIELDS_CANON: dict[str, dict[str, str]] = {
    entity: {f.lower(): f for f in fields} for entity, fields in _TQL_ORDER_FIELDS.items()
}

_TQL_KEYWORDS = frozenset({"and", "or", "not", "in", "by", "asc", "desc", "order"})

_TQL_QUOTED_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_TQL_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)*")
_TQL_FIELD_OP_RE = re.compile(
    r"(?<![\w.])([A-Za-z_][A-Za-z0-9]*(?:\.[A-Za-z0-9]+)*)\s*"
    r"(!~|~|>=|<=|!=|=|>|<|\bNOT\s+IN\s*\(|\bIN\s*\()",
    re.IGNORECASE,
)

_TQL_ORDER_BY_RE = re.compile(r"(?<![\w.])order\s+by\s", re.IGNORECASE)

_TQL_FUNC_RE = re.compile(r"(?<![\w.])[A-Za-z_]\w*\s*\(")


def tql_looks_advanced(text: str) -> bool:
    """Heuristic: does the text look like a raw TQL fragment (not plain text)?

    Only explicit TQL syntax counts — a dotted/identifier field followed by a
    TQL operator (``=`` ``~`` ``>=`` ``IN (`` etc., :data:`_TQL_FIELD_OP_RE`),
    a top-level ``ORDER BY`` clause or a function-call shape (``now()``);
    TQL has no functions, so the latter is sent to validation and rejected
    with a clear error rather than wrapped as a name value. Quoted values are
    stripped first, so ``name = "order by x"`` does not count as ORDER BY.
    Bare ``IN`` without a parenthesis ("PMP IN PROGRESS") is NOT an operator —
    plain text. Natural text with a bare ``=``/``~`` between two words (e.g.
    "a = b") IS detected as raw TQL and then rejected by
    ``_validate_tql_fields`` if the field is not allowed (conservative: real
    TQL is never silently wrapped as a name value).
    """
    stripped = _TQL_QUOTED_RE.sub('""', text or "")
    return bool(
        _TQL_FIELD_OP_RE.search(stripped)
        or _TQL_ORDER_BY_RE.search(stripped)
        or _TQL_FUNC_RE.search(stripped)
    )


def sanitize_tql_value(value: str) -> str:
    """Sanitize a value for safe use inside a TQL double-quoted string.

    Double quotes are escaped with a backslash (TQL has no doubled-quote
    form); the backslash is escaped first to prevent a quote bypass.
    """
    if not value:
        return value
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _norm_field(entity: str, token: str) -> str | None:
    """Bare field name if *token* is a field of *entity* (else None).

    Accepts the canonical ``<entity>.<field>`` form (any case) and the bare
    ``<field>`` form.
    """
    t = token.lower()
    if "." in t:
        prefix, _, name = t.partition(".")
        return name if prefix == entity.lower() else None
    return t


def _check_field(entity: str, token: str) -> str:
    """Return the canonical bare field name or raise naming all allowed fields."""
    name = _norm_field(entity, token)
    canonical = _TQL_FIELDS_LC[entity].get(name) if name is not None else None
    if canonical is None:
        allowed = ", ".join(f"{entity}.{f}" for f in _TQL_FIELDS[entity])
        raise ValueError(
            f"Unknown TQL field: '{token}'. Allowed TQL fields for {entity}: {allowed}"
        )
    return canonical


def _validate_tql_order_by(entity: str, order: str) -> None:
    """Validate an ORDER BY clause: safe chars (shared with CQL/JQL) +
    the per-entity ORDER BY field allow-list."""
    clause = order
    if re.match(r"order\s+by\s", clause, re.IGNORECASE):
        clause = re.sub(r"^order\s+by\s", "", clause, count=1, flags=re.IGNORECASE)
    validate_order_by(clause)
    for part in clause.split(","):
        bits = part.split()
        if not bits:
            continue
        field = bits[0]
        name = _norm_field(entity, field)
        if name is None or name not in _TQL_ORDER_FIELDS_LC[entity]:
            allowed = ", ".join(f"{entity}.{f}" for f in _TQL_ORDER_FIELDS[entity])
            raise ValueError(
                f"Unknown TQL ORDER BY field: '{field}'. "
                f"Allowed ORDER BY fields for {entity}: {allowed}"
            )


def _validate_tql_fields(query: str, entity: str) -> None:
    """Validate a raw TQL query against the per-entity field allow-list.

    Field names are CASE-INSENSITIVE (the query is sent verbatim). Quoted
    values are not fields. Function calls (``now()`` etc.) are rejected —
    TQL has no functions (they return 400 on this instance). Raises
    ValueError on the first unknown field, unsupported operator or
    unsupported ORDER BY field (the query never reaches the server).
    """
    head, order = split_top_level_order_by(query)

    for m in _TQL_FIELD_OP_RE.finditer(head):
        field = m.group(1)
        op = m.group(2).strip().upper()
        if op.endswith("("):
            op = op[:-1].strip()  # "IN (" -> "IN", "NOT IN (" -> "NOT IN"
        canonical = _check_field(entity, field)
        if op not in _TQL_FIELDS[entity][canonical]:
            allowed = ", ".join(_TQL_FIELDS[entity][canonical])
            raise ValueError(
                f"Field [{entity}.{canonical}] does not support operator {op!r} "
                f"(allowed: {allowed})"
            )

    unquoted = _TQL_QUOTED_RE.sub("", head)
    for m in _TQL_TOKEN_RE.finditer(unquoted):
        token = m.group()
        if unquoted[m.end() : m.end() + 1] == "(":
            raise ValueError(
                f"Function calls are not supported in TQL: '{token}(...)' — "
                "TQL has no date functions (now() etc. are rejected by the server)."
            )
        if token.lower() in _TQL_KEYWORDS:
            continue
        _check_field(entity, token)

    if order:
        _validate_tql_order_by(entity, order)


def _numeric_id(value: int | str, name: str) -> int:
    """Ids are numeric; reject anything else before it reaches the query."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a numeric id, got: {value!r}")
    text = str(value).strip()
    if not text.isdigit():
        raise ValueError(f"{name} must be a numeric id, got: {value!r}")
    return int(text)


class TQLBuilder:
    """Fluent TQL builder scoped to a single entity (mirrors JQLBuilder/CQLBuilder).

    Usage::

        tql = TQLBuilder("testCase") \\
            .project_id(12345) \\
            .folder_id(555) \\
            .key("PROJ-123") \\
            .name("Verify the value") \\
            .order_by("name", "ASC") \\
            .build()
        # → testCase.projectId IN (12345) AND testCase.folderTreeId IN (555)
        #   AND testCase.key IN ("PROJ-123") AND testCase.name = "Verify the value"
        #   ORDER BY testCase.name ASC

    Operator policy (live-verified, see docs/zephyr_tql_field_report.md):
    project_id / key / folder_id always emit IN (even for a single value —
    for testRun/testPlan only IN is verified), name emits = (exact match
    only, no fuzzy). ``build()`` joins conditions with AND, hoists any
    top-level ORDER BY from raw fragments to the end (like JQLBuilder) and
    validates the final query against the entity field allow-list.
    """

    def __init__(self, entity: str = "testCase") -> "TQLBuilder":
        self._entity = next(
            (e for e in TQL_ENTITIES if e.lower() == entity.lower()), None
        )
        if self._entity is None:
            raise ValueError(
                f"Unknown TQL entity: '{entity}'. Allowed: {', '.join(TQL_ENTITIES)}"
            )
        self._conditions: list[str] = []
        self._order: str = ""

    def project_id(self, *ids: int | str) -> "TQLBuilder":
        """Project clause: ``<entity>.projectId IN (12345, ...)`` (numeric ids)."""
        if not ids:
            return self
        safe = ", ".join(str(_numeric_id(i, "project_id")) for i in ids)
        self._conditions.append(f"{self._entity}.projectId IN ({safe})")
        return self

    def folder_id(self, *ids: int | str) -> "TQLBuilder":
        """Folder clause: ``<entity>.folderTreeId IN (555, ...)`` (ids from get_folders)."""
        if not ids:
            return self
        safe = ", ".join(str(_numeric_id(i, "folder_id")) for i in ids)
        self._conditions.append(f"{self._entity}.folderTreeId IN ({safe})")
        return self

    def key(self, *keys: str) -> "TQLBuilder":
        """Key clause: ``<entity>.key IN ("PROJ-123", ...)`` (quoted keys)."""
        if not keys:
            return self
        quoted = ", ".join(f'"{sanitize_tql_value(k)}"' for k in keys)
        self._conditions.append(f"{self._entity}.key IN ({quoted})")
        return self

    def name(self, value: str) -> "TQLBuilder":
        """Name clause: ``<entity>.name = "value"`` — exact match only (no ~)."""
        safe = sanitize_tql_value(value)
        self._conditions.append(f'{self._entity}.name = "{safe}"')
        return self

    def raw(self, fragment: str) -> "TQLBuilder":
        """Append an arbitrary TQL fragment as-is (validated on build)."""
        self._conditions.append(fragment)
        return self

    def order_by(self, field: str, direction: str = "ASC") -> "TQLBuilder":
        """Ordering clause: ``ORDER BY <entity>.<field> ASC|DESC`` (entity allow-list)."""
        name = _norm_field(self._entity, field)
        if name is None or name not in _TQL_ORDER_FIELDS_LC[self._entity]:
            allowed = ", ".join(
                f"{self._entity}.{f}" for f in _TQL_ORDER_FIELDS[self._entity]
            )
            raise ValueError(
                f"Unknown TQL ORDER BY field: '{field}'. "
                f"Allowed ORDER BY fields for {self._entity}: {allowed}"
            )
        canonical = _TQL_ORDER_FIELDS_CANON[self._entity][name]
        safe_dir = direction.strip().upper()
        if safe_dir not in ("ASC", "DESC"):
            safe_dir = "ASC"
        self._order = f"ORDER BY {self._entity}.{canonical} {safe_dir}"
        return self

    def order_by_raw(self, clause: str) -> "TQLBuilder":
        """Ordering clause: raw ``ORDER BY`` clause (e.g. "testPlan.id, testPlan.name DESC").

        A leading ``ORDER BY`` in *clause* is dropped so the result is not
        doubled; the clause is validated against the entity allow-list.
        """
        clause = clause.strip()
        if re.match(r"^order\s+by\s", clause, re.IGNORECASE):
            clause = re.sub(r"^order\s+by\s", "", clause, count=1, flags=re.IGNORECASE)
        _validate_tql_order_by(self._entity, clause)
        self._order = f"ORDER BY {clause}"
        return self

    def build(self) -> str:
        """Assemble and return the final TQL query.

        Any real top-level ``ORDER BY`` embedded in a raw fragment is hoisted
        to the very end so later conditions keep the query valid (like
        JQLBuilder.build); the final query is validated against the entity
        field allow-list. Returns "" when nothing is set.
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
        if parts:
            _validate_tql_fields(parts, self._entity)
        return parts


class TqlSearchParams(BaseModel):
    """Typed input for ``build_tql_query`` (same contract as JQL/CQL builders).

    ``tql`` is a raw TQL fragment (validated against the per-entity field
    allow-list, passed through verbatim); ``project_id`` / ``folder_id`` are
    numeric ids (project-key resolution stays in the client — it is an async
    HTTP lookup); ``entity`` is testCase / testRun / testPlan. Unknown
    parameters are rejected with a validation error naming the field.
    """

    model_config = ConfigDict(extra="forbid")

    entity: str = "testCase"
    tql: OptionalText = None
    project_id: int | str | None = None
    folder_id: int | str | None = None


def build_tql_query(params: TqlSearchParams) -> str:
    """Build a TQL query from :class:`TqlSearchParams` (same contract as JQL/CQL builders).

    If ``tql`` is set and looks like raw TQL (:func:`tql_looks_advanced`) it
    is validated client-side (unknown fields, unsupported operators, function
    calls, bad ORDER BY are rejected — the query never reaches the server)
    and returned verbatim. A ``tql`` with no explicit TQL syntax is plain
    text: it is wrapped as the only text-searchable condition this language
    allows — ``<entity>.name = "..."`` (exact match only, no fuzzy ``~``).
    Otherwise the query is auto-built from project_id / folder_id:
    ``<entity>.projectId IN (...) AND <entity>.folderTreeId IN (...)``.
    Returns "" when nothing is set.
    """
    entity = next(
        (e for e in TQL_ENTITIES if e.lower() == params.entity.lower()), None
    )
    if entity is None:
        raise ValueError(
            f"Unknown TQL entity: '{params.entity}'. Allowed: {', '.join(TQL_ENTITIES)}"
        )
    query = params.tql or ""
    if query:
        if not tql_looks_advanced(query):
            query = f'{entity}.name = "{sanitize_tql_value(query)}"'
        _validate_tql_fields(query, entity)
        return query
    builder = TQLBuilder(entity)
    if params.project_id is not None:
        builder.project_id(params.project_id)
    if params.folder_id is not None:
        builder.folder_id(params.folder_id)
    return builder.build()
