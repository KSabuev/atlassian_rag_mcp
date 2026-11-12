"""JQL query builder — sanitization, fluent API, strict field validation.

Counterpart of CQLBuilder/FtsBuilder: sanitize_jql_value(), jql_looks_advanced(),
module-level build_jql_query(), chainable JQLBuilder. Date parsing and ORDER BY
handling are shared via utils.query.

``JqlSearchParams`` fields are the JQL query field names (project, issuetype,
status, priority, assignee, labels, created, updated, duedate, resolved) plus
the raw query text ``jql`` and the ``order_by`` clause; unknown keys are
rejected. ``JQL_SCHEMA_HELP`` is the LLM-facing semantics embedded verbatim in
the search/count_issues descriptors — it lives here so it cannot drift from
the builder that interprets these rules.

Instance-specific rules (verified on this Jira build):
- Empty parentheses break the text function ("Невозможно разобрать текст ...
  для функции «text»"); sanitization strips them.
- Double quotes are escaped with a backslash — Jira rejects the doubled-quote
  form ("a""b" -> "expected ',' or ')'").
- Verified fields: assignee, comment, component, created, description, duedate,
  fixVersion, id, issuekey, issuetype, key, labels, priority, project, reporter,
  resolved (alias of resolutiondate), resolution, sprint, status, summary, text,
  updated. ``label`` (singular) and ``epic`` do NOT exist (only ``labels``;
  epics are the "Epic Link" custom field).
- ``in`` requires a list parenthesis or openSprints(); ``is`` requires
  (NOT) EMPTY — ``field is value`` is rejected, and English phrases like
  "created in 2025" must stay plain text. Keywords are case-insensitive.
- A quoted custom field + operator counts as JQL ("Epic Link" is not EMPTY);
  openSprints() is only valid attached to a field.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from atlassian_rag_mcp.utils.query import (
    OptionalText,
    OptionalTextList,
    build_date_condition,
    has_top_level_order_by,
    split_top_level_order_by,
    validate_order_by,
)

_CURRENT_USER_ALIASES = ("me", "currentuser", "current_user", "currentuser()")

_EMPTY_PAREN_RE = re.compile(r"(\s*)\(\s*\)(\s*)")

JQL_SCHEMA_HELP: str = (
    "Query language auto-detection (strict — verified on this instance):\n"
    "  - Plain free text → text ~ \"...\" (JQL text search). A multi-word phrase\n"
    "    matches ALL its tokens (AND semantics, no stopword filtering) — a natural\n"
    "    multi-word question can match nothing; use one distinctive term or a raw\n"
    "    `text ~ \"a\" OR text ~ \"b\"` for OR.\n"
    "  - Raw JQL passes through verbatim ONLY when it contains explicit JQL:\n"
    "    a known field + operator (project =, status =, issuetype =, priority =,\n"
    "    labels IN / NOT IN, assignee = / is, reporter =, created/updated/resolved/\n"
    "    duedate >= / <=, summary ~, description ~, comment ~, text ~, key =, id in,\n"
    "    sprint in openSprints(), component is not EMPTY, fixVersion is ...),\n"
    "    a quoted custom field + operator (\"Epic Link\" is not EMPTY), or\n"
    "    currentUser(). Raw JQL is AND-ed with any structured filters.\n"
    "  - Instance quirks (verified): text supports only ~ (no =, no !~, no IN);\n"
    "    NOT (text ~ \"x\") returns empty — negate with summary !~ / description !~;\n"
    "    empty parentheses () inside a plain text value break the query and are\n"
    "    stripped automatically; `is` accepts only (NOT) EMPTY, not values;\n"
    "    Russian status/priority names fail (auto-retried with numeric IDs) and\n"
    "    Cyrillic issuetype names silently match nothing (auto-retried with IDs);\n"
    "    `label` (singular) and `epic` fields do not exist.\n"
    "  - Scoped to the configured project ({project_key}) unless the query already\n"
    "    contains a project = / project IN clause (raw project wins).\n"
    "  - Status / priority / issue-type names that Jira rejects or match nothing are\n"
    "    automatically retried with their numeric IDs.\n"
    "  - Dates accept ISO YYYY-MM-DD, European DD.MM.YYYY and natural language:\n"
    "    today, yesterday, -7d (offsets relative to now, going back: also -2w, -1m,\n"
    "    -1y; now() works, but now() - 30d arithmetic and dates() do not).\n"
    "  - When no ORDER BY is present, ORDER BY priority DESC, updated DESC is\n"
    "    appended automatically (override with the order_by parameter; a top-level\n"
    "    ORDER BY inside raw JQL wins)."
)


def _strip_empty_parens(value: str) -> str:
    """Remove ``()`` / ``( )`` keeping word separation intact.

    ``openSprints()`` -> ``openSprints``, ``get_user() f`` -> ``get_user f``,
    ``a ( ) b`` -> ``a b``.
    """

    def _repl(m: re.Match[str]) -> str:
        return " " if (m.group(1) or m.group(2)) else ""

    return _EMPTY_PAREN_RE.sub(_repl, value)


def sanitize_jql_value(value: str) -> str:
    """Sanitize a value for safe use inside a JQL double-quoted string.

    Strips empty parentheses first (they make the Jira text function
    unparseable), then escapes the backslash (before the quote, to prevent a
    quote bypass) and the double quote.
    """
    if not value:
        return value

    value = _strip_empty_parens(value)
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    return value


def sanitize_jql_list(values: list[str]) -> list[str]:
    """Sanitize a list of values for use in JQL."""
    return [sanitize_jql_value(v) for v in values]


_JQL_FIELDS = (
    "assignee|comment|component|created|description|duedate|fixVersion|id|"
    "issuekey|issuetype|key|labels|priority|project|reporter|resolved|resolution|"
    "sprint|status|summary|text|updated"
)

_JQL_FIELD_OP_RE = re.compile(
    rf"(?<![\w.])(?:{_JQL_FIELDS})\s*(?:NOT\s+)?(?:!~|~|>=|<=|!=|=|>|<|\bNOT\s+IN\s*(?:\(|openSprints\s*\()|\bIN\s*(?:\(|openSprints\s*\()|\bIS\s+(?:NOT\s+)?EMPTY\b)",
    re.IGNORECASE,
)
_JQL_CUSTOM_FIELD_RE = re.compile(
    r'"(?:[^"\\]|\\.)*"\s*(?:NOT\s+)?(?:!~|~|>=|<=|!=|=|>|<|\bNOT\s+IN\s*(?:\(|openSprints\s*\()|\bIN\s*(?:\(|openSprints\s*\()|\bIS\s+(?:NOT\s+)?EMPTY\b)',
    re.IGNORECASE,
)
_JQL_FUNC_RE = re.compile(r"(?<![\w])currentUser\s*\(", re.IGNORECASE)


def jql_looks_advanced(text: str) -> bool:
    """Heuristic: does the text look like a raw JQL fragment (vs plain text search)?

    Only explicit JQL syntax counts: a known field followed by an operator
    (``status =``, ``labels IN (``, ``sprint in openSprints()``,
    ``assignee is EMPTY``, ``created >=`` ...), a quoted custom field +
    operator, or ``currentUser()``. Everything else is plain text wrapped as
    ``text ~ "..."``.
    """
    return bool(
        _JQL_FIELD_OP_RE.search(text)
        or _JQL_CUSTOM_FIELD_RE.search(text)
        or _JQL_FUNC_RE.search(text)
    )


class JqlSearchParams(BaseModel):
    """Typed input for ``build_jql_query``.

    Parameter names are exactly the JQL query field names (``project``,
    ``issuetype``, ``status``, ``priority``, ``assignee``, ``labels``,
    ``created``, ``updated``, ``duedate``, ``resolved``) plus the raw query
    text ``jql`` and the ORDER BY clause ``order_by``. All fields are optional;
    unknown parameters are rejected with a validation error naming the field.
    Text fields accept a string or a number (coerced to str); ``status`` /
    ``priority`` / ``labels`` accept a bare scalar or a list; date fields
    (``created`` / ``updated`` / ``duedate`` / ``resolved``) accept
    natural-language dates (``today``, ``-7d``, ``2025-01-01``) and date functions
    (``now()``, ``startOfYear()``, ... — passed through unquoted) and build
    ``field >= date`` conditions.
    """

    model_config = ConfigDict(extra="forbid")

    jql: OptionalText = None
    project: OptionalText = None
    issuetype: OptionalText = None
    status: OptionalTextList = None
    priority: OptionalTextList = None
    assignee: OptionalText = None
    labels: OptionalTextList = None
    created: OptionalText = None
    updated: OptionalText = None
    duedate: OptionalText = None
    resolved: OptionalText = None
    order_by: OptionalText = None


def _has_project_filter(text: str) -> bool:
    """Check if JQL already contains a ``project`` filter (case-insensitive).

    Strips quoted strings to avoid false positives on values containing ``project``.
    """
    stripped = re.sub(r'"[^"]*"', "", text)
    return bool(re.search(r"\bproject\s*(?:IN\s*\(|=)", stripped, re.IGNORECASE))


_DEFAULT_ORDER_BY = "priority DESC, updated DESC"


def build_jql_query(params: JqlSearchParams) -> str:
    """Build a JQL query from :class:`JqlSearchParams` (same contract as CQL/FTS builders).

    Reads ``jql`` (raw JQL or plain free text), ``project`` and the
    structured filters from ``params``.

    If the text looks like raw JQL, pass it through as-is. Otherwise treat it
    as a natural-language query and wrap in a text search.

    Structured filters are combined with ``AND`` using the :class:`JQLBuilder`
    fluent API (``status``, ``priority``, ``assignee``, date fields, ...).
    Accepts natural language for dates (``today``, ``-7d``, ``2025-01-01``).

    If *project* is set and the query doesn't already contain a ``project``
    filter, the project scope is added automatically (analogous to how
    ``build_cql_query`` adds ``space`` from settings). When no ORDER BY is
    present (and ``order_by`` is not set), the default
    ``ORDER BY priority DESC, updated DESC`` is appended.
    """
    text = params.jql or ""
    project = params.project

    b = JQLBuilder()

    if jql_looks_advanced(text):
        b.raw(text)
    elif text:
        b.text_contains(text)

    if params.issuetype:
        b.issuetype(params.issuetype)
    if params.status:
        b.status(*params.status)
    if params.priority:
        b.priority(*params.priority)
    if params.assignee:
        b.assignee(params.assignee)
    if params.labels:
        b.labels(*params.labels)
    if params.created:
        b.created(params.created)
    if params.updated:
        b.updated(params.updated)
    if params.duedate:
        b.duedate(params.duedate)
    if params.resolved:
        b.resolved(params.resolved)

    query = b.build()
    if project and not _has_project_filter(query):
        b.project(project)

    if b.build() and not has_top_level_order_by(b.build()):
        b.order_by_raw(params.order_by or _DEFAULT_ORDER_BY)
    return b.build()


class JQLBuilder:
    """Fluent JQL builder with chainable filters.

    Usage::

        jql = JQLBuilder() \\
            .project("PROJ") \\
            .status("In Progress") \\
            .priority("High") \\
            .updated("today") \\
            .order_by("created", "DESC") \\
            .build()

        # → project = "PROJ" AND status = "In Progress" AND ...
    """

    def __init__(self) -> None:
        self._conditions: list[str] = []
        self._order: str = ""

    def project(self, key: str) -> "JQLBuilder":
        """Project clause: ``project = "PROJ"``."""
        safe = sanitize_jql_value(key)
        self._conditions.append(f'project = "{safe}"')
        return self

    def issuetype(self, name: str) -> "JQLBuilder":
        """Issue type clause: ``issuetype = "Bug"``."""
        safe = sanitize_jql_value(name)
        self._conditions.append(f'issuetype = "{safe}"')
        return self

    def assignee(self, value: str) -> "JQLBuilder":
        """Assignee clause: ``assignee = "alice"``.

        Pass ``"currentUser"``/``"me"`` for ``assignee = currentUser()``,
        ``"empty"`` for ``assignee is EMPTY`` (unassigned).
        """
        normalized = value.strip().lower()
        if normalized in _CURRENT_USER_ALIASES:
            self._conditions.append("assignee = currentUser()")
        elif normalized in ("empty",):
            self._conditions.append("assignee is EMPTY")
        else:
            safe = sanitize_jql_value(value)
            self._conditions.append(f'assignee = "{safe}"')
        return self

    def status(self, *names: str) -> "JQLBuilder":
        """Status clause: ``status = "X"`` or ``status IN ("X", "Y")``."""
        if not names:
            return self
        safe_list = sanitize_jql_list(list(names))
        if len(safe_list) == 1:
            self._conditions.append(f'status = "{safe_list[0]}"')
        else:
            quoted = '", "'.join(safe_list)
            self._conditions.append(f'status IN ("{quoted}")')
        return self

    def priority(self, *names: str) -> "JQLBuilder":
        """Priority clause: ``priority = "X"`` or ``priority IN ("X", "Y")``."""
        if not names:
            return self
        safe_list = sanitize_jql_list(list(names))
        if len(safe_list) == 1:
            self._conditions.append(f'priority = "{safe_list[0]}"')
        else:
            quoted = '", "'.join(safe_list)
            self._conditions.append(f'priority IN ("{quoted}")')
        return self

    def labels(self, *labels: str) -> "JQLBuilder":
        """Multi-value label clause: ``labels IN ("a", "b")``."""
        safe_list = sanitize_jql_list(list(labels))
        quoted = '", "'.join(safe_list)
        self._conditions.append(f'labels IN ("{quoted}")')
        return self

    def updated(self, date_str: str) -> "JQLBuilder":
        """Date clause: ``updated >= "YYYY-MM-DD"`` (accepts natural language).

        Date functions (``now()``/``startOfYear()``/...) are emitted unquoted —
        inside quotes the server reads them as a plain string.
        """
        self._conditions.append(build_date_condition("updated", date_str, sanitize_jql_value))
        return self

    def created(self, date_str: str) -> "JQLBuilder":
        """Date clause: ``created >= "YYYY-MM-DD"`` (accepts natural language).

        Date functions (``now()``/``startOfYear()``/...) are emitted unquoted —
        inside quotes the server reads them as a plain string.
        """
        self._conditions.append(build_date_condition("created", date_str, sanitize_jql_value))
        return self

    def duedate(self, date_str: str) -> "JQLBuilder":
        """Date clause: ``duedate >= "YYYY-MM-DD"`` (accepts natural language).

        Date functions (``now()``/``startOfYear()``/...) are emitted unquoted —
        inside quotes the server reads them as a plain string.
        """
        self._conditions.append(build_date_condition("duedate", date_str, sanitize_jql_value))
        return self

    def resolved(self, date_str: str) -> "JQLBuilder":
        """Date clause: ``resolved >= "YYYY-MM-DD"`` (accepts natural language).

        Date functions (``now()``/``startOfYear()``/...) are emitted unquoted —
        inside quotes the server reads them as a plain string.
        """
        self._conditions.append(build_date_condition("resolved", date_str, sanitize_jql_value))
        return self

    def text_contains(self, text: str) -> "JQLBuilder":
        """Text search clause: ``text ~ "value"`` (all text fields)."""
        safe = sanitize_jql_value(text)
        self._conditions.append(f'text ~ "{safe}"')
        return self

    def raw(self, fragment: str) -> "JQLBuilder":
        """Raw fragment clause as-is.

        Use for filters that don't have a dedicated method
        (e.g. ``"sprint in openSprints()"``).
        """
        self._conditions.append(fragment)
        return self

    def order_by(self, field: str, direction: str = "DESC") -> "JQLBuilder":
        """Ordering clause: ``ORDER BY field DESC``."""
        validate_order_by(field)
        safe_dir = direction.strip().upper()
        if safe_dir not in ("ASC", "DESC"):
            safe_dir = "DESC"
        self._order = f"ORDER BY {field.strip()} {safe_dir}"
        return self

    def order_by_raw(self, clause: str) -> "JQLBuilder":
        """Ordering clause: raw ``ORDER BY`` clause (e.g. ``"priority DESC, updated DESC"``).

        A leading ``ORDER BY`` in *clause* is dropped so the result is not
        doubled; the clause is validated like :meth:`order_by`.
        """
        clause = clause.strip()
        if re.match(r"^order\s+by\s", clause, re.IGNORECASE):
            clause = re.sub(r"^order\s+by\s", "", clause, count=1, flags=re.IGNORECASE)
        validate_order_by(clause)
        self._order = f"ORDER BY {clause}"
        return self

    def build(self) -> str:
        """Assemble and return the final JQL string.

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
