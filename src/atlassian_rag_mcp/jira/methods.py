"""Jira methods registry — Registry-style meta method dispatcher (mirrors Confluence search algorithms).

Each method is a class with:
  - descriptor(): MethodDescriptor with id, description, parameter schema
  - execute(params, client): synchronous implementation (run in a worker thread)

Registration via the @register decorator. Public API:
  - get_methods()       -> list[MethodDescriptor]
  - execute_method()    -> {"method", "data"}; raises on backend failure
                           (the MCP tool layer converts it to an error result)

``_QUERY_PARAMS`` holds the structured query filters (issuetype..resolved)
shared verbatim by the search and count_issues methods; ``DICT_PARAMS_DESC``
(registry.py) is the shared dict-params wording for the meta-tools.
"""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.config.settings import settings
from atlassian_rag_mcp.jira.constants import EXPAND_FIELDS, ISSUE_FIELDS, SEARCH_FIELDS
from atlassian_rag_mcp.jira.jql_builder import (
    JQL_SCHEMA_HELP,
    JqlSearchParams,
    build_jql_query,
)
from atlassian_rag_mcp.jira.models import (
    IssueResult,
    JiraMethodResult,
    ListResult,
    SearchResult,
    TransitionResult,
    WorklogResult,
)
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.registry import (
    MethodDescriptor,
    MethodParameter,
    Registry,
    missing_required,
    projects_example,
    substitute_placeholders,
)
from atlassian_rag_mcp.utils.query import as_list

registry = Registry("jira")
register = registry.register

_NON_JQL_PARAMS = ("fields", "limit", "start_at")

_QUERY_PARAMS = [
    MethodParameter(
        name="issuetype",
        type="string",
        description=(
            "Issue type name or numeric ID (names auto-resolved to numeric ID). "
            "Available: {issuetypes}"
        ),
    ),
    MethodParameter(
        name="status",
        type="array",
        description=(
            "Exact status name(s) or numeric ID(s) (names auto-resolved to numeric ID). "
            "Multiple values allowed. Available: {statuses}"
        ),
    ),
    MethodParameter(
        name="priority",
        type="array",
        description=(
            "Priority name(s) or numeric ID(s) (names auto-resolved to numeric ID). "
            "Multiple values allowed. Available: {priorities}"
        ),
    ),
    MethodParameter(
        name="assignee",
        type="string",
        description="Assignee (username, display name, or 'currentUser'/'me'). 'EMPTY' matches unassigned.",
    ),
    MethodParameter(
        name="labels",
        type="array",
        description="Labels to filter by (matches issues having ANY of them)",
    ),
    MethodParameter(
        name="order_by",
        type="string",
        description=(
            "ORDER BY clause, e.g. 'created ASC' or 'priority DESC, updated DESC' "
            "(default: priority DESC, updated DESC)"
        ),
    ),
    MethodParameter(
        name="created",
        type="string",
        description=(
            "created >= date (natural language supported, see above). "
            "Date functions (now(), startOfYear(), ...) are passed through unquoted. "
            "-7d = 7 days ago (the last 7 days, relative to now; also -2w, -1m, -1y)."
        ),
    ),
    MethodParameter(name="updated", type="string",
                    description="updated >= date (natural language supported, see above). "
                                "Date functions (now(), startOfYear(), ...) are passed through "
                                "unquoted. "
                                "-7d = 7 days ago (the last 7 days, relative to now; "
                                "also -2w, -1m, -1y)."),
    MethodParameter(name="duedate", type="string",
                    description="duedate >= date (Jira field 'duedate'; natural language). "
                                "Date functions (now(), startOfYear(), ...) are passed through "
                                "unquoted. "
                                "-7d = 7 days ago (the last 7 days, relative to now; "
                                "also -2w, -1m, -1y)."),
    MethodParameter(name="resolved", type="string",
                    description="resolved >= date (natural language supported, see above). "
                                "Date functions (now(), startOfYear(), ...) are passed through "
                                "unquoted. "
                                "-7d = 7 days ago (the last 7 days, relative to now; "
                                "also -2w, -1m, -1y)."),
]


def _query_params() -> list[MethodParameter]:
    """Fresh copies of the shared query-filter definitions.

    get_methods() substitutes placeholders in place on every call, so the
    descriptors must not alias the shared _QUERY_PARAMS objects (otherwise a
    substituted description would leak into later calls / other methods).
    """
    return [p.model_copy() for p in _QUERY_PARAMS]


_VALUE_MAPS: dict = {}

_FALLBACK_VALUE_MAPS = {
    "{issuetypes}": "(see get_issue_types)",
    "{statuses}": "(see get_project_statuses)",
    "{priorities}": "(see get_priorities)",
    "{resolutions}": "(see get_resolutions)",
}


def reset_value_maps() -> None:
    """Clear the process-wide Jira value-map cache (for tests)."""
    _VALUE_MAPS.clear()


def fetch_jira_value_maps(client: Any, project_keys: list[str] | None = None) -> dict:
    """Fetch issue types/statuses per project plus global priorities/resolutions.

    Cached process-wide (fetched once per process); each fetch is isolated —
    a failure is logged and skipped, never raised. Returns
    {"projects": {key: {"types": [(name, id), ...], "statuses": [...]}, ...},
     "priorities": [(name, id), ...], "resolutions": [(name, id), ...]}.
    """
    if _VALUE_MAPS:
        return _VALUE_MAPS
    project_keys = project_keys or settings.jira_project_keys
    data: dict = {"projects": {}, "priorities": [], "resolutions": []}

    for key in project_keys:
        try:
            types = client.issues.get_issue_types(key)
            types = [(t.get("name", ""), t.get("id", "")) for t in (types.items or [])]
        except Exception as e:
            logger.warning(f"Could not fetch Jira issue types [{key}]: {e}")
            types = []
        try:
            statuses = client.issues.get_project_statuses(key)
            statuses = [(s.get("name", ""), s.get("id", "")) for s in (statuses.items or [])]
        except Exception as e:
            logger.warning(f"Could not fetch Jira statuses [{key}]: {e}")
            statuses = []
        if types or statuses:
            data["projects"][key] = {"types": types, "statuses": statuses}

    try:
        priorities = client.issues.get_priorities()
        data["priorities"] = [(p.get("name", ""), p.get("id", "")) for p in (priorities.items or [])]
    except Exception as e:
        logger.warning(f"Could not fetch Jira priorities: {e}")
    try:
        resolutions = client.issues.get_resolutions()
        data["resolutions"] = [(r.get("name", ""), r.get("id", "")) for r in (resolutions.items or [])]
    except Exception as e:
        logger.warning(f"Could not fetch Jira resolutions: {e}")

    if data["projects"] or data["priorities"] or data["resolutions"]:
        _VALUE_MAPS.clear()
        _VALUE_MAPS.update(data)
    return data


def render_value_maps(data: dict) -> dict[str, str]:
    """Render fetched value maps into placeholder strings for get_methods().

    Empty entries yield the fallback "(see <method>)" values.
    """
    type_lines: list[str] = []
    status_lines: list[str] = []
    for key, values in (data.get("projects") or {}).items():
        types = values.get("types") or []
        if types:
            type_lines.append(f"{key}: " + ", ".join(f"{name} ({pid})" for name, pid in types))
        statuses = values.get("statuses") or []
        if statuses:
            status_lines.append(f"{key}: " + ", ".join(f"{name} ({pid})" for name, pid in statuses))
    priorities = data.get("priorities") or []
    resolutions = data.get("resolutions") or []
    return {
        "{issuetypes}": "\n".join(type_lines) if type_lines else _FALLBACK_VALUE_MAPS["{issuetypes}"],
        "{statuses}": "\n".join(status_lines) if status_lines else _FALLBACK_VALUE_MAPS["{statuses}"],
        "{priorities}": (", ".join(f"{name} ({pid})" for name, pid in priorities)
                          or _FALLBACK_VALUE_MAPS["{priorities}"]),
        "{resolutions}": (", ".join(f"{name} ({pid})" for name, pid in resolutions)
                          or _FALLBACK_VALUE_MAPS["{resolutions}"]),
    }


def get_methods(client: Any = None) -> list[MethodDescriptor]:
    """Return list of registered MethodDescriptors (project-key examples from env).

    With a Jira client, the {issuetypes}/{statuses}/{priorities}/{resolutions}
    placeholders are filled from the live value maps (with a cache); without a
    client (or on failure) they fall back to "(see <method>)" hints.
    """
    keys = settings.jira_project_keys
    placeholders = {
        "{project_key}": keys[0],
        "{project_keys}": projects_example(),
    }
    if client is not None:
        try:
            placeholders.update(render_value_maps(fetch_jira_value_maps(client)))
        except Exception:
            pass
    for key, value in _FALLBACK_VALUE_MAPS.items():
        placeholders.setdefault(key, value)
    methods = registry.get_methods()
    for method in methods:
        method.description = substitute_placeholders(method.description, placeholders)
        for param in method.parameters:
            param.description = substitute_placeholders(param.description, placeholders)
    return methods


async def execute_method(
    method_name: str,
    params: dict[str, Any],
    client: Any,
) -> dict[str, Any]:
    """Execute a Jira method by ID. Returns {"method", "data", "error"}.

    Backend failures are not caught here: they propagate to the MCP tool
    layer (jira_execute_method), the single error boundary for this surface
    — same policy as Confluence's execute_algorithm.
    """
    params = params or {}
    cls = registry.get(method_name)
    if cls is None:
        return JiraMethodResult(
            method=method_name,
            data=None,
            error=f"Unknown method: {method_name}. Available: {', '.join(registry.ids())}",
        ).model_dump()

    missing = missing_required(cls.descriptor(), params)
    if missing:
        return JiraMethodResult(
            method=method_name,
            data=None,
            error=f"Missing required parameter(s): {', '.join(missing)}",
        ).model_dump()

    data = await registry.invoke(cls, params, client)
    return JiraMethodResult(method=method_name, data=data, error=None).model_dump()


@register
class GetIssue:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_issue",
            description="Get a single Jira issue by key with its fields, rendered description, changelog and comments.",
            parameters=[
                MethodParameter(
                    name="issue_key",
                    type="string",
                    description="Jira issue key (e.g., '{project_key}-123'; project from {project_keys})",
                    required=True,
                ),
                MethodParameter(
                    name="fields",
                    type="string",
                    description=f"Comma-separated fields to return (default: {ISSUE_FIELDS})",
                ),
                MethodParameter(
                    name="expand",
                    type="string",
                    description=f"Fields to expand (default: {EXPAND_FIELDS})",
                ),
                MethodParameter(
                    name="comment_limit",
                    type="integer",
                    description="Max comments to include, 0 for none (default: 10)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: IssueResult = client.issues.get_issue(
            issue_key=params["issue_key"],
            fields=params.get("fields") or ISSUE_FIELDS,
            expand=params.get("expand") or EXPAND_FIELDS,
            comment_limit=int(params.get("comment_limit", 10)),
        )
        return result.model_dump()


@register
class Search:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="search",
            description=(
                "Search Jira issues by JQL, free text, or structured filters.\n\n"
                + JQL_SCHEMA_HELP
            ),
            parameters=[
                MethodParameter(
                    name="jql",
                    type="string",
                    description=(
                        "JQL or free text, e.g. 'assignee = currentUser() AND resolution = EMPTY' "
                        "or 'bug in login'. Empty when only structured filters are used. "
                        "Scoped to the configured projects: {project_keys}; the first one "
                        "by default, unless the query contains its own "
                        "'project =' / 'project IN' clause."
                    ),
                ),
                *_query_params(),
                MethodParameter(
                    name="fields",
                    type="string",
                    description=f"Comma-separated fields to return (default: {SEARCH_FIELDS})",
                ),
                MethodParameter(name="limit", type="integer", description="Max results (default: 25)"),
                MethodParameter(
                    name="start_at",
                    type="integer",
                    description="Starting index, 0-based (default: 0)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        """Search issues by the JQL built from params.

        Some DC instances reject custom status/priority names in JQL but
        accept their numeric IDs — on rejection the search is retried with
        the resolved IDs. Some DC builds also silently return 0 for custom
        (Cyrillic) status/priority/issue-type names while their numeric IDs
        match — a zero result is re-run with the ID-substituted JQL.
        """
        search_params = {k: v for k, v in params.items() if k not in _NON_JQL_PARAMS}
        search_params["project"] = client.project_key
        jql = build_jql_query(JqlSearchParams(**search_params))
        try:
            result: SearchResult = cls._run_search(client, params, jql)
        except Exception as e:
            if not _is_value_rejected(e, params):
                raise
            subs = _resolve_value_ids(client, params)
            if not subs:
                raise
            jql = _substitute_values(jql, subs)
            logger.info(f"search: retrying with resolved IDs, jql={jql}")
            result = cls._run_search(client, params, jql)
        if result.total == 0:
            subs = _resolve_value_ids(client, params)
            if subs:
                jql_retry = _substitute_values(jql, subs)
                if jql_retry != jql:
                    result_retry = cls._run_search(client, params, jql_retry)
                    if result_retry.total > 0:
                        logger.info(f"search: ID retry matched {result_retry.total}, jql={jql_retry}")
                        result = result_retry
        return result.model_dump()

    @classmethod
    def _run_search(cls, client: Any, params: dict, jql: str) -> "SearchResult":
        return client.search_service.search_issues(
            jql=jql,
            fields=params.get("fields") or SEARCH_FIELDS,
            limit=int(params.get("limit", 25)),
            start_at=int(params.get("start_at", 0)),
        )


@register
class GetProjectIssues:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_project_issues",
            description="Get recent issues of a Jira project, ordered by created DESC.",
            parameters=[
                MethodParameter(
                    name="project_key",
                    type="string",
                    description="Jira project key (e.g., {project_keys}; default: the project configured in env)",
                ),
                MethodParameter(
                    name="fields",
                    type="string",
                    description=f"Comma-separated fields to return (default: {SEARCH_FIELDS})",
                ),
                MethodParameter(name="limit", type="integer", description="Max results (default: 25)"),
                MethodParameter(
                    name="start_at",
                    type="integer",
                    description="Starting index, 0-based (default: 0)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        project_key = params.get("project_key") or client.project_key
        if not project_key:
            raise ValueError("project_key is required (no project configured in env)")
        result: SearchResult = client.search_service.get_project_issues(
            project_key=project_key,
            fields=params.get("fields") or SEARCH_FIELDS,
            limit=int(params.get("limit", 25)),
            start_at=int(params.get("start_at", 0)),
        )
        return result.model_dump()


@register
class GetComments:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_comments",
            description="Get comments on a Jira issue (paginated).",
            parameters=[
                MethodParameter(
                    name="issue_key",
                    type="string",
                    description="Jira issue key (e.g., '{project_key}-123'; project from {project_keys})",
                    required=True,
                ),
                MethodParameter(
                    name="limit",
                    type="integer",
                    description="Max comments to return, 0 for all (default: 10)",
                ),
                MethodParameter(
                    name="start_at",
                    type="integer",
                    description="Comments to skip, 0-based (default: 0)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_comments(
            issue_key=params["issue_key"],
            limit=int(params.get("limit", 10)),
            start_at=int(params.get("start_at", 0)),
        )
        return result.model_dump()


@register
class GetChangelog:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_changelog",
            description="Get the changelog (history of field changes) for a Jira issue.",
            parameters=[
                MethodParameter(
                    name="issue_key",
                    type="string",
                    description="Jira issue key (e.g., '{project_key}-123'; project from {project_keys})",
                    required=True,
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: IssueResult = client.issues.get_changelog(issue_key=params["issue_key"])
        return result.model_dump()


@register
class GetTransitions:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_transitions",
            description="Get available status transitions for a Jira issue (IDs and names, with screens).",
            parameters=[
                MethodParameter(
                    name="issue_key",
                    type="string",
                    description="Jira issue key (e.g., '{project_key}-123'; project from {project_keys})",
                    required=True,
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: TransitionResult = client.issues.get_transitions(issue_key=params["issue_key"])
        return result.model_dump()


@register
class GetWorklog:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_worklog",
            description="Get worklog entries (time spent) for a Jira issue.",
            parameters=[
                MethodParameter(
                    name="issue_key",
                    type="string",
                    description="Jira issue key (e.g., '{project_key}-123'; project from {project_keys})",
                    required=True,
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: WorklogResult = client.issues.get_worklog(issue_key=params["issue_key"])
        return result.model_dump()


@register
class GetIssueTypes:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_issue_types",
            description="Get issue types available in a project (name, id, subtask/issuetype config).",
            parameters=[
                MethodParameter(
                    name="project_key",
                    type="string",
                    description="Jira project key (e.g., {project_keys}; default: the project configured in env)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        project_key = params.get("project_key") or client.project_key
        if not project_key:
            raise ValueError("project_key is required (no project configured in env)")
        result: ListResult = client.issues.get_issue_types(project_key)
        return result.model_dump()


@register
class GetPriorities:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_priorities",
            description="Get all Jira priorities with their IDs and names.",
            parameters=[],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_priorities()
        return result.model_dump()


@register
class GetResolutions:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_resolutions",
            description="Get all Jira resolutions with their IDs and names.",
            parameters=[],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_resolutions()
        return result.model_dump()


@register
class GetProjectStatuses:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_project_statuses",
            description="Get unique statuses used in a project workflow.",
            parameters=[
                MethodParameter(
                    name="project_key",
                    type="string",
                    description="Jira project key (e.g., {project_keys}; default: the project configured in env)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        project_key = params.get("project_key") or client.project_key
        if not project_key:
            raise ValueError("project_key is required (no project configured in env)")
        result: ListResult = client.issues.get_project_statuses(project_key)
        return result.model_dump()


@register
class GetProjects:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_projects",
            description="Get all Jira projects accessible to the current user (key, id, name).",
            parameters=[],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_projects()
        return result.model_dump()


@register
class Myself:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="myself",
            description="Get current user info (display name, email, account ID).",
            parameters=[],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        return client.jira.myself()


def _require_project_key(params: dict, client: Any) -> str:
    project_key = params.get("project_key") or client.project_key
    if not project_key:
        raise ValueError("project_key is required (no project configured in env)")
    return project_key


def _resolve_board_id(params: dict, client: Any) -> str:
    """Board ID from params, or resolved via project_key (explicit or env-configured)."""
    if params.get("board_id"):
        return params["board_id"]
    project_key = params.get("project_key") or client.project_key
    if not project_key:
        raise ValueError("board_id is required (no project configured in env)")
    board_id = client.issues.resolve_board_id(project_key)
    if not board_id:
        raise ValueError(f"No boards found for project {project_key}")
    return str(board_id)


def _is_value_rejected(e: Exception, params: dict) -> bool:
    """True when the JQL error names one of our status/priority filter values."""
    text = str(e)
    for key in ("status", "priority"):
        for v in as_list(params.get(key)):
            if v and (f"'{v}'" in text or f'"{v}"' in text):
                return True
    return False


def _resolve_value_ids(client: Any, params: dict) -> dict[str, str]:
    """Map status/priority/issue-type names to their numeric IDs (JQL accepts
    IDs even when custom names are rejected or silently don't match).

    Uses client.project_key — the same scope the search itself is forced to
    (Search/CountIssues overwrite ``project`` with it), so a retry can never
    resolve names against a different project.
    """
    subs: dict[str, str] = {}
    project_key = client.project_key
    status_cache: list | None = None
    for name in as_list(params.get("status")):
        if not name or name in subs:
            continue
        if status_cache is None:
            status_cache = client.jira.get_status_for_project(project_key) or []
        for wf in status_cache:
            for s in wf.get("statuses", []):
                if s.get("name") == name and s.get("id") is not None:
                    subs[name] = str(s["id"])
                    break
    priority_cache: list | None = None
    for name in as_list(params.get("priority")):
        if not name or name in subs:
            continue
        if priority_cache is None:
            priority_cache = client.jira.get_all_priorities() or []
        for p in priority_cache:
            if p.get("name") == name and p.get("id") is not None:
                subs[name] = str(p["id"])
                break
    name = params.get("issuetype")
    if name:
        try:
            types = (client.jira.issue_createmeta_issuetypes(project_key) or {}).get("values", [])
        except Exception:
            types = []
        for t in types:
            if t.get("name") == name and t.get("id") is not None:
                subs[name] = str(t["id"])
                break
    return subs


def _substitute_values(jql: str, subs: dict[str, str]) -> str:
    for name, value in subs.items():
        jql = jql.replace(f'"{name}"', value)
    return jql


@register
class GetIssueTree:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_issue_tree",
            description=(
                "Get the context graph of an issue: its subtasks and linked issues "
                "(including parents/epics). Returns 'tree' (nested structure) and "
                "'edges' (raw parent->child pairs from the Jira API)."
            ),
            parameters=[
                MethodParameter(
                    name="issue_key",
                    type="string",
                    description="Jira issue key (e.g., '{project_key}-123'; project from {project_keys})",
                    required=True,
                ),
                MethodParameter(
                    name="depth",
                    type="integer",
                    description="Max recursion depth (default: unlimited, hard cap 150)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: IssueResult = client.issues.get_issue_tree(
            issue_key=params["issue_key"],
            depth=params.get("depth"),
        )
        return result.model_dump()


@register
class GetIssueLabels:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_issue_labels",
            description="Get labels of a Jira issue.",
            parameters=[
                MethodParameter(
                    name="issue_key",
                    type="string",
                    description="Jira issue key (e.g., '{project_key}-123'; project from {project_keys})",
                    required=True,
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_issue_labels(issue_key=params["issue_key"])
        return result.model_dump()


@register
class GetIssueAttachments:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_issue_attachments",
            description="Get issue attachments (filename, size, content and thumbnail URLs).",
            parameters=[
                MethodParameter(
                    name="issue_key",
                    type="string",
                    description="Jira issue key (e.g., '{project_key}-123'; project from {project_keys})",
                    required=True,
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_issue_attachments(issue_key=params["issue_key"])
        return result.model_dump()


@register
class GetIssueRemoteLinks:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_issue_remote_links",
            description="Get remote links on an issue (links to external systems like Git repos, CRMs).",
            parameters=[
                MethodParameter(
                    name="issue_key",
                    type="string",
                    description="Jira issue key (e.g., '{project_key}-123'; project from {project_keys})",
                    required=True,
                ),
                MethodParameter(name="global_id", type="string", description="Filter by external global ID"),
                MethodParameter(name="internal_id", type="string", description="Filter by internal ID"),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_issue_remote_links(
            issue_key=params["issue_key"],
            global_id=params.get("global_id"),
            internal_id=params.get("internal_id"),
        )
        return result.model_dump()


@register
class GetIssueWatchers:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_issue_watchers",
            description="Get users watching a Jira issue.",
            parameters=[
                MethodParameter(
                    name="issue_key",
                    type="string",
                    description="Jira issue key (e.g., '{project_key}-123'; project from {project_keys})",
                    required=True,
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_issue_watchers(issue_key=params["issue_key"])
        return result.model_dump()


@register
class IssueExists:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="issue_exists",
            description="Check whether a Jira issue exists (returns true/false).",
            parameters=[
                MethodParameter(
                    name="issue_key",
                    type="string",
                    description="Jira issue key (e.g., '{project_key}-123'; project from {project_keys})",
                    required=True,
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        return bool(client.issues.issue_exists(issue_key=params["issue_key"]))


@register
class CountIssues:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="count_issues",
            description=(
                "Count issues matching a JQL query (data fetch is skipped, only the total "
                "is returned). Accepts the same query inputs as search (raw JQL, free text "
                "or structured filters). JQL semantics (auto-detection, retries, dates, "
                "ordering): identical to the search method."
            ),
            parameters=[
                MethodParameter(
                    name="jql",
                    type="string",
                    description=(
                        "JQL or free text, e.g. 'resolution = EMPTY' or 'bug in login'. "
                        "Scoped to the configured projects: {project_keys}; the first one "
                        "by default, unless the query contains its own "
                        "'project =' / 'project IN' clause."
                    ),
                ),
                *_query_params(),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        count_params = {k: v for k, v in params.items() if k not in _NON_JQL_PARAMS}
        count_params["project"] = client.project_key
        jql = build_jql_query(JqlSearchParams(**count_params))
        return int(client.issues.count_issues(jql))


@register
class GetBoards:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_boards",
            description="Get agile boards (id, name, type, project). Use the board id for epic/sprint methods.",
            parameters=[
                MethodParameter(
                    name="project_key",
                    type="string",
                    description="Filter by project key (e.g., {project_keys}; default: the project configured in env)",
                ),
                MethodParameter(name="board_name", type="string", description="Filter by board name"),
                MethodParameter(name="board_type", type="string", description="Board type: 'scrum' or 'kanban'"),
                MethodParameter(name="limit", type="integer", description="Max results (default: 50)"),
                MethodParameter(
                    name="start_at",
                    type="integer",
                    description="Starting index, 0-based (default: 0)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        project_key = params.get("project_key") or client.project_key
        result: ListResult = client.issues.get_boards(
            project_key=project_key,
            board_name=params.get("board_name"),
            board_type=params.get("board_type"),
            limit=int(params.get("limit", 50)),
            start_at=int(params.get("start_at", 0)),
        )
        return result.model_dump()


@register
class GetEpics:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_epics",
            description="Get epics of a board. board_id is resolved automatically from project_key if omitted.",
            parameters=[
                MethodParameter(
                    name="board_id",
                    type="integer",
                    description="Agile board ID (get it from get_boards)",
                ),
                MethodParameter(
                    name="project_key",
                    type="string",
                    description=(
                        "Project key to resolve the board from (e.g., {project_keys}; default: the"
                        " project configured in env). Used when board_id is omitted."
                    ),
                ),
                MethodParameter(name="done", type="boolean", description="Only closed epics (default: false)"),
                MethodParameter(name="limit", type="integer", description="Max results (default: 50)"),
                MethodParameter(
                    name="start_at",
                    type="integer",
                    description="Starting index, 0-based (default: 0)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        board_id = _resolve_board_id(params, client)
        result: ListResult = client.issues.get_epics(
            board_id=board_id,
            done=bool(params.get("done", False)),
            limit=int(params.get("limit", 50)),
            start_at=int(params.get("start_at", 0)),
        )
        return result.model_dump()


@register
class GetIssuesForEpic:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_issues_for_epic",
            description="Get all issues belonging to an epic (board-independent).",
            parameters=[
                MethodParameter(
                    name="epic_id",
                    type="string",
                    description=(
                        "Epic issue key (e.g., '{project_key}-123', get it from get_epics; "
                        "project from {project_keys})"
                    ),
                    required=True,
                ),
                MethodParameter(
                    name="fields",
                    type="string",
                    description=f"Comma-separated fields to return (default: {SEARCH_FIELDS})",
                ),
                MethodParameter(name="limit", type="integer", description="Max results (default: 50, 0 for all)"),
                MethodParameter(
                    name="start_at",
                    type="integer",
                    description="Starting index, 0-based (default: 0)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: SearchResult = client.issues.get_issues_for_epic(
            epic_id=params["epic_id"],
            fields=params.get("fields") or SEARCH_FIELDS,
            limit=int(params.get("limit", 50)),
            start_at=int(params.get("start_at", 0)),
        )
        return result.model_dump()


@register
class GetSprints:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_sprints",
            description=(
                "Get sprints of a board (scrum boards only — kanban boards have no sprints)."
                "board_id is resolved automatically from project_key if omitted (prefers scrum"
                "boards)."
            ),
            parameters=[
                MethodParameter(
                    name="board_id",
                    type="integer",
                    description="Agile board ID (get it from get_boards)",
                ),
                MethodParameter(
                    name="project_key",
                    type="string",
                    description=(
                        "Project key to resolve the board from (e.g., {project_keys}; default: the"
                        " project configured in env). Used when board_id is omitted."
                    ),
                ),
                MethodParameter(
                    name="state",
                    type="string",
                    description="Filter by state: 'active', 'closed' or 'future'",
                ),
                MethodParameter(name="limit", type="integer", description="Max results (default: 50)"),
                MethodParameter(
                    name="start_at",
                    type="integer",
                    description="Starting index, 0-based (default: 0)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        board_id = _resolve_board_id(params, client)
        result: ListResult = client.issues.get_sprints(
            board_id=board_id,
            state=params.get("state"),
            limit=int(params.get("limit", 50)),
            start_at=int(params.get("start_at", 0)),
        )
        return result.model_dump()


@register
class GetSprintIssues:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_sprint_issues",
            description="Get all issues of a sprint.",
            parameters=[
                MethodParameter(
                    name="sprint_id",
                    type="integer",
                    description="Sprint ID (get it from get_sprints)",
                    required=True,
                ),
                MethodParameter(name="limit", type="integer", description="Max results (default: 25)"),
                MethodParameter(
                    name="start_at",
                    type="integer",
                    description="Starting index, 0-based (default: 0)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: SearchResult = client.issues.get_sprint_issues(
            sprint_id=params["sprint_id"],
            start_at=int(params.get("start_at", 0)),
            limit=int(params.get("limit", 25)),
        )
        return result.model_dump()


@register
class GetProject:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_project",
            description="Get details of a single Jira project (key, name, lead, categories, URLs).",
            parameters=[
                MethodParameter(
                    name="project_key",
                    type="string",
                    description="Jira project key (e.g., {project_keys}; default: the project configured in env)",
                ),
                MethodParameter(
                    name="expand",
                    type="string",
                    description="Fields to expand, e.g. 'project-urllinks'",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        project_key = _require_project_key(params, client)
        return client.issues.get_project(project_key, expand=params.get("expand"))


@register
class GetProjectComponents:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_project_components",
            description="Get components of a Jira project (name, id, lead).",
            parameters=[
                MethodParameter(
                    name="project_key",
                    type="string",
                    description="Jira project key (e.g., {project_keys}; default: the project configured in env)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        project_key = _require_project_key(params, client)
        result: ListResult = client.issues.get_project_components(project_key)
        return result.model_dump()


@register
class GetProjectVersions:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_project_versions",
            description="Get versions of a Jira project (name, release date, released/archived flags).",
            parameters=[
                MethodParameter(
                    name="project_key",
                    type="string",
                    description="Jira project key (e.g., {project_keys}; default: the project configured in env)",
                ),
                MethodParameter(name="expand", type="string", description="Fields to expand"),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        project_key = _require_project_key(params, client)
        result: ListResult = client.issues.get_project_versions(
            project_key, expand=params.get("expand"),
        )
        return result.model_dump()


@register
class GetAllStatuses:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_all_statuses",
            description="Get all statuses of the Jira instance (unlike get_project_statuses which is workflow-scoped).",
            parameters=[],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_all_statuses()
        return result.model_dump()


@register
class GetFields:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_fields",
            description="Get all Jira fields (id, key, name, custom, type). Useful for building JQL or issue updates.",
            parameters=[],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_fields()
        return result.model_dump()


@register
class GetLastIssueNumber:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_last_issue_number",
            description="Get the last (highest) issue number of a project.",
            parameters=[
                MethodParameter(
                    name="project_key",
                    type="string",
                    description="Jira project key (e.g., {project_keys}; default: the project configured in env)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        project_key = _require_project_key(params, client)
        return int(client.issues.get_last_issue_number(project_key))


@register
class GetUsers:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_users",
            description="Get all users of the Jira instance (paginated). DC/Server: uses 'username=.' search.",
            parameters=[
                MethodParameter(name="limit", type="integer", description="Max results (default: 50)"),
                MethodParameter(
                    name="start_at",
                    type="integer",
                    description="Starting index, 0-based (default: 0)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_users(
            limit=int(params.get("limit", 50)),
            start_at=int(params.get("start_at", 0)),
        )
        return result.model_dump()


@register
class FindUser:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="find_user",
            description="Find users by query (name/email), username, or account ID.",
            parameters=[
                MethodParameter(
                    name="query",
                    type="string",
                    description=(
                        "Search query (name, email, ...). Cloud: fuzzy search. DC/Server: treated"
                        "as exact username. Provide at least one of query/username/account_id."
                    ),
                ),
                MethodParameter(
                    name="username",
                    type="string",
                    description="Exact username (required on DC/Server)",
                ),
                MethodParameter(name="account_id", type="string", description="Account ID (Cloud only)"),
                MethodParameter(
                    name="include_inactive_users",
                    type="boolean",
                    description="Include deactivated users (default: false)",
                ),
                MethodParameter(name="limit", type="integer", description="Max results (default: 50)"),
                MethodParameter(
                    name="start_at",
                    type="integer",
                    description="Starting index, 0-based (default: 0)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        if not any(params.get(k) for k in ("query", "username", "account_id")):
            raise ValueError("Provide at least one of: query, username, account_id")
        result: ListResult = client.issues.find_user(
            query=params.get("query"),
            username=params.get("username"),
            account_id=params.get("account_id"),
            include_inactive_users=bool(params.get("include_inactive_users", False)),
            limit=int(params.get("limit", 50)),
            start_at=int(params.get("start_at", 0)),
        )
        return result.model_dump()


@register
class GetGroups:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_groups",
            description="Get Jira groups (optionally filtered by name query).",
            parameters=[
                MethodParameter(name="query", type="string", description="Group name search query"),
                MethodParameter(name="exclude", type="string", description="Group name to exclude"),
                MethodParameter(name="limit", type="integer", description="Max results (default: 20)"),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        result: ListResult = client.issues.get_groups(
            query=params.get("query"),
            exclude=params.get("exclude"),
            limit=int(params.get("limit", 20)),
        )
        return result.model_dump()


@register
class GetProjectUsers:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_project_users",
            description="Get users who can be assigned to issues in a project.",
            parameters=[
                MethodParameter(
                    name="project_key",
                    type="string",
                    description="Jira project key (e.g., {project_keys}; default: the project configured in env)",
                ),
                MethodParameter(name="limit", type="integer", description="Max results (default: 50)"),
                MethodParameter(
                    name="start_at",
                    type="integer",
                    description="Starting index, 0-based (default: 0)",
                ),
            ],
        )

    @classmethod
    def execute(cls, params: dict[str, Any], client: Any) -> Any:
        project_key = _require_project_key(params, client)
        result: ListResult = client.issues.get_project_users(
            project_key,
            limit=int(params.get("limit", 50)),
            start_at=int(params.get("start_at", 0)),
        )
        return result.model_dump()
