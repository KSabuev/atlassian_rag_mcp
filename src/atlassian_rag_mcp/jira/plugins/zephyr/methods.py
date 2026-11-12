"""Zephyr Scale methods registry — Registry-style meta method dispatcher (mirrors Jira methods).

Each method is a class with:
  - descriptor(): MethodDescriptor with id, description, parameter schema
  - execute(params, client): async implementation over ZephyrClient

Registration via the @register decorator. Public API:
  - get_methods()       -> list[MethodDescriptor]
  - execute_method()    -> {"method", "data"}; raises on backend failure
                           (the MCP tool layer converts it to an error result)

All methods are read-only. The Zephyr plugin is accessed via client.plugins["zephyr"].
"""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.config.settings import settings
from atlassian_rag_mcp.jira.plugins.zephyr.client import ZephyrClient
from atlassian_rag_mcp.jira.plugins.zephyr.models import ZephyrMethodResult
from atlassian_rag_mcp.jira.plugins.zephyr.tql_builder import (
    TQL_HELP,
    TQL_TESTCASE_EXTRA,
    TQL_TESTPLAN_EXTRA,
    TQL_TESTRUN_EXTRA,
    fields_override_param,
    key_param,
    project_key_param,
    zephyr_page_params,
    zephyr_search_params,
)
from atlassian_rag_mcp.registry import (
    MethodDescriptor,
    MethodParameter,
    Registry,
    missing_required,
    projects_example,
    substitute_placeholders,
)

registry = Registry("zephyr")
register = registry.register


def get_methods() -> list[MethodDescriptor]:
    """Return list of registered MethodDescriptors (project-key examples from env).

    No client argument (unlike the Jira registry): Zephyr descriptors carry no
    live value-map placeholders — every method resolves its own Zephyr plugin
    from the passed client inside execute_method.
    """
    keys = settings.jira_project_keys
    placeholders = {
        "{project_key}": keys[0],
        "{project_keys}": projects_example(),
    }
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
    """Execute a Zephyr method by ID. Returns {"method", "data", "error"}.

    Backend failures are not caught here: they propagate to the MCP tool
    layer (zephyr_execute_method), the single error boundary for this
    surface — same policy as Confluence's execute_algorithm.
    """
    params = params or {}
    cls = registry.get(method_name)
    if cls is None:
        return ZephyrMethodResult(
            method=method_name,
            data=None,
            error=f"Unknown method: {method_name}. Available: {', '.join(registry.ids())}",
        ).model_dump()

    missing = missing_required(cls.descriptor(), params)
    if missing:
        return ZephyrMethodResult(
            method=method_name,
            data=None,
            error=f"Missing required parameter(s): {', '.join(missing)}",
        ).model_dump()

    data = await registry.invoke(cls, params, client)
    return ZephyrMethodResult(method=method_name, data=data, error=None).model_dump()


def _plugin(client: Any) -> ZephyrClient:
    """Get the Zephyr plugin from the Jira client, raising a clear error if absent."""
    plugin = (getattr(client, "plugins", None) or {}).get("zephyr")
    if plugin is None:
        raise ValueError(
            "Zephyr plugin is not enabled — set JIRA_PLUGINS=zephyr in the environment"
        )
    return plugin


def _project_key(params: dict[str, Any], client: Any) -> str:
    """Resolve project_key: explicit parameter wins, then the configured project."""
    pk = params.get("project_key") or getattr(client, "project_key", None)
    if not pk:
        raise ValueError(
            "project_key is required — pass it as a parameter "
            "(or configure JIRA_PROJECT_KEYS in the environment)"
        )
    return pk


def _int(params: dict[str, Any], name: str, default: int) -> int:
    value = params.get(name)
    if value in (None, ""):
        return default
    return int(value)


def _page_params(params: dict[str, Any]) -> tuple[int, int]:
    return _int(params, "limit", 25), _int(params, "offset", 0)



@register
class GetProjects:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_projects",
            description=(
                "List all Zephyr test projects with their numeric id and key. "
                "Use the id in TQL queries (projectId IN (...)) and the key as "
                "the project_key parameter of other methods."
            ),
            parameters=[],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_projects()



@register
class SearchTestCases:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="search_test_cases",
            description=(
                "Search test cases by TQL. Without parameters returns test "
                "cases of the configured project. Each item: id, key, name, "
                "status, statusId, priority, folderId, owner, createdOn, "
                "updatedOn, archived, labels.\n\n"
                + TQL_HELP + " " + TQL_TESTCASE_EXTRA
            ),
            parameters=[
                *zephyr_search_params("Folder id from get_folders (test case folder tree)."),
                MethodParameter(
                    name="fields",
                    type="string",
                    description=(
                        "Comma-separated TQL field list, or an array of field names. "
                        "'id' is prepended automatically if omitted. Defaults to a rich set "
                        "(status, priority, folder, times, labels)."
                    ),
                ),
                *zephyr_page_params(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        limit, offset = _page_params(params)
        return await plugin.search_test_cases(
            project_key=params.get("project_key") or getattr(client, "project_key", None),
            project_id=_int(params, "project_id", 0) or None,
            folder_id=_int(params, "folder_id", 0) or None,
            tql=params.get("tql") or None,
            fields=params.get("fields") or None,
            limit=limit,
            offset=offset,
        )


@register
class GetTestCase:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_case",
            description=(
                "Get a single test case by key with full details: name, objective, "
                "precondition, status, priority, owner, folder, times, and steps "
                "(testScript.stepByStepScript.steps: index, description, "
                "expectedResult, testData). labels and customFieldValues are omitted "
                "from the response when empty. Returns None if the key does not exist."
            ),
            parameters=[
                key_param("case"),
                fields_override_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_case(params["test_case_key"], params.get("fields"))


@register
class GetTestCaseVersions:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_case_versions",
            description=(
                "Get the version history of a test case (newest first): for each "
                "version — major version number, who and when it was created/updated. "
                "Use to see how often a test case changed."
            ),
            parameters=[
                key_param("case"),
                fields_override_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_case_versions(params["test_case_key"], params.get("fields"))


@register
class GetTestCaseHistory:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_case_history",
            description=(
                "Get the history of a test case. Returns {items, total}. Depending "
                "on the server version, items are either execution entries (id, key, "
                "result status, assignee, execution date/times, the test run it "
                "belongs to (id, key), the case version (id, majorVersion), related "
                "issue links; fields omitted for executions without data; total is "
                "the full history size, not the page size) or change-history entries "
                "(id, sourceId, source, type, historyDate, userKey and "
                "changeHistoryItems [{id, fieldName, originalValue, newValue}] — who "
                "changed which field and when; in this shape the server ignores "
                "limit/offset and total is the returned list length)."
            ),
            parameters=[
                key_param("case"),
                MethodParameter(name="limit", type="integer", description="Page size (default 20)."),
                MethodParameter(name="offset", type="integer", description="Page offset (default 0)."),
                fields_override_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_case_history(
            params["test_case_key"],
            limit=_int(params, "limit", 20),
            offset=_int(params, "offset", 0),
            fields=params.get("fields"),
        )


@register
class GetTestCaseTraceLinks:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_case_trace_links",
            description=(
                "Get traceability links of a test case. link_type=\"issue\" — linked "
                "Jira issues (field issueId); link_type=\"weblink\" — external URLs "
                "(fields url, urlDescription); link_type=\"confluencepage\" — linked "
                "Confluence pages (field confluencePageId; pass the ids to "
                "get_confluence_pages to get the page title and URL)."
            ),
            parameters=[
                key_param("case"),
                MethodParameter(name="link_type", type="string",
                                      description="Link type: \"issue\" (default), \"weblink\" or \"confluencepage\"."),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_case_trace_links(
            params["test_case_key"], params.get("link_type") or "issue"
        )


@register
class GetTestCaseComments:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_case_comments",
            description=(
                "Get comments on a test case. Each item: {id, body, createdBy, "
                "createdOn}. Returns [] if the case has no comments."
            ),
            parameters=[
                key_param("case"),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_case_comments(params["test_case_key"])


@register
class GetTestCaseAttachments:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_case_attachments",
            description=(
                "Get files attached to the current version of a test case. Each item: "
                "{id, name, fileSize, createdOn, userKey}. Pass the attachment id to "
                "download_attachment to fetch the file content."
            ),
            parameters=[
                key_param("case"),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_case_attachments(params["test_case_key"])



@register
class GetFolders:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_folders",
            description=(
                "Get the flat folder tree of a project for the given folder type. "
                "Each item: {id, name, parentId, projectId, folderType, depth}. "
                "Use the id as folder_id in the matching search method: "
                "search_test_cases for folder_type=\"testcase\", search_test_runs "
                "for \"testrun\", search_test_plans for \"testplan\"."
            ),
            parameters=[
                project_key_param(),
                MethodParameter(name="folder_type", type="string",
                                      description="Folder tree: \"testcase\" (default), \"testrun\" or \"testplan\"."),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        pk = _project_key(params, client)
        return await plugin.get_folders(pk, params.get("folder_type") or "testcase")



@register
class SearchTestRuns:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="search_test_runs",
            description=(
                "Search test runs (cycles) by TQL. Without parameters returns "
                "runs of the configured project. Each item: id, key, name, "
                "status (this Zephyr version does not return run dates or "
                "statistics via the API).\n\n"
                + TQL_HELP + " " + TQL_TESTRUN_EXTRA
            ),
            parameters=[
                *zephyr_search_params('Run folder id from get_folders(folder_type="testrun").'),
                MethodParameter(
                    name="fields",
                    type="string",
                    description=(
                        "Comma-separated TQL field list. Defaults to id, key, name, status (the"
                        "only fields this version returns)."
                    ),
                ),
                *zephyr_page_params(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        limit, offset = _page_params(params)
        return await plugin.search_test_runs(
            project_key=params.get("project_key") or getattr(client, "project_key", None),
            project_id=_int(params, "project_id", 0) or None,
            folder_id=_int(params, "folder_id", 0) or None,
            tql=params.get("tql") or None,
            fields=params.get("fields") or None,
            limit=limit,
            offset=offset,
        )


@register
class GetTestRun:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_run",
            description=(
                "Get a single test run by key. This Zephyr version returns only "
                "id, key, name, status — run dates and pass/fail statistics are not "
                "available via the API. Use get_test_run_items / get_test_run_results "
                "for the individual executions and their statuses."
            ),
            parameters=[
                key_param("run"),
                fields_override_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_run(params["test_run_key"], params.get("fields"))


@register
class GetTestRunItems:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_run_items",
            description=(
                "Get the test cases included in a test run. Returns {items, total}; "
                "each item: {id, index, $lastTestResult (id, assignedTo, testCase, "
                "environment, ...), issueCount (omitted when 0)}. Pass an item id to "
                "get_test_run_results to see its individual executions."
            ),
            parameters=[
                key_param("run"),
                fields_override_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_run_items(params["test_run_key"], params.get("fields"))


@register
class GetTestRunResults:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_run_results",
            description=(
                "Get test executions of a test run (or of a single run item). Each "
                "execution: id, key, testResultStatus, automated, estimatedTime, "
                "actualEndDate, comment, assignedTo, executedBy, per-step results "
                "(testScriptResults: index, description, expectedResult, testData, "
                "step attachments) and execution-level attachments. Fields are "
                "omitted for executions without data. attachments items have an id — "
                "pass it to download_attachment to fetch the file. When item_id is "
                "omitted, the run items are resolved first and results are fetched "
                "per item (the Zephyr API requires itemId)."
            ),
            parameters=[
                key_param("run"),
                MethodParameter(
                    name="item_id",
                    type="integer",
                    description=(
                        "Run item id from get_test_run_items; when set, only executions of that"
                        "item are returned."
                    ),
                ),
                fields_override_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_run_results(
            params["test_run_key"],
            item_id=_int(params, "item_id", 0) or None,
            fields=params.get("fields"),
        )


@register
class GetTestResult:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_result",
            description=(
                "Get a single test execution result by its numeric id. The id is "
                "available in items of get_test_run_results and get_test_case_history. "
                "Returns the full execution: id, key, testResultStatus, automated, "
                "estimatedTime, actualEndDate, comment, assignedTo, executedBy, "
                "per-step results, attachments (fields omitted when empty). "
                "Returns None if the id does not exist."
            ),
            parameters=[
                MethodParameter(
                    name="test_result_id",
                    type="integer",
                    description="Numeric test result id from get_test_run_results or get_test_case_history.",
                    required=True,
                ),
                fields_override_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_result(int(params["test_result_id"]), params.get("fields"))



@register
class SearchTestPlans:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="search_test_plans",
            description=(
                "Search test plans by TQL. Without parameters returns plans of "
                "the configured project. Each item: id, key, name, projectId, "
                "folderId (this Zephyr version does not return plan dates or "
                "statistics via the API).\n\n"
                + TQL_HELP + " " + TQL_TESTPLAN_EXTRA
            ),
            parameters=[
                *zephyr_search_params('Plan folder id from get_folders(folder_type="testplan").'),
                MethodParameter(
                    name="fields",
                    type="string",
                    description="Comma-separated TQL field list. Defaults to id, key, name, projectId, folderId.",
                ),
                *zephyr_page_params(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        limit, offset = _page_params(params)
        return await plugin.search_test_plans(
            project_key=params.get("project_key") or getattr(client, "project_key", None),
            project_id=_int(params, "project_id", 0) or None,
            folder_id=_int(params, "folder_id", 0) or None,
            tql=params.get("tql") or None,
            fields=params.get("fields") or None,
            limit=limit,
            offset=offset,
        )


@register
class GetTestPlan:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_plan",
            description=(
                "Get a single test plan by key: id, key, name, projectId, folderId "
                "(this Zephyr version does not return plan dates or statistics via "
                "the API). Returns None if the key does not exist."
            ),
            parameters=[
                key_param("plan"),
                fields_override_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_plan(params["test_plan_key"], params.get("fields"))



@register
class GetTestCaseStatuses:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_case_statuses",
            description=(
                "List test case statuses configured for the project: {id, name, "
                "i18nKey, color}. The id is used in the status field of test cases "
                "(see get_test_case)."
            ),
            parameters=[
                project_key_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_case_statuses(_project_key(params, client))


@register
class GetTestCasePriorities:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_case_priorities",
            description=(
                "List test case priorities configured for the project: {id, name, "
                "i18nKey, color}. The id is used in the priority field of test cases "
                "(see get_test_case)."
            ),
            parameters=[
                project_key_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_case_priorities(_project_key(params, client))


@register
class GetTestResultStatuses:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_test_result_statuses",
            description=(
                "List test result statuses (execution outcomes, e.g. Pass, Fail, "
                "Blocked) for the project: {id, name, i18nKey, color}. These "
                "statuses appear in execution results (get_test_run_results)."
            ),
            parameters=[
                project_key_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_test_result_statuses(_project_key(params, client))


@register
class GetCustomFields:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_custom_fields",
            description=(
                "List custom fields configured for the project and entity type "
                "(field_type: \"testcase\" default, \"teststep\", \"folder\", "
                "\"testrun\", \"testplan\"). Each item: {id, name, type, ...}. "
                "Test case custom field values are returned in the customFieldValues "
                "field of get_test_case."
            ),
            parameters=[
                project_key_param(),
                MethodParameter(
                    name="field_type",
                    type="string",
                    description=(
                        "Entity type: \"testcase\" (default), \"teststep\", \"folder\", \"testrun\""
                        "or \"testplan\"."
                    ),
                ),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_custom_fields(
            _project_key(params, client), params.get("field_type") or "testcase"
        )


@register
class GetProjectPermissions:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_project_permissions",
            description=(
                "Get permission settings of the test project: which actions "
                "(create/edit/delete test cases, runs, plans, ...) are allowed for "
                "which users/groups."
            ),
            parameters=[
                project_key_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_project_permissions(_project_key(params, client))


@register
class GetDatasets:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_datasets",
            description=(
                "List test datasets of the project (test data referenced in steps as "
                "{{variable}}). Each item: {id, name, key, ...}."
            ),
            parameters=[
                project_key_param(),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.get_datasets(_project_key(params, client))



@register
class GetConfluencePages:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="get_confluence_pages",
            description=(
                "Get metadata of Confluence pages referenced by test case trace "
                "links. Pass the confluencePageId values collected from "
                "get_test_case_trace_links(link_type=\"confluencepage\"). Returns "
                "for each id: {id, title, url, appId, iconUrl} — page metadata "
                "(this Zephyr version does not return the page body). Pages that no "
                "longer exist are returned with a notFound=true flag."
            ),
            parameters=[
                MethodParameter(
                    name="page_ids",
                    type="array",
                    description=(
                        "List of Confluence page ids (integers) from"
                        "get_test_case_trace_links(link_type=\"confluencepage\")."
                    ),
                    required=True,
                ),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        page_ids = [int(i) for i in params["page_ids"]]
        return await plugin.get_confluence_pages(page_ids)


@register
class DownloadAttachment:
    @classmethod
    def descriptor(cls) -> MethodDescriptor:
        return MethodDescriptor(
            id="download_attachment",
            description=(
                "Download a file attached to a test execution or test case. The "
                "attachment id is taken from the attachments field returned by "
                "get_test_run_results, get_test_case_history or "
                "get_test_case_attachments. Returns {id, name, content_type, size, "
                "content_base64, truncated} — file content as base64; files over 10 MB "
                "are truncated (truncated=True). Returns None if the attachment does "
                "not exist or is inaccessible. Prefer the name from the attachments "
                "metadata over the returned header name (Cyrillic names are mangled "
                "in the header)."
            ),
            parameters=[
                MethodParameter(
                    name="attachment_id",
                    type="integer",
                    description=(
                        "Attachment id from the attachments field of get_test_run_results /"
                        "get_test_case_history / get_test_case_attachments."
                    ),
                    required=True,
                ),
            ],
        )

    @classmethod
    async def execute(cls, params: dict[str, Any], client: Any) -> Any:
        plugin = _plugin(client)
        return await plugin.download_attachment(int(params["attachment_id"]))
