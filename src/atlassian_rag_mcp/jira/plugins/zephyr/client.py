"""Zephyr Scale client for Jira.

Communicates via the /rest/tests/1.0/ REST API.
Uses httpx.AsyncClient for async HTTP.

Quirks of this server version (verified):
- the fields list must start with 'id' (otherwise HTTP 500);
- labels/customFieldValues are omitted from the response when empty; steps
  are returned under testScript.stepByStepScript.steps (a nested sub-field
  spec like testScript(steps(...)) returns empty steps);
- the run endpoints only return id/key/name/status — date and statistics
  fields are silently dropped;
- the search endpoints sort by name ascending by default; an explicit
  ORDER BY <entity>.name works (Russian-locale collation, not pure Unicode);
  large pages are accepted (verified up to 500 items via maxResults);
- the execution-history endpoint returns either {"data": [...],
  "totalCount": N} or a plain list, depending on server version/config.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
from atlassian import Jira

from atlassian_rag_mcp.exceptions import ZephyrError
from atlassian_rag_mcp.jira.plugins.zephyr.tql_builder import TqlSearchParams, build_tql_query
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.utils.http_retry import httpx_transport_retry

_API_PREFIX = "/rest/tests/1.0"


def _fields_with_id(fields: str | list[str]) -> str:
    """This Zephyr version returns HTTP 500 when the fields list does not start with 'id'."""
    if isinstance(fields, (list, tuple)):
        fields = ",".join(f for f in fields if isinstance(f, str)) if fields else ""
    if not fields:
        return fields
    first = fields.split(",", 1)[0].strip()
    return fields if first == "id" else f"id,{fields}"


_DEFAULT_TC_FIELDS = (
    "id,key,projectId,archived,name,objective,majorVersion,latestVersion,"
    "precondition,folder(id,fullName),status,priority,estimatedTime,averageTime,"
    "componentId,owner,labels,customFieldValues"
)

_DEFAULT_TC_FULL = _DEFAULT_TC_FIELDS + ",testScript"

_DEFAULT_TC_SEARCH_FIELDS = (
    "id,key,projectId,name,averageTime,estimatedTime,labels,folderId,componentId,"
    "status(id,name,i18nKey,color),priority(id,name,i18nKey,color),"
    "lastTestResultStatus(name,i18nKey,color),majorVersion,"
    "createdOn,createdBy,updatedOn,updatedBy,customFieldValues,owner,folderId"
)

_DEFAULT_RUN_FIELDS = "id,key,name,status(id,name,i18nKey,color)"
_DEFAULT_RUN_SEARCH_FIELDS = "id,key,name,status(id,name,i18nKey,color)"

_DEFAULT_RUN_ITEMS_FIELDS = "id,index,issueCount,$lastTestResult"

_DEFAULT_RUN_RESULTS_FIELDS = (
    "id,key,testResultStatus(name,i18nKey,color),automated,estimatedTime,"
    "actualEndDate,comment,assignedTo,executedBy,"
    "testScriptResults(id,index,description,expectedResult,testData,attachments),"
    "attachments"
)

_DEFAULT_HISTORY_FIELDS = (
    "id,testResultStatus(name,i18nKey,color),key,assignedTo,executionDate,"
    "estimatedTime,executionTime,automated,testRun,testCase,issueLinks"
)

_TRACE_LINK_FIELDS = {
    "issue": "id,issueId,type(name,i18nKey,systemKey),testCaseId,testRunId,testPlanId",
    "weblink": "id,url,urlDescription,type(name,i18nKey,systemKey),testCaseId,testRunId,testPlanId",
    "confluencepage": "id,confluencePageId,title,type(name,i18nKey,systemKey),testCaseId,testRunId,testPlanId",
}

_PLAN_TQL_FIELDS = "testPlan.projectId,testPlan.folderTreeId,testPlan.key,testPlan.name"


def _response_detail(resp: httpx.Response) -> str:
    """Extract a human-readable message from a Zephyr API error response body."""
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "").strip()[:300]
    if isinstance(body, dict):
        messages = body.get("messages")
        if isinstance(messages, list) and messages:
            return "; ".join(str(m) for m in messages)[:300]
        for key in ("error", "message"):
            if body.get(key):
                return str(body[key])[:300]
    return str(body)[:300]


def _status_error(resp: httpx.Response, path: str) -> ZephyrError:
    """Build a ZephyrError for an HTTP error response on the given endpoint."""
    return ZephyrError(
        f"Zephyr API error: GET {path} (HTTP {resp.status_code}) — {_response_detail(resp)}"
    )


def _cause_detail(e: Exception) -> str:
    """Most specific failure text from the exception chain (e.g. the OS-level
    cause of a connection error); falls back to the exception type name."""
    texts: list[str] = []
    current: BaseException | None = e
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if str(current):
            texts.append(str(current))
        current = current.__cause__ or current.__context__
    for text in reversed(texts):
        return text
    return type(e).__name__


def _zephyr_error(e: Exception, path: str) -> ZephyrError:
    """Build a ZephyrError carrying the requested endpoint and the failure cause."""
    if isinstance(e, httpx.HTTPStatusError):
        return _status_error(e.response, path)
    return ZephyrError(f"Zephyr API error: GET {path} — {_cause_detail(e)}")


class ZephyrClient:
    """Zephyr Scale plugin — /rest/tests/1.0/ API."""

    def __init__(
        self,
        jira: Jira,
        base_url: str = "",
        username: str = "",
        password: str = "",
    ) -> None:
        self.jira = jira
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            auth=(username, password),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=httpx.Timeout(30.0),
        )
        self._checked = False
        self._available: bool | None = None

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def _check_availability(self) -> None:
        """Check the Zephyr API once (lazily).

        A transient network failure is not latched: the checked flag is
        reset so the next call re-checks, and the failure is surfaced to the
        caller. HTTP 404/405 means the plugin is not installed — soft path:
        methods return empty results.
        """
        if self._checked:
            return
        self._checked = True
        path = f"{_API_PREFIX}/project"
        try:
            resp = await self._get(f"{self._base_url}{path}")
        except Exception as e:
            self._checked = False
            raise _zephyr_error(e, path) from e
        if resp.status_code == 200:
            self._available = True
            logger.info(f"Zephyr Scale available at {_API_PREFIX}")
            return
        if resp.status_code in (404, 405):
            self._available = False
            logger.debug(f"Zephyr Scale not installed (HTTP {resp.status_code})")
            return
        raise _status_error(resp, path)

    async def is_available(self) -> bool:
        """Check Zephyr Scale API availability."""
        await self._check_availability()
        return bool(self._available)

    async def validate(self) -> bool:
        """Validate Zephyr Scale connectivity and return bool."""
        ok = await self.is_available()
        if ok:
            logger.info("Zephyr Scale verified")
        return ok

    @httpx_transport_retry()
    async def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        """GET with a single retry on transient transport errors."""
        return await self._client.get(url, **kwargs)

    async def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        resp = await self._get(f"{self._base_url}{url}", params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_projects(self) -> list[dict[str, Any]]:
        """List all Zephyr test projects.

        Returns project id, key, and name. Use the id for projectId IN (...) queries
        and the key for project_key parameters in other methods.

        Example:
            >>> plugin.get_projects()
            [{"id": 12345, "key": "PROJ", "name": "Project name"}, ...]
        """
        if not await self.is_available():
            return []
        try:
            return await self._get_json(f"{_API_PREFIX}/project")
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/project") from e

    async def get_test_case(
        self, key: str, fields: str | None = None
    ) -> dict[str, Any] | None:
        """Get a single test case with full details.

        By default returns: id, key, name, objective, precondition, status, priority,
        folder, owner, estimated/average time, component, and testScript with steps
        (testScript.stepByStepScript.steps: index, description, expectedResult, testData).
        labels and customFieldValues are omitted from the response when empty.

        Use after search_test_cases to drill into a specific test case.

        Example:
            >>> plugin.get_test_case("PROJ-123")
            {key, name, status, precondition, testScript: {stepByStepScript: {steps: [...]}}}
        """
        if not await self.is_available():
            return None
        f = _fields_with_id(fields) if fields else _DEFAULT_TC_FULL
        path = f"{_API_PREFIX}/testcase/{key}"
        try:
            resp = await self._get(f"{self._base_url}{path}", params={"fields": f})
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise _zephyr_error(e, path) from e

    async def search_test_cases(
        self,
        project_key: str | None = None,
        project_id: int | None = None,
        folder_id: int | None = None,
        tql: str | None = None,
        fields: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Search test cases using Zephyr TQL query language.

        Returns {results: [...], total, startAt, maxResults} (total is the
        FULL match count). The ``tql`` contract (verbatim raw syntax vs
        plain-text wrap, validation, auto-build from project/folder): see
        ``TQL_HELP`` in tql_builder.py — single source, live-verified.

        Examples:
            >>> plugin.search_test_cases(project_key="PROJ", folder_id=123)
            >>> plugin.search_test_cases(fields="id", limit=1)
            >>> plugin.search_test_cases(tql='testCase.projectId IN (12345) '
            ...     'AND testCase.key IN ("PROJ-123","PROJ-1234")')
        """
        if not await self.is_available():
            return {"results": [], "total": 0, "startAt": offset, "maxResults": limit}

        if tql is None:
            pid = project_id
            if pid is None and project_key is not None:
                pid = await self._resolve_project_id(project_key)
            query = build_tql_query(
                TqlSearchParams(entity="testCase", project_id=pid, folder_id=folder_id)
            )
        else:
            query = build_tql_query(TqlSearchParams(entity="testCase", tql=tql))

        params: dict[str, Any] = {
            "fields": _fields_with_id(fields) if fields else _DEFAULT_TC_SEARCH_FIELDS,
            "query": query or "",
            "startAt": offset,
            "maxResults": limit,
        }
        logger.info(f"TQL -> Zephyr: {params['query']}")
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testcase/search",
                params=params,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testcase/search") from e

    async def get_folders(
        self, project_key: str, folder_type: str = "testcase"
    ) -> list[dict[str, Any]]:
        """Get flat folder tree for a project and folder type.

        Returns list of {id, name, parentId, projectId, folderType}.
        Use folder id from the result as folder_id in the matching search method
        (search_test_cases / search_test_runs / search_test_plans — maps to
        testCase.folderTreeId / testRun.folderTreeId / testPlan.folderTreeId IN).

        Args:
            project_key: Jira project key (e.g., "PROJ").
            folder_type: "testcase" (default), "testrun" or "testplan".

        Example:
            >>> plugin.get_folders("PROJ")
            [{"id": 123, "name": "Regression", "parentId": 456, ...}, ...]
            >>>
            >>> folders = plugin.get_folders("PROJ", folder_type="testrun")
            >>> plugin.search_test_runs(project_key="PROJ", folder_id=folders[0]["id"])
        """
        if not await self.is_available():
            return []
        if folder_type not in ("testcase", "testrun", "testplan"):
            logger.error(f"Zephyr get_folders: invalid folder_type {folder_type!r}")
            return []
        proj_id = await self._resolve_project_id(project_key)
        if not proj_id:
            return []
        try:
            data = await self._get_json(
                f"{_API_PREFIX}/project/{proj_id}/foldertree/{folder_type}",
            )
            if isinstance(data, dict) and "children" in data:
                return list(self._flatten_tree(data["children"]))
            if isinstance(data, list):
                return [self._normalize_folder(item) for item in data if "id" in item]
            return []
        except Exception as e:
            raise _zephyr_error(
                e, f"{_API_PREFIX}/project/{proj_id}/foldertree/{folder_type}"
            ) from e

    async def _resolve_testcase_id(self, key: str) -> int | None:
        """Resolve the numeric test case ID from a case key (e.g., "PROJ-123").

        Returns None when the case does not exist (404); raises ZephyrError
        on other API failures.
        """
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testcase/{key}",
                params={"fields": "id"},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json().get("id")
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testcase/{key}") from e

    async def _resolve_testrun_id(self, key: str) -> int | None:
        """Resolve the numeric test run ID from a run key (e.g., "PROJ-123").

        Returns None when the run does not exist (404); raises ZephyrError
        on other API failures.
        """
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testrun/{key}",
                params={"fields": "id"},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json().get("id")
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testrun/{key}") from e

    async def get_test_run(
        self, key: str, fields: str | None = None
    ) -> dict[str, Any] | None:
        """Get a single test run: id, key, name, status.

        Note: this Zephyr version's run endpoints only return id/key/name/status —
        dates and pass/fail statistics are not available via the API (the fields are
        silently dropped). Use get_test_run_items / get_test_run_results for the
        individual executions and their statuses.

        The API returns HTTP 500 if called without the fields parameter —
        this method always sends a default field list.

        Example:
            >>> plugin.get_test_run("PROJ-123")
            {id, key, name, status: {id, name, ...}}
            >>>
            >>> plugin.get_test_run_items("PROJ-123")
        """
        if not await self.is_available():
            return None
        f = _fields_with_id(fields) if fields else _DEFAULT_RUN_FIELDS
        path = f"{_API_PREFIX}/testrun/{key}"
        try:
            resp = await self._get(f"{self._base_url}{path}", params={"fields": f})
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise _zephyr_error(e, path) from e

    async def get_test_case_versions(
        self, key: str, fields: str | None = None
    ) -> list[dict[str, Any]] | None:
        """Get all versions of a test case, newest first.

        Returns the full version history: major version number, creator and
        date of each version. Use to see how often a test case changed.
        Returns an empty list if the key does not exist (the API answers 200 []).

        Example:
            >>> plugin.get_test_case_versions("PROJ-123")
            [{"id": 9, "majorVersion": 9, "createdBy": "...", "createdOn": "..."}, ...]
        """
        if not await self.is_available():
            return None
        f = _fields_with_id(fields) if fields else "id,majorVersion,createdBy,createdOn,updatedBy,updatedOn"
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testcase/{key}/allVersions",
                params={"fields": f},
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testcase/{key}/allVersions") from e

    async def get_test_case_history(
        self,
        key: str,
        limit: int = 20,
        offset: int = 0,
        fields: str | None = None,
    ) -> dict[str, Any]:
        """Get the history of a test case.

        Returns {items: [...], total: N}. Depending on the server version,
        items are either:
        - execution entries: status (testResultStatus), assignee, execution
          date/time, the test run the execution belongs to, issue links;
        - change-history entries: id, sourceId, source, type, historyDate,
          userKey, changeHistoryItems [{id, fieldName, originalValue, newValue}]
          (who changed which field and when). Newest changes first.
        In the change-history shape the server ignores limit/offset.

        Example:
            >>> plugin.get_test_case_history("PROJ-123", limit=5)
            {"items": [{...}], "total": 123}
        """
        if not await self.is_available():
            return {"items": [], "total": 0}
        tc_id = await self._resolve_testcase_id(key)
        if not tc_id:
            return {"items": [], "total": 0}
        f = _fields_with_id(fields) if fields else _DEFAULT_HISTORY_FIELDS
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testcase/{tc_id}/history",
                params={"fields": f, "limit": limit, "offset": offset},
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                items = data.get("data", [])
                total = data.get("totalCount", len(items))
            else:
                items = data if isinstance(data, list) else []
                total = len(items)
            return {"items": items, "total": total}
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testcase/{tc_id}/history") from e

    async def get_test_case_trace_links(
        self, key: str, link_type: str = "issue"
    ) -> list[dict[str, Any]] | None:
        """Get traceability links of a test case.

        link_type:
            "issue"          — linked Jira issues (id, issueId, type, testCaseId, ...)
            "weblink"        — external URLs (id, url, urlDescription, ...)
            "confluencepage" — linked Confluence pages (id, confluencePageId, title, ...)

        The confluencePageId values can be passed to get_confluence_pages
        to fetch the page content.

        Example:
            >>> plugin.get_test_case_trace_links("PROJ-123", link_type="issue")
            [{"id": 1, "issueId": 456, "type": {...}, ...}]
        """
        if not await self.is_available():
            return None
        if link_type not in ("issue", "weblink", "confluencepage"):
            logger.error(f"Zephyr get_test_case_trace_links: invalid link_type {link_type!r}")
            return None
        tc_id = await self._resolve_testcase_id(key)
        if not tc_id:
            return []
        f = _TRACE_LINK_FIELDS[link_type]
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testcase/{tc_id}/tracelinks/{link_type}",
                params={"fields": f},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testcase/{tc_id}/tracelinks/{link_type}") from e

    async def get_test_case_comments(self, key: str) -> list[dict[str, Any]] | None:
        """Get comments attached to a test case.

        Each item: {id, body, createdBy, createdOn}.

        Example:
            >>> plugin.get_test_case_comments("PROJ-123")
            [{"id": 1, "body": "Нужно обновить шаги", "createdBy": "JIRAUSER11607", ...}]
        """
        if not await self.is_available():
            return None
        tc_id = await self._resolve_testcase_id(key)
        if not tc_id:
            return []
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testcase/{tc_id}/comments"
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testcase/{tc_id}/comments") from e

    async def get_test_case_attachments(self, key: str) -> list[dict[str, Any]] | None:
        """Get files attached to a test case (current version).

        Each item: {id, name, fileName, fileSize, createdOn, userKey}.
        The id can be passed to download_attachment to get the file content.

        Example:
            >>> plugin.get_test_case_attachments("PROJ-123")
            [{"id": 317908, "name": "file.txt", "fileSize": 9353, ...}]
        """
        if not await self.is_available():
            return None
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testcase/{key}",
                params={"fields": "attachments"},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                return data.get("attachments") or []
            return []
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testcase/{key}") from e

    async def search_test_runs(
        self,
        project_key: str | None = None,
        project_id: int | None = None,
        folder_id: int | None = None,
        tql: str | None = None,
        fields: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Search test runs/cycles.

        Returns paginated results; each item: id, key, name, status.
        Note: this Zephyr version's run endpoints do not return dates or
        pass/fail statistics via the API.

        TQL field contract (supported/unsupported fields, pagination,
        ordering): see TQL_HELP in tql_builder.py (single source,
        live-verified on the instance).

        Use the run key with get_test_run for details, or get_test_run_items
        / get_test_run_results for individual executions.

        Example:
            >>> plugin.search_test_runs(project_key="PROJ", limit=5)
            >>> plugin.search_test_runs(tql='testRun.projectId IN (12345)')
            >>> plugin.search_test_runs(tql='testRun.key IN ("PROJ-123")')
        """
        if not await self.is_available():
            return {"results": [], "total": 0, "startAt": offset, "maxResults": limit}

        if tql is None:
            pid = project_id
            if pid is None and project_key is not None:
                pid = await self._resolve_project_id(project_key)
            query = build_tql_query(
                TqlSearchParams(entity="testRun", project_id=pid, folder_id=folder_id)
            )
        else:
            query = build_tql_query(TqlSearchParams(entity="testRun", tql=tql))

        params: dict[str, Any] = {
            "fields": _fields_with_id(fields) if fields else _DEFAULT_RUN_SEARCH_FIELDS,
            "query": query or "",
            "startAt": offset,
            "maxResults": limit,
        }
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testrun/search",
                params=params,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testrun/search") from e

    async def get_test_run_items(
        self, key: str, fields: str | None = None
    ) -> dict[str, Any]:
        """Get the test cases included in a test run (run items).

        Returns {items: [...], total: N} where each item:
        {id, index, $lastTestResult: {id, assignedTo, testCase, environment, ...}}
        and issueCount (omitted when 0).
        Use item id with get_test_run_results to see individual executions.

        Example:
            >>> plugin.get_test_run_items("PROJ-123")
            {"items": [{id, index, $lastTestResult: {...}}], "total": 21}
        """
        if not await self.is_available():
            return {"items": [], "total": 0}
        run_id = await self._resolve_testrun_id(key)
        if not run_id:
            return {"items": [], "total": 0}
        f = _fields_with_id(fields) if fields else _DEFAULT_RUN_ITEMS_FIELDS
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testrun/{run_id}/testrunitems",
                params={"fields": f},
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                items = data.get("testRunItems") or []
                return {"items": items, "total": data.get("total", len(items))}
            if isinstance(data, list):
                return {"items": data, "total": len(data)}
            return {"items": [], "total": 0}
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testrun/{run_id}/testrunitems") from e

    async def get_test_run_results(
        self,
        key: str,
        item_id: int | None = None,
        fields: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get test execution results of a test run (or of a single run item).

        Returns a list of executions, each with: id, key, testResultStatus,
        automated, estimatedTime, actualEndDate, comment, assignedTo, executedBy,
        per-step results (testScriptResults: index, description, expectedResult,
        testData, step attachments) and execution-level attachments.
        Fields are omitted for executions that have no data (e.g. an execution
        that was never started has no date/assignee/attachments).

        Each attachments item: {id, name, fileSize, createdOn, userKey}.
        Pass an attachment id to download_attachment to fetch the file.

        Without item_id all executions of the run are returned; with item_id
        only executions of that run item (from get_test_run_items).

        Note: this Zephyr version requires the itemId parameter — when it is
        omitted, the run items are resolved first and results are fetched per item.

        Example:
            >>> plugin.get_test_run_results("PROJ-123")
            >>> plugin.get_test_run_results("PROJ-123", item_id=398049)
        """
        if not await self.is_available():
            return []
        run_id = await self._resolve_testrun_id(key)
        if not run_id:
            return []
        f = _fields_with_id(fields) if fields else _DEFAULT_RUN_RESULTS_FIELDS
        try:
            if item_id is None:
                resp_items = await self._get(
                    f"{self._base_url}{_API_PREFIX}/testrun/{run_id}/testrunitems",
                    params={"fields": "id"},
                )
                resp_items.raise_for_status()
                items_data = resp_items.json()
                if isinstance(items_data, dict):
                    item_list = items_data.get("testRunItems") or []
                elif isinstance(items_data, list):
                    item_list = items_data
                else:
                    item_list = []
                out: list[dict[str, Any]] = []
                for it in item_list:
                    resp = await self._get(
                        f"{self._base_url}{_API_PREFIX}/testrun/{run_id}/testresults",
                        params={"fields": f, "itemId": it["id"]},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if isinstance(data, list):
                        out.extend(data)
                return out
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testrun/{run_id}/testresults",
                params={"fields": f, "itemId": item_id},
            )
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testrun/{run_id}/testresults") from e

    async def get_test_result(
        self, result_id: int, fields: str | None = None
    ) -> dict[str, Any] | None:
        """Get a single test execution result by its numeric id.

        The id is available in items of get_test_run_results / get_test_case_history.
        Returns the full execution: id, key, testResultStatus, automated,
        estimatedTime, actualEndDate, comment, assignedTo, executedBy, per-step
        results, attachments. Fields are omitted for executions with no data.

        Example:
            >>> plugin.get_test_result(12345)
            {id, testResultStatus: {...}, testScriptResults: [...], attachments: [...], ...}
        """
        if not await self.is_available():
            return None
        f = _fields_with_id(fields) if fields else _DEFAULT_RUN_RESULTS_FIELDS
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testresult/{result_id}",
                params={"fields": f},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testresult/{result_id}") from e

    async def search_test_plans(
        self,
        project_key: str | None = None,
        project_id: int | None = None,
        folder_id: int | None = None,
        tql: str | None = None,
        fields: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Search test plans using TQL.

        Returns paginated results: {results: [...], total: N, startAt: M, maxResults: K}.
        Each item: id, key, name, projectId, folderId (this Zephyr version does not
        return plan dates or statistics via the API).

        TQL field contract (supported/unsupported fields, pagination,
        ordering): see TQL_HELP in tql_builder.py (single source,
        live-verified on the instance).

        Example:
            >>> plugin.search_test_plans(project_key="PROJ", limit=5)
            >>> plugin.search_test_plans(tql='testPlan.projectId IN (12345) ORDER BY testPlan.name ASC')
        """
        if not await self.is_available():
            return {"results": [], "total": 0, "startAt": offset, "maxResults": limit}

        if tql is None:
            pid = project_id
            if pid is None and project_key is not None:
                pid = await self._resolve_project_id(project_key)
            query = build_tql_query(
                TqlSearchParams(entity="testPlan", project_id=pid, folder_id=folder_id)
            )
        else:
            query = build_tql_query(TqlSearchParams(entity="testPlan", tql=tql))

        params: dict[str, Any] = {
            "fields": _fields_with_id(fields) if fields else "id,key,name,projectId,folderId",
            "query": query or "",
            "startAt": offset,
            "maxResults": limit,
        }
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testplan/search",
                params=params,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testplan/search") from e

    async def get_test_plan(
        self, key: str, fields: str | None = None
    ) -> dict[str, Any] | None:
        """Get a single test plan by key with its details.

        Returns id, key, name, projectId, folderId. This Zephyr version does not
        return plan dates or progress statistics via the API.

        Example:
            >>> plugin.get_test_plan("PROJ-123")
            {id, key, name, projectId, folderId}
        """
        if not await self.is_available():
            return None
        f = _fields_with_id(fields) if fields else "id,key,name,projectId,folderId"
        try:
            resp = await self._get(
                f"{self._base_url}{_API_PREFIX}/testplan/{key}",
                params={"fields": f},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/testplan/{key}") from e

    async def get_test_case_statuses(self, project_key: str) -> list[dict[str, Any]]:
        """List test case statuses configured for a project.

        Each item: {id, name, i18nKey, color}. The id is used in the status
        field of test cases (see get_test_case).

        Example:
            >>> plugin.get_test_case_statuses("PROJ")
            [{"id": 1, "name": "Not started", ...}, {"id": 2, "name": "Done", ...}]
        """
        if not await self.is_available():
            return []
        proj_id = await self._resolve_project_id(project_key)
        if not proj_id:
            return []
        try:
            data = await self._get_json(
                f"{_API_PREFIX}/project/{proj_id}/testcasestatus"
            )
            return data if isinstance(data, list) else []
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/project/{proj_id}/testcasestatus") from e

    async def get_test_case_priorities(self, project_key: str) -> list[dict[str, Any]]:
        """List test case priorities configured for a project.

        Each item: {id, name, i18nKey, color}. The id is used in the priority
        field of test cases (see get_test_case).

        Example:
            >>> plugin.get_test_case_priorities("PROJ")
            [{"id": 1, "name": "Critical", ...}, ...]
        """
        if not await self.is_available():
            return []
        proj_id = await self._resolve_project_id(project_key)
        if not proj_id:
            return []
        try:
            data = await self._get_json(
                f"{_API_PREFIX}/project/{proj_id}/testcasepriority",
                params={"fields": "id,name,i18nKey,color"},
            )
            return data if isinstance(data, list) else []
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/project/{proj_id}/testcasepriority") from e

    async def get_test_result_statuses(self, project_key: str) -> list[dict[str, Any]]:
        """List test result statuses (execution outcomes) for a project.

        Each item: {id, name, i18nKey, color} — e.g. "Passed", "Failed", "Untested".
        These statuses appear in execution results (get_test_run_results).

        Example:
            >>> plugin.get_test_result_statuses("PROJ")
            [{"id": 1, "name": "Untested", ...}, ...]
        """
        if not await self.is_available():
            return []
        proj_id = await self._resolve_project_id(project_key)
        if not proj_id:
            return []
        try:
            data = await self._get_json(
                f"{_API_PREFIX}/project/{proj_id}/testresultstatus",
                params={"fields": "id,name,i18nKey,color"},
            )
            return data if isinstance(data, list) else []
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/project/{proj_id}/testresultstatus") from e

    async def get_custom_fields(
        self, project_key: str, field_type: str = "testcase"
    ) -> list[dict[str, Any]]:
        """List custom fields configured for a project and entity type.

        field_type: "testcase" (default), "teststep", "folder", "testrun" or "testplan".
        Each item: {id, name, type, ...}. Custom field values of a test case are
        returned in its customFieldValues field (see get_test_case).

        Example:
            >>> plugin.get_custom_fields("PROJ")
            [{"id": 10011, "name": "Covered module", ...}, ...]
        """
        if not await self.is_available():
            return []
        if field_type not in ("testcase", "teststep", "folder", "testrun", "testplan"):
            logger.error(f"Zephyr get_custom_fields: invalid field_type {field_type!r}")
            return []
        proj_id = await self._resolve_project_id(project_key)
        if not proj_id:
            return []
        try:
            data = await self._get_json(
                f"{_API_PREFIX}/project/{proj_id}/customfields/{field_type}"
            )
            return data if isinstance(data, list) else []
        except Exception as e:
            raise _zephyr_error(
                e, f"{_API_PREFIX}/project/{proj_id}/customfields/{field_type}"
            ) from e

    async def get_project_permissions(self, project_key: str) -> dict[str, Any] | None:
        """Get permission settings of the test project.

        Returns which actions (create/edit/delete test cases, runs, plans, ...)
        are allowed for which users/groups.

        Example:
            >>> plugin.get_project_permissions("PROJ")
            {testcasePermission: {...}, testRunPermission: {...}, ...}
        """
        if not await self.is_available():
            return None
        proj_id = await self._resolve_project_id(project_key)
        if not proj_id:
            return None
        try:
            return await self._get_json(
                f"{_API_PREFIX}/project/{proj_id}/userpermissions"
            )
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/project/{proj_id}/userpermissions") from e

    async def get_datasets(self, project_key: str) -> list[dict[str, Any]]:
        """List test datasets of a project (test data used via {{variable}} in steps).

        Each item: {id, name, key, ...}. Datasets are referenced in test case
        step testData as {{dataset_key}}.

        Example:
            >>> plugin.get_datasets("PROJ")
            [{"id": 1, "name": "Базовые данные", ...}]
        """
        if not await self.is_available():
            return []
        proj_id = await self._resolve_project_id(project_key)
        if not proj_id:
            return []
        try:
            data = await self._get_json(
                f"{_API_PREFIX}/dataset", params={"projectId": proj_id}
            )
            if isinstance(data, dict):
                return data.get("data") or []
            return data if isinstance(data, list) else []
        except Exception as e:
            raise _zephyr_error(e, f"{_API_PREFIX}/dataset") from e

    async def get_confluence_pages(self, page_ids: list[int]) -> list[dict[str, Any]]:
        """Get metadata of Confluence pages referenced by test case trace links.

        Pages that no longer exist or are not accessible (HTTP 401/403/404)
        are skipped.

        Args:
            page_ids: list of Confluence page ids, obtained from
                get_test_case_trace_links(link_type="confluencepage") —
                field confluencePageId of each link.

        Returns for each id: {id, title, url, appId, iconUrl} (page metadata —
        this Zephyr version does not return the page body). Pages that no longer
        exist are returned with a notFound=true flag.

        Note: comma-separated id lists are not supported by this version —
        pages are fetched one request per id.

        Example:
            >>> plugin.get_confluence_pages([12345, 67890])
            [{id: 12345, title: "ЭТАП 1. ...", url: "https://wiki.../pages/12345"}, ...]
        """
        if not await self.is_available() or not page_ids:
            return []
        out: list[dict[str, Any]] = []
        for pid in page_ids:
            try:
                data = await self._get_json(
                    f"{_API_PREFIX}/confluence",
                    params={"confluencePageIds": pid},
                )
            except Exception as e:
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (401, 403, 404):
                    continue
                raise _zephyr_error(e, f"{_API_PREFIX}/confluence") from e
            if isinstance(data, dict):
                out.extend(data.get("data") or [])
            elif isinstance(data, list):
                out.extend(data)
        return out

    async def download_attachment(
        self,
        attachment_id: int,
        max_bytes: int = 10 * 1024 * 1024,
    ) -> dict[str, Any] | None:
        """Download a file attached to a test execution or test case.

        The attachment id is taken from the attachments field returned by
        get_test_run_results, get_test_case_history, or get_test_case_attachments.
        The file name should be taken from that metadata (field name), not from
        the HTTP header (Cyrillic names are mangled in the header).

        Returns {id, name, content_type, size, content_base64, truncated}.
        Files larger than max_bytes are truncated (truncated=True).
        Returns None if the attachment does not exist (404) or is not accessible.

        Note: the request sends Accept: */* — with the client's default
        Accept: application/json this endpoint returns HTTP 500.

        Example:
            >>> plugin.download_attachment(317908)
            {id: 317908, name: "file.txt", content_type: "application/octet-stream",
             size: 9353, content_base64: "...", truncated: False}
        """
        if not await self.is_available():
            return None
        path = f"{_API_PREFIX}/attachment/{attachment_id}"
        try:
            resp = await self._get(
                f"{self._base_url}{path}",
                headers={"Accept": "*/*"},
            )
            if resp.status_code == 404:
                return None
            if resp.status_code in (401, 403):
                logger.error(
                    f"Zephyr download_attachment {attachment_id}: "
                    f"authentication required (HTTP {resp.status_code})"
                )
                return None
            resp.raise_for_status()
            content = resp.content
            truncated = len(content) > max_bytes
            if truncated:
                content = content[:max_bytes]
            name = ""
            cd = resp.headers.get("content-disposition", "")
            if "filename=" in cd:
                name = cd.split("filename=", 1)[1].strip().strip('"')
            return {
                "id": attachment_id,
                "name": name,
                "content_type": resp.headers.get("content-type", ""),
                "size": len(resp.content),
                "content_base64": base64.b64encode(content).decode("ascii"),
                "truncated": truncated,
            }
        except Exception as e:
            raise _zephyr_error(e, path) from e

    async def _resolve_project_id(self, project_key: str) -> str | None:
        """Resolve a project key to its Zephyr project id.

        Returns None when the key is not among the Zephyr projects; raises
        ZephyrError when the Zephyr API itself fails.
        """
        projects = await self.get_projects()
        for p in projects:
            if p.get("key") == project_key:
                return p.get("id")
        return None

    def _flatten_tree(
        self,
        nodes: list[dict[str, Any]],
        parent_id: int | None = None,
        depth: int = 0,
    ) -> list[dict[str, Any]]:
        """Recursively flatten a nested folder tree into a flat list of folders."""
        result: list[dict[str, Any]] = []
        for node in nodes:
            if "id" not in node or "name" not in node:
                continue
            current_parent = node.get("parentId", parent_id)
            result.append(self._normalize_folder(node, parent_id=current_parent, depth=depth))
            children = node.get("children")
            if isinstance(children, list):
                result.extend(self._flatten_tree(children, parent_id=node["id"], depth=depth + 1))
        return result

    def _normalize_folder(
        self,
        item: dict[str, Any],
        parent_id: int | None = None,
        depth: int = 0,
    ) -> dict[str, Any]:
        """Map a raw folder dict to the normalized folder shape."""
        return {
            "id": item["id"],
            "name": item.get("name", ""),
            "parentId": item.get("parentId", parent_id),
            "projectId": item.get("projectId"),
            "folderType": item.get("folderType", "TEST_CASE"),
            "depth": depth,
        }
