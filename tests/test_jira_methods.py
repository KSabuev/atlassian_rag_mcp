"""Tests for the Jira methods registry (jira/methods.py)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from atlassian_rag_mcp.jira.methods import (
    execute_method,
    get_methods,
)
from atlassian_rag_mcp.jira.models import (
    IssueResult,
    ListResult,
    SearchResult,
    TransitionResult,
    WorklogResult,
)
from atlassian_rag_mcp.registry import MethodDescriptor, MethodParameter

EXPECTED_METHODS = {
    "get_issue", "search", "get_project_issues", "get_comments", "get_changelog",
    "get_transitions", "get_worklog", "get_issue_types", "get_priorities",
    "get_resolutions", "get_project_statuses", "get_projects", "myself",
    "get_issue_tree", "get_issue_labels", "get_issue_attachments",
    "get_issue_remote_links", "get_issue_watchers", "issue_exists", "count_issues",
    "get_boards", "get_epics", "get_issues_for_epic", "get_sprints", "get_sprint_issues",
    "get_project", "get_project_components", "get_project_versions",
    "get_all_statuses", "get_fields", "get_last_issue_number",
    "get_users", "find_user", "get_groups", "get_project_users",
}


@pytest.fixture
def fake_client():
    client = MagicMock()
    client.project_key = "PROJ"
    client.issues.get_issue = MagicMock(
        return_value=IssueResult(issue_key="PROJ-1", fields={"summary": "Test"}))
    client.issues.get_comments = MagicMock(
        return_value=ListResult(items=[{"id": "1", "body": "c"}], total=1))
    client.issues.get_changelog = MagicMock(
        return_value=IssueResult(issue_key="PROJ-1", fields={"changelog": {"histories": []}}))
    client.issues.get_transitions = MagicMock(
        return_value=TransitionResult(issue_key="PROJ-1", transitions=[{"id": "11"}]))
    client.issues.get_worklog = MagicMock(return_value=WorklogResult(issue_key="PROJ-1", worklogs=[]))
    client.issues.get_issue_types = MagicMock(return_value=ListResult(items=[{"name": "Bug"}], total=1))
    client.issues.get_priorities = MagicMock(return_value=ListResult(items=[{"name": "High"}], total=1))
    client.issues.get_resolutions = MagicMock(return_value=ListResult(items=[{"name": "Fixed"}], total=1))
    client.issues.get_project_statuses = MagicMock(return_value=ListResult(items=[{"name": "Done"}], total=1))
    client.issues.get_projects = MagicMock(
        return_value=ListResult(items=[{"key": "PROJ", "id": "10000", "name": "PROJ"}], total=1))
    client.search_service.search_issues = MagicMock(
        return_value=SearchResult(issues=[{"key": "PROJ-1"}], total=1, start_at=0, max_results=25))
    client.search_service.get_project_issues = MagicMock(
        return_value=SearchResult(issues=[{"key": "PROJ-1"}], total=1, start_at=0, max_results=25))
    client.issues.get_issue_tree = MagicMock(
        return_value=IssueResult(issue_key="PROJ-1", fields={"tree": {"key": "PROJ-1", "children": []}}))
    client.issues.get_issue_labels = MagicMock(return_value=ListResult(items=[{"name": "bug"}], total=1))
    client.issues.get_issue_attachments = MagicMock(return_value=ListResult(items=[], total=0))
    client.issues.get_issue_remote_links = MagicMock(return_value=ListResult(items=[], total=0))
    client.issues.get_issue_watchers = MagicMock(
        return_value=ListResult(items=[{"name": "user"}], total=1))
    client.issues.issue_exists = MagicMock(return_value=True)
    client.issues.count_issues = MagicMock(return_value=1)
    client.issues.get_boards = MagicMock(
        return_value=ListResult(items=[{"id": "10100", "name": "Board", "type": "scrum"}], total=1))
    client.issues.get_epics = MagicMock(
        return_value=ListResult(items=[{"id": "PROJ-127", "key": "PROJ-127"}], total=1))
    client.issues.get_issues_for_epic = MagicMock(
        return_value=SearchResult(issues=[{"key": "PROJ-1"}], total=1, start_at=0, max_results=25))
    client.issues.get_sprints = MagicMock(
        return_value=ListResult(items=[{"id": 7, "name": "S1", "state": "active"}], total=1))
    client.issues.get_sprint_issues = MagicMock(
        return_value=SearchResult(issues=[{"key": "PROJ-1"}], total=1, start_at=0, max_results=25))
    client.issues.get_project = MagicMock(return_value={"key": "PROJ", "name": "PROJ"})
    client.issues.get_project_components = MagicMock(return_value=ListResult(items=[], total=0))
    client.issues.get_project_versions = MagicMock(return_value=ListResult(items=[], total=0))
    client.issues.get_all_statuses = MagicMock(
        return_value=ListResult(items=[{"id": "1", "name": "Done"}], total=1))
    client.issues.get_fields = MagicMock(
        return_value=ListResult(items=[{"id": "summary", "name": "Summary"}], total=1))
    client.issues.get_last_issue_number = MagicMock(return_value=1)
    client.issues.get_users = MagicMock(
        return_value=ListResult(items=[{"name": "user"}], total=1))
    client.issues.find_user = MagicMock(
        return_value=ListResult(items=[{"name": "user"}], total=1))
    client.issues.get_groups = MagicMock(
        return_value=ListResult(items=[{"name": "group"}], total=1))
    client.issues.get_project_users = MagicMock(
        return_value=ListResult(items=[{"name": "user"}], total=1))
    client.issues.resolve_board_id = MagicMock(return_value="10100")
    client.jira.myself = MagicMock(
        return_value={"name": "user", "displayName": "User Name", "emailAddress": "u@example.com"})
    client.jira.get_all_priorities = MagicMock(return_value=[])
    client.jira.get_status_for_project = MagicMock(return_value=[])
    client.jira.issue_createmeta_issuetypes = MagicMock(return_value={"values": []})
    return client


class TestRegistry:
    def test_all_expected_methods_registered(self):
        ids = {m.id for m in get_methods()}
        assert ids == EXPECTED_METHODS

    def test_descriptors_are_valid(self):
        for m in get_methods():
            assert isinstance(m, MethodDescriptor)
            assert m.id
            assert m.description.strip()
            for p in m.parameters:
                assert isinstance(p, MethodParameter)
                assert p.name
                assert p.type in ("string", "array", "integer", "boolean")
                assert p.description.strip()

    def test_no_duplicate_ids(self):
        ids = [m.id for m in get_methods()]
        assert len(ids) == len(set(ids))

    def test_required_params_declared(self):
        methods = {m.id: m for m in get_methods()}
        issue_key_methods = (
            "get_issue", "get_comments", "get_changelog", "get_transitions", "get_worklog",
            "get_issue_tree", "get_issue_labels", "get_issue_attachments",
            "get_issue_remote_links", "get_issue_watchers", "issue_exists",
        )
        for mid in issue_key_methods:
            required = {p.name for p in methods[mid].parameters if p.required}
            assert "issue_key" in required, f"{mid} must require issue_key"

        assert "epic_id" in {p.name for p in methods["get_issues_for_epic"].parameters if p.required}
        assert "sprint_id" in {p.name for p in methods["get_sprint_issues"].parameters if p.required}
        assert not {p.name for p in methods["count_issues"].parameters if p.required}

    def test_search_has_filters(self):
        search = next(m for m in get_methods() if m.id == "search")
        names = {p.name for p in search.parameters}
        for expected in ("jql", "issuetype", "status", "priority",
                         "order_by", "assignee", "labels", "limit", "start_at"):
            assert expected in names

    def test_jql_param_is_lean(self):
        search = next(m for m in get_methods() if m.id == "search")
        jql = next(p for p in search.parameters if p.name == "jql")
        assert "JQL or free text" in jql.description
        assert "Empty when only structured filters" in jql.description
        # Combination/scoping/ordering semantics live in JQL_SCHEMA_HELP,
        # not in the parameter description (one fact — one home).
        assert "take precedence" not in jql.description
        assert "AND-ed" not in jql.description
        assert "auto-scoping" not in jql.description
        assert "ORDER BY" not in jql.description

    def test_jql_schema_help_documents_combination_semantics(self):
        from atlassian_rag_mcp.jira.methods import JQL_SCHEMA_HELP

        assert "AND-ed with any structured filters" in JQL_SCHEMA_HELP
        # The ORDER BY bullet is the single home for raw-JQL ordering.
        assert "ORDER BY inside raw JQL wins" in JQL_SCHEMA_HELP
        assert "kept at the very end" not in JQL_SCHEMA_HELP
        assert "raw project wins" in JQL_SCHEMA_HELP

    def test_jql_schema_help_rendered_in_search_only(self):
        # Marker without placeholders: get_methods() fills {project_key} in the
        # rendered help, so the raw constant is no longer a substring.
        marker = "Query language auto-detection (strict — verified on this instance):"
        for m in get_methods():
            if m.id == "search":
                assert marker in m.description
            else:
                assert marker not in m.description, m.id

    def test_count_issues_references_search(self):
        ci = next(m for m in get_methods() if m.id == "count_issues")
        assert "the same query inputs as search" in ci.description
        assert "identical to the search method" in ci.description

    def test_date_params_document_sign_semantics(self):
        search = next(m for m in get_methods() if m.id == "search")
        params = {p.name: p for p in search.parameters}
        for name in ("created", "updated", "duedate", "resolved"):
            desc = params[name].description
            assert "-7d = 7 days ago (the last 7 days" in desc, name
            assert "+7d" not in desc, name
            assert "future" not in desc, name

    def test_jql_schema_help_documents_sign_semantics(self):
        from atlassian_rag_mcp.jira.methods import JQL_SCHEMA_HELP
        # Sign semantics live in the date-parameter descriptions, not in JQL_SCHEMA_HELP;
        # no future (+) offsets or future-relative words are advertised.
        search = next(m for m in get_methods() if m.id == "search")
        for name in ("created", "updated", "duedate", "resolved"):
            desc = next(p.description for p in search.parameters if p.name == name)
            assert "-7d = 7 days ago (the last 7 days" in desc, name
            assert "+7d" not in desc, name
        assert "+7d" not in JQL_SCHEMA_HELP
        assert "tomorrow" not in JQL_SCHEMA_HELP
        assert "next week" not in JQL_SCHEMA_HELP
        assert "next month" not in JQL_SCHEMA_HELP

    def test_count_issues_descriptor_schema(self):
        ci = next(m for m in get_methods() if m.id == "count_issues")
        names = {p.name for p in ci.parameters}
        assert names == {"jql", "issuetype", "status", "priority", "assignee",
                         "labels", "order_by", "created", "updated", "duedate", "resolved"}
        jql = next(p for p in ci.parameters if p.name == "jql")
        assert not jql.required
        assert "structured filters" in ci.description

    def test_search_description_retry_line_single(self):
        search = next(m for m in get_methods() if m.id == "search")
        assert search.description.count("automatically retried with their numeric IDs") == 1

    def test_jql_schema_help_date_and_quirks(self):
        from atlassian_rag_mcp.jira.methods import JQL_SCHEMA_HELP
        lines = JQL_SCHEMA_HELP.split("\n")
        start = next(i for i, ln in enumerate(lines) if "Dates accept" in ln)
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if lines[j].startswith("  - "):
                end = j
                break
        date_para = "\n".join(lines[start:end])
        assert "-7d" in date_para
        assert "now() - 30d" in date_para
        quirks = [ln for ln in lines if "stripped automatically" in ln]
        assert quirks
        assert all("dates:" not in ln for ln in quirks)


class TestProjectKeysPlaceholder:
    def test_multi_key_list_in_project_key_params(self, monkeypatch):
        from atlassian_rag_mcp.config import settings

        monkeypatch.setattr(settings, "jira_project_key", "AAA,BBB,CCC,DDD")
        count = 0
        for m in get_methods():
            for p in m.parameters:
                if p.name == "project_key":
                    assert '["AAA", "BBB", "CCC", "DDD"]' in p.description, f"{m.id}: {p.description}"
                    count += 1
        assert count >= 10

    def test_issue_key_example_shows_first_key_and_list(self, monkeypatch):
        from atlassian_rag_mcp.config import settings

        monkeypatch.setattr(settings, "jira_project_key", "AAA,BBB,CCC,DDD")
        get_issue = next(m for m in get_methods() if m.id == "get_issue")
        issue_key = next(p for p in get_issue.parameters if p.name == "issue_key")
        assert "AAA-123" in issue_key.description
        assert '["AAA", "BBB", "CCC", "DDD"]' in issue_key.description
        assert "BBB-123" not in issue_key.description


class TestExecuteMethod:
    async def test_unknown_method(self, fake_client):
        result = await execute_method("nope", {}, fake_client)
        assert result["error"]
        assert "Unknown method: nope" in result["error"]
        assert result["data"] is None

    async def test_missing_required_param(self, fake_client):
        result = await execute_method("get_issue", {}, fake_client)
        assert result["error"]
        assert "Missing required parameter(s): issue_key" in result["error"]

    async def test_empty_string_counts_as_missing(self, fake_client):
        result = await execute_method("get_worklog", {"issue_key": ""}, fake_client)
        assert "Missing required parameter(s): issue_key" in result["error"]

    async def test_get_issue(self, fake_client):
        result = await execute_method("get_issue", {"issue_key": "PROJ-1"}, fake_client)
        assert result["error"] is None
        assert result["method"] == "get_issue"
        assert result["data"]["issue_key"] == "PROJ-1"
        fake_client.issues.get_issue.assert_called_once()

    async def test_search_delegates_to_search_service(self, fake_client):
        result = await execute_method(
            "search",
            {"jql": "project = PROJ", "limit": 3, "status": "Done"},
            fake_client,
        )
        assert result["error"] is None
        assert result["data"]["total"] == 1
        _, kwargs = fake_client.search_service.search_issues.call_args
        assert kwargs["limit"] == 3
        assert "project = PROJ" in kwargs["jql"]
        assert 'status = "Done"' in kwargs["jql"]

    async def test_search_falls_back_to_numeric_ids(self, fake_client):
        from requests.exceptions import HTTPError

        calls = {"n": 0}

        def flaky(jql, fields="summary", limit=25, start_at=0):
            calls["n"] += 1
            if "В работе" in jql:
                raise HTTPError("Значение 'В работе' отсутствует для поля 'status'.")
            return SearchResult(issues=[{"key": "PROJ-1"}], total=1, start_at=0, max_results=25)

        fake_client.search_service.search_issues = MagicMock(side_effect=flaky)
        fake_client.jira.get_status_for_project = MagicMock(return_value=[
            {"statuses": [{"name": "В работе", "id": 10100}]},
        ])
        result = await execute_method("search", {"status": "В работе"}, fake_client)
        assert result["error"] is None
        assert calls["n"] == 2
        second_jql = fake_client.search_service.search_issues.call_args_list[1][1]["jql"]
        assert "10100" in second_jql
        assert "В работе" not in second_jql

    async def test_search_zero_result_falls_back_to_ids(self, fake_client):
        calls: list[str] = []

        def search(jql, fields="summary", limit=25, start_at=0):
            calls.append(jql)
            if "10100" in jql:
                return SearchResult(issues=[{"key": "PROJ-1"}], total=42, start_at=0, max_results=25)
            return SearchResult(issues=[], total=0, start_at=0, max_results=25)

        fake_client.search_service.search_issues = MagicMock(side_effect=search)
        fake_client.jira.get_status_for_project = MagicMock(return_value=[
            {"statuses": [{"name": "Сделать", "id": 10100}]},
        ])
        result = await execute_method("search", {"status": "Сделать"}, fake_client)
        assert result["error"] is None
        assert result["data"]["total"] == 42
        assert len(calls) == 2
        assert "10100" in calls[1]

    async def test_search_non_value_error_not_retried(self, fake_client):
        from requests.exceptions import HTTPError

        fake_client.search_service.search_issues = MagicMock(
            side_effect=HTTPError("API down"),
        )
        with pytest.raises(HTTPError, match="API down"):
            await execute_method("search", {"status": "Done"}, fake_client)
        assert fake_client.search_service.search_issues.call_count == 1

    async def test_search_text_wrapped(self, fake_client):
        await execute_method("search", {"jql": "bug in login"}, fake_client)
        _, kwargs = fake_client.search_service.search_issues.call_args
        assert "text ~" in kwargs["jql"]
        assert 'project = "PROJ"' in kwargs["jql"]

    async def test_get_project_issues_defaults_to_configured(self, fake_client):
        result = await execute_method("get_project_issues", {}, fake_client)
        assert result["error"] is None
        _, kwargs = fake_client.search_service.get_project_issues.call_args
        assert kwargs["project_key"] == "PROJ"

    async def test_get_project_issues_without_project_fails(self):
        client = MagicMock()
        client.project_key = None
        with pytest.raises(ValueError, match="project_key is required"):
            await execute_method("get_project_issues", {}, client)

    async def test_myself(self, fake_client):
        result = await execute_method("myself", {}, fake_client)
        assert result["error"] is None
        assert result["data"]["name"] == "user"

    async def test_sdk_error_raises(self, fake_client):
        fake_client.issues.get_issue.side_effect = RuntimeError("API down")
        with pytest.raises(RuntimeError, match="API down"):
            await execute_method("get_issue", {"issue_key": "PROJ-1"}, fake_client)


class TestIssueServiceComments:
    def test_pagination_is_client_side(self):
        from atlassian_rag_mcp.jira.issues import IssueService
        jira = MagicMock()
        jira.issue_get_comments = MagicMock(return_value={
            "comments": [{"id": "1"}, {"id": "2"}, {"id": "3"}, {"id": "4"}, {"id": "5"}],
            "total": 5,
        })
        svc = IssueService(jira)
        r = svc.get_comments("PROJ-1", limit=2, start_at=1)
        assert [c["id"] for c in r.items] == ["2", "3"]
        assert r.total == 5

    def test_limit_zero_returns_all(self):
        from atlassian_rag_mcp.jira.issues import IssueService
        jira = MagicMock()
        jira.issue_get_comments = MagicMock(return_value={
            "comments": [{"id": "1"}, {"id": "2"}],
            "total": 2,
        })
        svc = IssueService(jira)
        r = svc.get_comments("PROJ-1", limit=0)
        assert len(r.items) == 2


class TestBuildIssueTree:
    def test_nested_from_flat_pairs(self):
        from atlassian_rag_mcp.jira.issues import build_issue_tree
        pairs = [{"PROJ-1": "PROJ-2"}, {"PROJ-2": "PROJ-0"}, {"PROJ-2": "PROJ-3"}]
        result = build_issue_tree(pairs, "PROJ-1")
        assert result["node_count"] == 4
        assert result["tree"]["key"] == "PROJ-1"
        assert [c["key"] for c in result["tree"]["children"]] == ["PROJ-2"]
        assert [c["key"] for c in result["tree"]["children"][0]["children"]] == ["PROJ-0", "PROJ-3"]
        assert result["edges"] == [
            {"from": "PROJ-1", "to": "PROJ-2"},
            {"from": "PROJ-2", "to": "PROJ-0"},
            {"from": "PROJ-2", "to": "PROJ-3"},
        ]

    def test_topmost_used_when_root_is_child(self):
        from atlassian_rag_mcp.jira.issues import build_issue_tree
        pairs = [{"PROJ-1": "PROJ-2"}, {"PROJ-0": "PROJ-1"}]
        result = build_issue_tree(pairs, "PROJ-1")
        assert result["tree"]["key"] == "PROJ-0"
        assert result["tree"]["children"][0]["key"] == "PROJ-1"
        assert result["tree"]["children"][0]["children"][0]["key"] == "PROJ-2"

    def test_cycle_does_not_recurse_forever(self):
        from atlassian_rag_mcp.jira.issues import build_issue_tree
        pairs = [{"A": "B"}, {"B": "A"}]
        result = build_issue_tree(pairs, "A")
        assert result["node_count"] == 2

    def test_empty_pairs(self):
        from atlassian_rag_mcp.jira.issues import build_issue_tree
        result = build_issue_tree([], "PROJ-1")
        assert result["tree"] == {"key": "PROJ-1", "children": []}
        assert result["node_count"] == 1


class TestBoardResolution:
    async def test_board_resolved_from_project(self, fake_client):
        fake_client.issues.resolve_board_id = MagicMock(return_value="10100")
        result = await execute_method("get_epics", {"project_key": "PROJ"}, fake_client)
        assert result["error"] is None
        _, kwargs = fake_client.issues.get_epics.call_args
        assert kwargs["board_id"] == "10100"

    async def test_board_id_param_wins(self, fake_client):
        result = await execute_method("get_epics", {"board_id": 42}, fake_client)
        assert result["error"] is None
        _, kwargs = fake_client.issues.get_epics.call_args
        assert kwargs["board_id"] == 42

    async def test_no_boards_error(self, fake_client):
        fake_client.issues.resolve_board_id = MagicMock(return_value=None)
        with pytest.raises(Exception, match="No boards found for project PROJ"):
            await execute_method("get_epics", {}, fake_client)

    async def test_no_board_no_project_error(self):
        client = MagicMock()
        client.project_key = None
        with pytest.raises(Exception, match="board_id is required"):
            await execute_method("get_sprints", {}, client)


class TestNewMethods:
    async def test_issue_exists_returns_bool(self, fake_client):
        fake_client.issues.issue_exists = MagicMock(return_value=True)
        result = await execute_method("issue_exists", {"issue_key": "PROJ-1"}, fake_client)
        assert result["error"] is None
        assert result["data"] is True

    async def test_count_issues_builds_jql(self, fake_client):
        fake_client.issues.count_issues = MagicMock(return_value=42)
        result = await execute_method("count_issues", {"jql": "bug"}, fake_client)
        assert result["data"] == 42
        jql = fake_client.issues.count_issues.call_args[0][0]
        assert "text ~" in jql
        assert 'project = "PROJ"' in jql

    async def test_find_user_requires_identifier(self, fake_client):
        with pytest.raises(Exception, match="at least one of"):
            await execute_method("find_user", {}, fake_client)

    async def test_find_user_by_query(self, fake_client):
        fake_client.issues.find_user = MagicMock(return_value=ListResult(items=[{"name": "u"}], total=1))
        result = await execute_method("find_user", {"query": "ann"}, fake_client)
        assert result["error"] is None
        assert result["data"]["total"] == 1

    async def test_get_project_returns_dict(self, fake_client):
        fake_client.issues.get_project = MagicMock(return_value={"key": "PROJ", "name": "PROJ"})
        result = await execute_method("get_project", {}, fake_client)
        assert result["error"] is None
        assert result["data"]["key"] == "PROJ"

    async def test_get_last_issue_number_returns_int(self, fake_client):
        fake_client.issues.get_last_issue_number = MagicMock(return_value=31185)
        result = await execute_method("get_last_issue_number", {}, fake_client)
        assert result["data"] == 31185

    async def test_get_issues_for_epic_no_board(self, fake_client):
        fake_client.issues.get_issues_for_epic = MagicMock(
            return_value=SearchResult(issues=[{"key": "PROJ-128"}], total=21, start_at=0, max_results=50),
        )
        result = await execute_method("get_issues_for_epic", {"epic_id": "PROJ-127", "limit": 5}, fake_client)
        assert result["error"] is None
        _, kwargs = fake_client.issues.get_issues_for_epic.call_args
        assert kwargs["epic_id"] == "PROJ-127"
        assert "board_id" not in kwargs

    async def test_sprint_issues_defaults(self, fake_client):
        fake_client.issues.get_sprint_issues = MagicMock(
            return_value=SearchResult(issues=[], total=0, start_at=0, max_results=25),
        )
        result = await execute_method("get_sprint_issues", {"sprint_id": 7}, fake_client)
        assert result["error"] is None
        _, kwargs = fake_client.issues.get_sprint_issues.call_args
        assert kwargs["limit"] == 25
        assert kwargs["start_at"] == 0


class TestIssueServicePlatformBehavior:
    def _svc(self, cloud: bool):
        from atlassian_rag_mcp.jira.issues import IssueService
        jira = MagicMock()
        jira.cloud = cloud
        return IssueService(jira), jira

    def test_count_issues_dc_uses_jql_total(self):
        svc, jira = self._svc(cloud=False)
        jira.jql = MagicMock(return_value={"total": 123, "issues": []})
        assert svc.count_issues("project = PROJ") == 123
        jira.jql.assert_called_once()

    def test_count_issues_cloud_uses_approximate(self):
        svc, jira = self._svc(cloud=True)
        jira.approximate_issue_count = MagicMock(return_value=7)
        assert svc.count_issues("project = PROJ") == 7

    def test_last_issue_number_from_key_string(self):
        svc, jira = self._svc(cloud=False)
        jira.get_project_issuekey_last = MagicMock(return_value="PROJ-1005")
        assert svc.get_last_issue_number("PROJ") == 1005

    def test_last_issue_number_from_dict(self):
        svc, jira = self._svc(cloud=True)
        jira.get_project_issuekey_last = MagicMock(return_value={"issueNumber": 42, "issueId": 1})
        assert svc.get_last_issue_number("PROJ") == 42

    def test_resolve_board_prefers_scrum(self):
        svc, jira = self._svc(cloud=False)
        jira.get_all_agile_boards = MagicMock(return_value={
            "values": [
                {"id": 1, "name": "Kanban", "type": "kanban"},
                {"id": 2, "name": "Scrum", "type": "scrum"},
            ],
            "size": 2,
        })
        assert svc.resolve_board_id("PROJ") == 2

    def test_resolve_board_falls_back_to_first(self):
        svc, jira = self._svc(cloud=False)
        jira.get_all_agile_boards = MagicMock(return_value={
            "values": [{"id": 9, "name": "K", "type": "kanban"}],
            "size": 1,
        })
        assert svc.resolve_board_id("PROJ") == 9

    def test_get_users_dc_uses_dot_search(self):
        svc, jira = self._svc(cloud=False)
        jira.user_find_by_user_string = MagicMock(return_value=[{"name": "u"}])
        r = svc.get_users(limit=5)
        assert r.total == 1
        assert jira.user_find_by_user_string.call_args.kwargs["username"] == "."

    def test_get_users_cloud_uses_listing(self):
        svc, jira = self._svc(cloud=True)
        jira.users_get_all = MagicMock(return_value=[{"name": "u"}])
        r = svc.get_users(limit=5)
        assert r.total == 1
        jira.users_get_all.assert_called_once()

    def test_find_user_dc_query_maps_to_username(self):
        svc, jira = self._svc(cloud=False)
        jira.user_find_by_user_string = MagicMock(return_value=[])
        svc.find_user(query="ivan")
        kwargs = jira.user_find_by_user_string.call_args.kwargs
        assert kwargs["username"] == "ivan"
        assert kwargs.get("query") is None


class TestMethodIdsAreValidEnumNames:
    def test_ids_are_valid_python_identifiers(self):
        for m in get_methods():
            assert m.id.isidentifier(), f"{m.id} is not a valid enum name"


_ALL_METHOD_PARAMS = [
    ("get_issue", {"issue_key": "PROJ-1"}),
    ("search", {}),
    ("get_project_issues", {}),
    ("get_comments", {"issue_key": "PROJ-1"}),
    ("get_changelog", {"issue_key": "PROJ-1"}),
    ("get_transitions", {"issue_key": "PROJ-1"}),
    ("get_worklog", {"issue_key": "PROJ-1"}),
    ("get_issue_types", {}),
    ("get_priorities", {}),
    ("get_resolutions", {}),
    ("get_project_statuses", {}),
    ("get_projects", {}),
    ("myself", {}),
    ("get_issue_tree", {"issue_key": "PROJ-1"}),
    ("get_issue_labels", {"issue_key": "PROJ-1"}),
    ("get_issue_attachments", {"issue_key": "PROJ-1"}),
    ("get_issue_remote_links", {"issue_key": "PROJ-1"}),
    ("get_issue_watchers", {"issue_key": "PROJ-1"}),
    ("issue_exists", {"issue_key": "PROJ-1"}),
    ("count_issues", {"jql": "bug"}),
    ("get_boards", {}),
    ("get_epics", {}),
    ("get_issues_for_epic", {"epic_id": "PROJ-127"}),
    ("get_sprints", {}),
    ("get_sprint_issues", {"sprint_id": 7}),
    ("get_project", {}),
    ("get_project_components", {}),
    ("get_project_versions", {}),
    ("get_all_statuses", {}),
    ("get_fields", {}),
    ("get_last_issue_number", {}),
    ("get_users", {}),
    ("find_user", {"query": "ann"}),
    ("get_groups", {}),
    ("get_project_users", {}),
]


class TestEveryMethodExecutes:
    """Smoke: every registered method runs without error on a happy path."""

    @pytest.mark.parametrize("mid, params", _ALL_METHOD_PARAMS)
    async def test_method_executes(self, mid, params, fake_client):
        res = await execute_method(mid, params, fake_client)
        assert res["error"] is None, f"{mid}: {res['error']}"
        assert res["method"] == mid
        assert res["data"] is not None


class TestErrorPaths:
    """execute_method is pure: backend failures propagate to the MCP tool layer."""

    async def test_exception_propagates(self, fake_client):
        fake_client.issues.get_issue = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            await execute_method("get_issue", {"issue_key": "PROJ-1"}, fake_client)

    async def test_empty_message_falls_back_to_type_name(self):
        from atlassian_rag_mcp.registry import error_message
        assert error_message(ValueError()) == "ValueError"

    async def test_http_error_uses_status_code(self):
        from atlassian_rag_mcp.registry import error_message

        class FakeHTTPError(Exception):
            def __init__(self):
                super().__init__()
                self.response = MagicMock(status_code=404)

        assert error_message(FakeHTTPError()) == "HTTP 404"

    async def test_sdk_error_on_any_method(self, fake_client):
        fake_client.issues.get_worklog = MagicMock(
            side_effect=RuntimeError("403 Forbidden")
        )
        with pytest.raises(RuntimeError, match="403 Forbidden"):
            await execute_method("get_worklog", {"issue_key": "PROJ-1"}, fake_client)


class TestJqlError:
    """JQL failures carry the failing query plus the server's own error text."""

    def test_search_issues_raises_jql_error_with_query_and_server_text(self):
        from requests.exceptions import HTTPError

        from atlassian_rag_mcp.exceptions import JqlError
        from atlassian_rag_mcp.jira.search import SearchService

        jira = MagicMock()
        jira.jql = MagicMock(
            side_effect=HTTPError(
                "Ошибка в запросе JQL: ... (строка 1, символ 15)",
                response=MagicMock(status_code=400),
            ),
        )
        svc = SearchService(jira)
        with pytest.raises(JqlError) as ei:
            svc.search_issues("bogus ~~~ AND x = y")
        msg = str(ei.value)
        assert "bogus ~~~ AND x = y" in msg
        assert "(HTTP 400)" in msg
        assert "Ошибка в запросе JQL" in msg

    def test_count_issues_dc_raises_jql_error(self):
        from requests.exceptions import HTTPError

        from atlassian_rag_mcp.exceptions import JqlError
        from atlassian_rag_mcp.jira.issues import IssueService

        jira = MagicMock()
        jira.cloud = False
        jira.jql = MagicMock(side_effect=HTTPError("Field 'x' does not exist."))
        svc = IssueService(jira)
        with pytest.raises(JqlError) as ei:
            svc.count_issues("x = 1")
        msg = str(ei.value)
        assert "x = 1" in msg
        assert "Field 'x' does not exist." in msg


class TestValueMaps:
    """Dynamic {issuetypes}/{statuses}/{priorities} placeholders in get_methods()."""

    @pytest.fixture(autouse=True)
    def _reset_value_maps(self):
        from atlassian_rag_mcp.jira.methods import reset_value_maps

        reset_value_maps()
        yield
        reset_value_maps()

    @staticmethod
    def _client(monkeypatch):
        from atlassian_rag_mcp.config import settings

        monkeypatch.setattr(settings, "jira_project_key", "PPP,MM,TTT")
        client = MagicMock()

        types_by_key = {
            "PPP": [{"name": "Задача", "id": "10002"}, {"name": "Подзадача", "id": "10003"}],
            "MM": [{"name": "Задача", "id": "10100"}],
            "TTT": [{"name": "Ошибка", "id": "10004"}, {"name": "Incident", "id": "10900"}],
        }
        statuses_by_key = {
            "PPP": [{"name": "Сделать", "id": "10000"}, {"name": "В работе", "id": "3"}],
            "MM": [{"name": "Закрыт", "id": "6"}],
            "TTT": [{"name": "В работе", "id": "3"}, {"name": "Решенные", "id": "5"}],
        }
        client.issues.get_issue_types = MagicMock(
            side_effect=lambda key: ListResult(
                items=types_by_key.get(key, []), total=len(types_by_key.get(key, []))))
        client.issues.get_project_statuses = MagicMock(
            side_effect=lambda key: ListResult(
                items=statuses_by_key.get(key, []), total=len(statuses_by_key.get(key, []))))
        client.issues.get_priorities = MagicMock(
            return_value=ListResult(items=[
                {"name": "Блокирующий", "id": "10000"}, {"name": "Критичный", "id": "10100"},
                {"name": "Высокий", "id": "2"},
            ], total=3))
        client.issues.get_resolutions = MagicMock(
            return_value=ListResult(
                items=[{"name": "Выполнено", "id": "10000"}, {"name": "Готово", "id": "10400"}], total=2))
        return client

    @staticmethod
    def _search_params(methods):
        search = next(m for m in methods if m.id == "search")
        return {p.name: p.description for p in search.parameters}

    def test_client_fills_placeholders_for_all_projects(self, monkeypatch):
        client = self._client(monkeypatch)
        methods = get_methods(client=client)
        params = self._search_params(methods)

        assert "PPP: Задача (10002), Подзадача (10003)" in params["issuetype"]
        assert "MM: Задача (10100)" in params["issuetype"]
        assert "TTT: Ошибка (10004), Incident (10900)" in params["issuetype"]
        assert "PPP: Сделать (10000), В работе (3)" in params["status"]
        assert "MM: Закрыт (6)" in params["status"]
        assert "TTT: В работе (3), Решенные (5)" in params["status"]
        assert "Блокирующий (10000), Критичный (10100), Высокий (2)" in params["priority"]
        for m in methods:
            for text in [m.description] + [p.description for p in m.parameters]:
                assert "{" not in text, f"{m.id}: {text}"

    def test_without_client_falls_back(self):
        methods = get_methods()
        params = self._search_params(methods)

        assert "(see get_issue_types)" in params["issuetype"]
        assert "(see get_project_statuses)" in params["status"]
        assert "(see get_priorities)" in params["priority"]
        for m in methods:
            for text in [m.description] + [p.description for p in m.parameters]:
                assert "{" not in text, f"{m.id}: {text}"

    def test_cache_prevents_second_fetch(self, monkeypatch):
        client = self._client(monkeypatch)

        get_methods(client=client)
        get_methods(client=client)

        assert client.issues.get_issue_types.call_count == 3
        assert client.issues.get_project_statuses.call_count == 3
        assert client.issues.get_priorities.call_count == 1
        assert client.issues.get_resolutions.call_count == 1

    def test_partial_statuses_failure_skips_project(self, monkeypatch):
        client = self._client(monkeypatch)
        original = client.issues.get_project_statuses

        def flaky(key):
            if key == "MM":
                raise RuntimeError("API down")
            return original(key)

        client.issues.get_project_statuses = MagicMock(side_effect=flaky)
        methods = get_methods(client=client)
        params = self._search_params(methods)

        assert "PPP: Сделать (10000), В работе (3)" in params["status"]
        assert "TTT: В работе (3), Решенные (5)" in params["status"]
        assert "MM:" not in params["status"]
        assert "PPP: Задача (10002), Подзадача (10003)" in params["issuetype"]
