"""Tests for Jira Pydantic models and integration."""

from __future__ import annotations

import asyncio

import pytest

from atlassian_rag_mcp.jira.models import (
    IssueResult,
    ListResult,
    SearchResult,
    TransitionResult,
    WorklogResult,
)


class TestJiraModels:
    """Test Jira Pydantic model validation and serialization."""

    def test_issue_result_valid(self):
        r = IssueResult(issue_key="PROJ-123", fields={"summary": "Test"})
        assert r.issue_key == "PROJ-123"
        assert r.fields["summary"] == "Test"
        assert r.error is None

    def test_issue_result_error(self):
        r = IssueResult(issue_key="PROJ-123", error="Not found")
        assert r.error == "Not found"
        assert r.fields == {}

    def test_search_result_valid(self):
        r = SearchResult(
            issues=[{"key": "PROJ-1"}, {"key": "PROJ-2"}],
            total=100,
            start_at=0,
            max_results=25,
        )
        assert len(r.issues) == 2
        assert r.total == 100
        assert r.start_at == 0
        assert r.max_results == 25
        assert r.error is None

    def test_search_result_error(self):
        r = SearchResult(error="API timeout")
        assert r.error == "API timeout"
        assert r.issues == []
        assert r.total == 0

    def test_transition_result_valid(self):
        r = TransitionResult(
            issue_key="PROJ-123",
            transitions=[{"id": "1", "name": "In Progress"}],
        )
        assert r.issue_key == "PROJ-123"
        assert len(r.transitions) == 1
        assert r.transitions[0]["name"] == "In Progress"

    def test_worklog_result_valid(self):
        r = WorklogResult(
            issue_key="PROJ-123",
            worklogs=[{"author": "user1", "timeSpent": "1h"}],
        )
        assert r.issue_key == "PROJ-123"
        assert len(r.worklogs) == 1

    def test_list_result_valid(self):
        r = ListResult(items=[{"id": 1}, {"id": 2}], total=2)
        assert len(r.items) == 2
        assert r.total == 2
        assert r.error is None

    def test_serialization_roundtrip(self):
        """Test model_dump() and model_validate() roundtrip."""
        original = SearchResult(
            issues=[{"key": "PROJ-1", "fields": {"summary": "Test"}}],
            total=5,
            start_at=0,
            max_results=10,
        )
        data = original.model_dump()
        restored = SearchResult.model_validate(data)
        assert restored.total == 5
        assert restored.issues[0]["key"] == "PROJ-1"


class TestJiraModelsEdgeCases:
    """Test edge cases and validation."""

    def test_empty_search_result(self):
        r = SearchResult()
        assert r.issues == []
        assert r.total == 0
        assert r.start_at == 0
        assert r.max_results == 0

    def test_issue_result_no_fields(self):
        r = IssueResult(issue_key="PROJ-123")
        assert r.fields == {}

    def test_list_result_empty(self):
        r = ListResult()
        assert r.items == []
        assert r.total == 0


def _jira_configured() -> bool:
    try:
        from atlassian_rag_mcp.config import settings
        return bool(settings.jira_url)
    except Exception:
        return False


def _jira_project_key() -> str:
    from atlassian_rag_mcp.config import settings
    keys = settings.jira_project_keys
    return keys[0] if keys else "PROJ"


@pytest.mark.skipif(
    not _jira_configured(),
    reason="Jira integration tests require JIRA_URL (env or .env)",
)
@pytest.mark.asyncio(loop_scope="class")
class TestJiraIntegration:
    """Integration tests against real Jira/Zephyr (require network access)."""

    @pytest.fixture(scope="class")
    async def ctx(self):
        from atlassian_rag_mcp.api.lifespan import _init
        ctx = await _init()
        yield ctx
        await ctx.aclose()

    async def test_jira_search_returns_searchresult(self, ctx):
        result = await asyncio.to_thread(
            ctx.jira_client.search_service.search_issues,
            jql=f"project = {_jira_project_key()}", limit=3,
        )
        assert isinstance(result, SearchResult)
        assert result.total > 0
        assert len(result.issues) <= 3
        assert result.error is None

    async def test_jira_get_project_issues_returns_searchresult(self, ctx):
        result = await asyncio.to_thread(
            ctx.jira_client.search_service.get_project_issues,
            project_key=_jira_project_key(), limit=3,
        )
        assert isinstance(result, SearchResult)
        assert result.total > 0
        assert result.error is None

    async def test_jira_get_issue_returns_issueresult(self, ctx):
        search = await asyncio.to_thread(
            ctx.jira_client.search_service.search_issues,
            jql=f"project = {_jira_project_key()}", limit=1,
        )
        assert search.issues
        key = search.issues[0]["key"]

        result = await asyncio.to_thread(ctx.jira_client.issues.get_issue, issue_key=key)
        assert isinstance(result, IssueResult)
        assert result.issue_key == key
        assert result.error is None
        assert isinstance(result.fields, dict)

    async def test_jira_get_transitions_returns_transitionresult(self, ctx):
        search = await asyncio.to_thread(
            ctx.jira_client.search_service.search_issues,
            jql=f"project = {_jira_project_key()}", limit=50,
        )
        assert search.issues

        picked = None
        for item in search.issues:
            result = await asyncio.to_thread(ctx.jira_client.issues.get_transitions, item["key"])
            if result.transitions:
                picked = result
                break

        assert picked is not None, "no issue with available transitions found in project"
        assert picked.issue_key
        assert isinstance(picked.transitions, list)
        assert len(picked.transitions) > 0

    async def test_jira_get_worklog_returns_worklogresult(self, ctx):
        search = await asyncio.to_thread(
            ctx.jira_client.search_service.search_issues,
            jql=f"project = {_jira_project_key()}", limit=1,
        )
        key = search.issues[0]["key"]

        result = await asyncio.to_thread(ctx.jira_client.issues.get_worklog, key)
        assert isinstance(result, WorklogResult)
        assert result.issue_key == key
        assert isinstance(result.worklogs, list)

    async def test_get_issue_types_returns_listresult(self, ctx):
        result = await asyncio.to_thread(ctx.jira_client.issues.get_issue_types, _jira_project_key())
        assert isinstance(result, ListResult)
        assert isinstance(result.items, list)
        assert result.total == len(result.items)
        assert result.total > 0

    async def test_get_priorities_returns_listresult(self, ctx):
        result = await asyncio.to_thread(ctx.jira_client.issues.get_priorities)
        assert isinstance(result, ListResult)
        assert result.total > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
