"""Tests for the Zephyr methods registry (zephyr_get_methods / zephyr_execute_method)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atlassian_rag_mcp.jira.plugins.zephyr import (
    execute_method,
    get_methods,
    registry,
)

_PLUGINregistry = (
    "get_projects", "search_test_cases", "get_test_case", "get_test_case_versions",
    "get_test_case_history", "get_test_case_trace_links", "get_test_case_comments",
    "get_test_case_attachments", "get_folders", "search_test_runs", "get_test_run",
    "get_test_run_items", "get_test_run_results", "get_test_result", "search_test_plans",
    "get_test_plan", "get_test_case_statuses", "get_test_case_priorities",
    "get_test_result_statuses", "get_custom_fields", "get_project_permissions",
    "get_datasets", "get_confluence_pages", "download_attachment",
)


@pytest.fixture
def make_client():
    def _make(plugin=None, project_key="PROJ"):
        client = MagicMock()
        client.plugins = {"zephyr": plugin} if plugin is not None else {}
        client.project_key = project_key
        return client
    return _make


@pytest.fixture
def plugin():
    p = MagicMock()
    for name in _PLUGINregistry:
        setattr(p, name, AsyncMock())
    return p


class TestRegistry:
    def test_method_count(self):
        assert len(registry) == 24

    def test_descriptors_are_valid(self):
        for d in get_methods():
            assert d.id
            assert d.description
            for p in d.parameters:
                assert p.name
                assert p.type in ("string", "array", "integer", "boolean")

    def test_ids_are_unique(self):
        ids = [d.id for d in get_methods()]
        assert len(ids) == len(set(ids))

    def test_every_method_class_exists(self):
        for d in get_methods():
            assert d.id in registry
            assert hasattr(registry[d.id], "execute")


class TestProjectKeysPlaceholder:
    def test_multi_key_list_in_project_key_params(self, monkeypatch):
        from atlassian_rag_mcp.config import settings

        monkeypatch.setattr(settings, "jira_project_key", "AAA,BBB,CCC,DDD")
        count = 0
        for d in get_methods():
            for p in d.parameters:
                if p.name == "project_key":
                    assert '["AAA", "BBB", "CCC", "DDD"]' in p.description, f"{d.id}: {p.description}"
                    count += 1
        assert count == 10

    def test_defaults_to_first_project(self, monkeypatch):
        from atlassian_rag_mcp.config import settings

        monkeypatch.setattr(settings, "jira_project_key", "AAA,BBB,CCC,DDD")
        d = next(d for d in get_methods() if d.id == "search_test_cases")
        p = next(p for p in d.parameters if p.name == "project_key")
        assert "Defaults to the first project in the list." in p.description


class TestExecuteMethod:
    async def test_unknown_method(self, plugin, make_client):
        res = await execute_method("nope", {}, make_client(plugin))
        assert "Unknown method" in res["error"]
        assert res["method"] == "nope"

    async def test_missing_required_parameter(self, plugin, make_client):
        res = await execute_method("get_test_case", {}, make_client(plugin))
        assert "Missing required parameter" in res["error"]
        assert "test_case_key" in res["error"]

    async def test_missing_required_array_parameter(self, plugin, make_client):
        res = await execute_method("get_confluence_pages", {"page_ids": []}, make_client(plugin))
        assert "page_ids" in res["error"]

    async def test_plugin_not_enabled(self, make_client):
        with pytest.raises(ValueError, match="not enabled"):
            await execute_method("get_projects", {}, make_client(None))

    async def test_get_projects(self, plugin, make_client):
        plugin.get_projects.return_value = [{"id": 1, "key": "P1"}]
        res = await execute_method("get_projects", {}, make_client(plugin))
        assert res["error"] is None
        assert res["data"] == [{"id": 1, "key": "P1"}]

    async def test_project_key_defaults_to_client(self, plugin, make_client):
        plugin.get_test_case_statuses.return_value = [{"id": 1}]
        res = await execute_method("get_test_case_statuses", {}, make_client(plugin))
        assert res["error"] is None
        plugin.get_test_case_statuses.assert_awaited_once_with("PROJ")

    async def test_project_key_explicit_overrides(self, plugin, make_client):
        plugin.get_folders.return_value = []
        res = await execute_method(
            "get_folders", {"project_key": "P2", "folder_type": "testplan"},
            make_client(plugin),
        )
        assert res["error"] is None
        plugin.get_folders.assert_awaited_once_with("P2", "testplan")

    async def test_project_key_missing_everywhere(self, plugin, make_client):
        with pytest.raises(ValueError, match="project_key is required"):
            await execute_method(
                "get_test_case_statuses", {}, make_client(plugin, project_key=None)
            )

    async def test_download_attachment_converts_int(self, plugin, make_client):
        plugin.download_attachment.return_value = {"id": 123, "content_base64": "aGk="}
        res = await execute_method(
            "download_attachment", {"attachment_id": "123"}, make_client(plugin)
        )
        assert res["error"] is None
        plugin.download_attachment.assert_awaited_once_with(123)

    async def test_get_test_result_converts_int(self, plugin, make_client):
        plugin.get_test_result.return_value = {"id": 55}
        res = await execute_method(
            "get_test_result", {"test_result_id": "55"}, make_client(plugin)
        )
        assert res["error"] is None
        plugin.get_test_result.assert_awaited_once_with(55, None)

    async def test_confluence_pages_converts_ids(self, plugin, make_client):
        plugin.get_confluence_pages.return_value = []
        await execute_method(
            "get_confluence_pages", {"page_ids": [1, "2"]}, make_client(plugin)
        )
        plugin.get_confluence_pages.assert_awaited_once_with([1, 2])

    async def test_plugin_error_propagates(self, plugin, make_client):
        plugin.get_projects.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            await execute_method("get_projects", {}, make_client(plugin))

    async def test_plugin_error_empty_message_falls_back_to_type_name(self):
        from atlassian_rag_mcp.registry import error_message

        plugin_error = RuntimeError()
        assert error_message(plugin_error) == "RuntimeError"


_ALL_ZEPHYR_PARAMS = [
    ("get_projects", {}),
    ("search_test_cases", {}),
    ("get_test_case", {"test_case_key": "PROJ-1"}),
    ("get_test_case_versions", {"test_case_key": "PROJ-1"}),
    ("get_test_case_history", {"test_case_key": "PROJ-1"}),
    ("get_test_case_trace_links", {"test_case_key": "PROJ-1"}),
    ("get_test_case_comments", {"test_case_key": "PROJ-1"}),
    ("get_test_case_attachments", {"test_case_key": "PROJ-1"}),
    ("get_folders", {}),
    ("search_test_runs", {}),
    ("get_test_run", {"test_run_key": "PROJ-1"}),
    ("get_test_run_items", {"test_run_key": "PROJ-1"}),
    ("get_test_run_results", {"test_run_key": "PROJ-1"}),
    ("get_test_result", {"test_result_id": 1}),
    ("search_test_plans", {}),
    ("get_test_plan", {"test_plan_key": "PROJ-1"}),
    ("get_test_case_statuses", {}),
    ("get_test_case_priorities", {}),
    ("get_test_result_statuses", {}),
    ("get_custom_fields", {}),
    ("get_project_permissions", {}),
    ("get_datasets", {}),
    ("get_confluence_pages", {"page_ids": [1]}),
    ("download_attachment", {"attachment_id": 1}),
]


class TestEveryMethodExecutes:
    """Smoke: every registered method runs without error on a happy path."""

    @pytest.mark.parametrize("mid, params", _ALL_ZEPHYR_PARAMS)
    async def test_method_executes(self, mid, params, plugin, make_client):
        res = await execute_method(mid, params, make_client(plugin))
        assert res["error"] is None, f"{mid}: {res['error']}"
        assert res["method"] == mid


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
    reason="Zephyr integration tests require JIRA_URL (env or .env)",
)
@pytest.mark.asyncio(loop_scope="class")
class TestZephyrIntegration:
    """Integration tests against real Zephyr (require network access)."""

    @pytest.fixture(scope="class")
    async def ctx(self):
        from atlassian_rag_mcp.api.lifespan import _init, reset_ctx

        reset_ctx()
        ctx = await _init()
        yield ctx
        await ctx.aclose()
        reset_ctx()

    async def test_zephyr_plugin_available(self, ctx):
        plugin = ctx.jira_client.plugins.get("zephyr")
        assert plugin is not None
        available = await plugin.is_available()
        assert available is True

    async def test_zephyr_get_projects(self, ctx):
        plugin = ctx.jira_client.plugins.get("zephyr")
        projects = await plugin.get_projects()
        assert isinstance(projects, list)
        assert len(projects) > 0
        assert all("key" in p and "id" in p for p in projects)

    async def test_zephyr_get_folders(self, ctx):
        plugin = ctx.jira_client.plugins.get("zephyr")
        folders = await plugin.get_folders(_jira_project_key())
        assert isinstance(folders, list)
        assert len(folders) > 0

    async def test_zephyr_search_test_cases(self, ctx):
        plugin = ctx.jira_client.plugins.get("zephyr")
        result = await plugin.search_test_cases(project_key=_jira_project_key(), limit=3)
        assert isinstance(result, dict)
        assert "total" in result
        assert "results" in result
        assert result["total"] > 0
        assert len(result["results"]) <= 3

    async def test_zephyr_search_test_runs(self, ctx):
        plugin = ctx.jira_client.plugins.get("zephyr")
        result = await plugin.search_test_runs(project_key=_jira_project_key(), limit=3)
        assert isinstance(result, dict)
        assert "total" in result
        assert "results" in result
        assert result["total"] > 0
        assert len(result["results"]) <= 3
