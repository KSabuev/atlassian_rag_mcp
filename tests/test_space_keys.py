"""Tests for multi-space support: settings, build_filter."""

from __future__ import annotations

from atlassian_rag_mcp.config.settings import Settings
from atlassian_rag_mcp.storage.utils import build_filter


def _settings(space_key: str, jira_project_keys: str = "") -> Settings:
    return Settings(
        CONFLUENCE_URL="https://c.example.com/",
        CONFLUENCE_USERNAME="u",
        CONFLUENCE_PASSWORD="p",
        CONFLUENCE_SPACE_KEY=space_key,
        JIRA_URL="https://j.example.com/",
        JIRA_USERNAME="u",
        JIRA_PASSWORD="p",
        JIRA_PROJECT_KEYS=jira_project_keys,
        EMBEDDING_URL="http://e.example.com/v1",
        EMBEDDING_MODEL="m",
        _env_file=None,
    )


class TestSettingsSpaceKeys:
    def test_single(self):
        assert _settings("AAA").confluence_space_keys == ["AAA"]

    def test_multiple(self):
        assert _settings("AAA,EEE").confluence_space_keys == ["AAA", "EEE"]

    def test_strips_whitespace(self):
        assert _settings("AAA, EEE ,FFF").confluence_space_keys == ["AAA", "EEE", "FFF"]

    def test_drops_empty(self):
        assert _settings("AAA,,EEE,").confluence_space_keys == ["AAA", "EEE"]

    def test_empty(self):
        assert _settings("").confluence_space_keys == []


class TestSettingsJiraProjectKeys:
    def test_single(self):
        assert _settings("AAA", jira_project_keys="AAA").jira_project_keys == ["AAA"]

    def test_multiple(self):
        s = _settings("AAA", jira_project_keys="AAA,BBB,CCC,DDD")
        assert s.jira_project_keys == ["AAA", "BBB", "CCC", "DDD"]

    def test_strips_whitespace(self):
        assert _settings("AAA", jira_project_keys="AAA, BBB ,CCC").jira_project_keys == ["AAA", "BBB", "CCC"]

    def test_drops_empty(self):
        assert _settings("AAA", jira_project_keys="AAA,,BBB,").jira_project_keys == ["AAA", "BBB"]

    def test_empty(self):
        assert _settings("AAA").jira_project_keys == []

    def test_default_project_is_first(self):
        s = _settings("AAA", jira_project_keys="AAA,BBB,CCC,DDD")
        assert s.jira_project_keys[0] == "AAA"


class TestBuildFilterSpaces:
    def test_no_spaces(self):
        assert build_filter(spaces=None) is None

    def test_empty_spaces(self):
        assert build_filter(spaces=[]) is None

    def test_single_space(self):
        assert build_filter(spaces=["AAA"]) == {
            "must": [{"key": "space_key", "match": {"any": ["AAA"]}}]
        }

    def test_multiple_spaces(self):
        assert build_filter(spaces=["AAA", "EEE"]) == {
            "must": [{"key": "space_key", "match": {"any": ["AAA", "EEE"]}}]
        }

    def test_combined_with_page_ids(self):
        f = build_filter(page_ids=["p1"], spaces=["AAA"])
        keys = [c["key"] for c in f["must"]]
        assert "page_id" in keys
        assert "space_key" in keys



