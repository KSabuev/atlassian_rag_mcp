"""Tests for JQLBuilder fluent API, build_jql_query, and JiraClient.search filter forwarding."""

from __future__ import annotations

import datetime

import pytest

from atlassian_rag_mcp.jira.jql_builder import (
    JQLBuilder,
    JqlSearchParams,
    build_jql_query,
    jql_looks_advanced,
    sanitize_jql_value,
)
from atlassian_rag_mcp.utils.query import build_date_condition, parse_date


class TestJQLBuilder:
    def test_project(self):
        assert JQLBuilder().project("PROJ").build() == 'project = "PROJ"'

    def test_chain_combines_with_and(self):
        jql = (
            JQLBuilder()
            .project("PROJ")
            .status("In Progress")
            .priority("High")
            .build()
        )
        assert jql == 'project = "PROJ" AND status = "In Progress" AND priority = "High"'

    def test_status_multi(self):
        jql = JQLBuilder().status("In Progress", "In Review").build()
        assert jql == 'status IN ("In Progress", "In Review")'

    def test_priority_multi(self):
        jql = JQLBuilder().priority("High", "Critical").build()
        assert jql == 'priority IN ("High", "Critical")'

    def test_order_by_raw(self):
        assert (
            JQLBuilder().project("PROJ").order_by_raw("priority DESC, updated DESC").build()
            == 'project = "PROJ" ORDER BY priority DESC, updated DESC'
        )
        assert (
            JQLBuilder().order_by_raw("ORDER BY created ASC").build() == "ORDER BY created ASC"
        )

    def test_order_by_raw_rejects_unsafe(self):
        with pytest.raises(ValueError):
            JQLBuilder().order_by_raw("created DESC; DROP TABLE x")

    def test_labels(self):
        jql = JQLBuilder().labels("frontend", "ui").build()
        assert jql == 'labels IN ("frontend", "ui")'

    def test_issuetype(self):
        assert JQLBuilder().issuetype("Bug").build() == 'issuetype = "Bug"'

    def test_assignee_current_user(self):
        assert JQLBuilder().assignee("currentUser").build() == "assignee = currentUser()"
        assert JQLBuilder().assignee("me").build() == "assignee = currentUser()"

    def test_assignee_fixed_value(self):
        assert JQLBuilder().assignee("jdoe").build() == 'assignee = "jdoe"'

    def test_assignee_empty_value_maps_to_empty(self):
        assert JQLBuilder().assignee("EMPTY").build() == "assignee is EMPTY"
        assert JQLBuilder().assignee("empty").build() == "assignee is EMPTY"

    def test_date_filters(self):
        b = JQLBuilder()
        b.created("2025-01-01")
        b.updated("2025-03-01")
        b.duedate("2025-05-01")
        b.resolved("2025-07-01")
        jql = b.build()
        assert 'created >= "2025-01-01"' in jql
        assert 'updated >= "2025-03-01"' in jql
        assert 'duedate >= "2025-05-01"' in jql
        assert 'resolved >= "2025-07-01"' in jql

    def test_date_functions_unquoted(self):
        assert JQLBuilder().updated("now()").build() == "updated >= now()"
        assert JQLBuilder().created('now("-4w")').build() == 'created >= now("-4w")'
        assert JQLBuilder().duedate("startOfYear()").build() == "duedate >= startOfYear()"
        assert JQLBuilder().resolved("endOfMonth()").build() == "resolved >= endOfMonth()"

    def test_text_contains_and_raw(self):
        b = JQLBuilder().text_contains("login").raw("sprint in openSprints()")
        assert b.build() == 'text ~ "login" AND sprint in openSprints()'

    def test_order_by(self):
        assert (
            JQLBuilder().project("PROJ").order_by("created", "DESC").build()
            == 'project = "PROJ" ORDER BY created DESC'
        )

    def test_build_empty(self):
        assert JQLBuilder().build() == ""

    def test_escaping(self):
        assert JQLBuilder().project('we"ird').build() == 'project = "we\\"ird"'
        assert JQLBuilder().text_contains('a\\b').build() == 'text ~ "a\\\\b"'

    def test_sanitize_jql_value(self):
        assert sanitize_jql_value('a"b') == 'a\\"b'
        assert sanitize_jql_value('a\\b') == 'a\\\\b'
        assert sanitize_jql_value('a\\"b') == 'a\\\\\\"b'

    def test_sanitize_strips_empty_parentheses(self):
        assert sanitize_jql_value("openSprints()") == "openSprints"
        assert sanitize_jql_value("get_user() function") == "get_user function"
        assert sanitize_jql_value("a ( ) b") == "a b"
        assert sanitize_jql_value("ABC (повторный)") == "ABC (повторный)"

    def test_parse_date_iso(self):
        assert parse_date("2025-01-01") == "2025-01-01"

    def test_parse_date_today(self):
        assert parse_date("today") == datetime.date.today().isoformat()

    def test_parse_date_relative(self):
        assert parse_date("+7d") == (datetime.date.today() + datetime.timedelta(days=7)).isoformat()

    def test_parse_date_european(self):
        assert parse_date("31.12.2025") == "2025-12-31"

    def test_parse_date_iso_dots(self):
        assert parse_date("2025.01.15") == "2025-01-15"

    def test_parse_date_month_clamped(self):
        # Jan 31 + 1 month must not produce an invalid date.
        from atlassian_rag_mcp.utils.query import _clamp_to_month

        assert _clamp_to_month(2026, 2, 31).isoformat() == "2026-02-28"
        assert _clamp_to_month(2024, 2, 31).isoformat() == "2024-02-29"
        assert _clamp_to_month(2025, 12, 1).isoformat() == "2025-12-01"


class TestBuildDateCondition:
    """Shared date-condition helper (utils.query) used by the CQL and JQL builders."""

    def test_natural_date_quoted(self):
        assert build_date_condition("updated", "2025-01-01", lambda v: v) == 'updated >= "2025-01-01"'
        expected = f'created >= "{datetime.date.today().isoformat()}"'
        assert build_date_condition("created", "today", lambda v: v) == expected

    def test_date_function_unquoted(self):
        assert build_date_condition("created", "now()", lambda v: v) == "created >= now()"
        assert build_date_condition("lastmodified", 'now("-4w")', lambda v: v) == 'lastmodified >= now("-4w")'
        assert build_date_condition("updated", "startOfYear()", lambda v: v) == "updated >= startOfYear()"

    def test_value_sanitized(self):
        assert build_date_condition("created", "2025-01-01", lambda v: v.replace("01", "X")) == 'created >= "2025-X-X"'


class TestJqlLooksAdvanced:
    def test_raw_jql(self):
        assert jql_looks_advanced('status = "В работе"') is True
        assert jql_looks_advanced("status in (3, 6)") is True
        assert jql_looks_advanced('status IN ("a", "b")') is True
        assert jql_looks_advanced("status NOT in (3)") is True
        assert jql_looks_advanced("assignee is EMPTY") is True
        assert jql_looks_advanced("assignee is not EMPTY") is True
        assert jql_looks_advanced("component is not EMPTY") is True
        assert jql_looks_advanced("resolution = EMPTY") is True
        assert jql_looks_advanced("created >= 2025-01-01") is True
        assert jql_looks_advanced("created >= -7d") is True
        assert jql_looks_advanced("updated <= now()") is True
        assert jql_looks_advanced('summary ~ "ABC"') is True
        assert jql_looks_advanced('text ~ "ABC" OR text ~ "XYZ"') is True
        assert jql_looks_advanced('labels IN ("a", "b")') is True
        assert jql_looks_advanced('labels NOT IN ("a")') is True
        assert jql_looks_advanced('project = "PROJ"') is True
        assert jql_looks_advanced('key = "PROJ-1"') is True
        assert jql_looks_advanced("id in (1, 2)") is True
        assert jql_looks_advanced("sprint in openSprints()") is True
        assert jql_looks_advanced("assignee = currentUser()") is True
        assert jql_looks_advanced('"Epic Link" is not EMPTY') is True
        assert jql_looks_advanced('"Custom Field" = "value"') is True
        assert jql_looks_advanced('NOT (status = "Закрыт")') is True
        assert jql_looks_advanced("issuetype = 10004") is True
        assert jql_looks_advanced('duedate >= "2025-01-01"') is True
        assert jql_looks_advanced("priority = 2") is True
        assert jql_looks_advanced("sprint in openSprints()") is True
        assert jql_looks_advanced("resolved >= 2024-01-01") is True
        assert jql_looks_advanced('resolved > "2024-01-01"') is True
        assert jql_looks_advanced("resolved is EMPTY") is True
        assert jql_looks_advanced("resolved in (EMPTY)") is True

    def test_plain_text(self):
        assert jql_looks_advanced("ABC") is False
        assert jql_looks_advanced("тестовые данные") is False
        assert jql_looks_advanced("что такое тест") is False
        assert jql_looks_advanced("bug in login") is False
        assert jql_looks_advanced("a = b") is False
        assert jql_looks_advanced("версия > 2.0") is False
        assert jql_looks_advanced("x < y") is False
        assert jql_looks_advanced("PMP IN PROGRESS") is False
        assert jql_looks_advanced("what is the status") is False
        assert jql_looks_advanced("SMP IS NULL") is False
        assert jql_looks_advanced("a~b") is False
        assert jql_looks_advanced("создан ORDER BY id") is False
        assert jql_looks_advanced("обработка EMPTY значений") is False
        assert jql_looks_advanced("status in 2025") is False
        assert jql_looks_advanced("the status is in progress") is False
        assert jql_looks_advanced("created in 2025") is False
        assert jql_looks_advanced('say "hi"') is False
        assert jql_looks_advanced('has "quote"') is False
        assert jql_looks_advanced("label is not empty") is False
        assert jql_looks_advanced("openSprints()") is False
        assert jql_looks_advanced("sort order by severity") is False
        assert jql_looks_advanced("release.2026.2.3.964-alt") is False


class TestBuildJql:
    def test_plain_text_wrapped(self):
        jql = build_jql_query(JqlSearchParams(jql="bug in login", project="PROJ"))
        assert 'text ~ "bug in login"' in jql
        assert 'project = "PROJ"' in jql
        assert "ORDER BY priority DESC, updated DESC" in jql

    def test_raw_jql_passthrough(self):
        jql = build_jql_query(JqlSearchParams(jql="issuetype = Epic", project="PROJ"))
        assert jql.startswith("issuetype = Epic")
        assert 'project = "PROJ"' in jql

    def test_raw_jql_project_not_duplicated(self):
        jql = build_jql_query(JqlSearchParams(jql="project = OTHER AND issuetype = Epic", project="PROJ"))
        assert 'project = "PROJ"' not in jql
        assert "project = OTHER" in jql

    def test_raw_jql_order_by_not_duplicated(self):
        jql = build_jql_query(JqlSearchParams(jql="issuetype = Epic ORDER BY created DESC", project="PROJ"))
        assert jql.count("ORDER BY") == 1
        assert "ORDER BY created DESC" in jql

    def test_structured_filters_combined(self):
        jql = build_jql_query(
            JqlSearchParams(
                jql="login",
                project="PROJ",
                issuetype="Bug",
                status=["In Progress", "In Review"],
                priority="High",
                assignee="jdoe",
                labels=["frontend", "ui"],
            )
        )
        assert 'issuetype = "Bug"' in jql
        assert 'status IN ("In Progress", "In Review")' in jql
        assert 'priority = "High"' in jql
        assert 'assignee = "jdoe"' in jql
        assert 'labels IN ("frontend", "ui")' in jql
        assert 'text ~ "login"' in jql
        assert 'project = "PROJ"' in jql

    def test_order_by_param(self):
        jql = build_jql_query(JqlSearchParams(project="PROJ", order_by="created ASC"))
        assert jql == 'project = "PROJ" ORDER BY created ASC'

    def test_assignee_me(self):
        jql = build_jql_query(JqlSearchParams(project="PROJ", assignee="me"))
        assert "assignee = currentUser()" in jql

    def test_assignee_empty_in_build_jql(self):
        jql = build_jql_query(JqlSearchParams(project="PROJ", assignee="EMPTY"))
        assert "assignee is EMPTY" in jql
        assert 'assignee = "EMPTY"' not in jql

    def test_labels_as_single_string(self):
        jql = build_jql_query(JqlSearchParams(project="PROJ", labels="frontend"))
        assert 'labels IN ("frontend")' in jql

    def test_date_filters_natural_language(self):
        today = datetime.date.today().isoformat()
        jql = build_jql_query(
            JqlSearchParams(
                project="PROJ",
                created="2025-01-01",
                updated="today",
                duedate="2025-03-01",
                resolved="2025-05-01",
            )
        )
        assert 'created >= "2025-01-01"' in jql
        assert f'updated >= "{today}"' in jql
        assert 'duedate >= "2025-03-01"' in jql
        assert 'resolved >= "2025-05-01"' in jql

    def test_filters_without_text(self):
        jql = build_jql_query(JqlSearchParams(project="PROJ", status="Done"))
        assert jql.startswith('status = "Done"')

    def test_empty_input_returns_empty(self):
        assert build_jql_query(JqlSearchParams()) == ""
        assert build_jql_query(JqlSearchParams(project=None)) == ""

    def test_project_only(self):
        assert (
            build_jql_query(JqlSearchParams(project="PROJ"))
            == 'project = "PROJ" ORDER BY priority DESC, updated DESC'
        )

    def test_escaping_in_text(self):
        jql = build_jql_query(JqlSearchParams(jql='has "quote"', project="PROJ"))
        assert 'text ~ "has \\"quote\\""' in jql


class TestJqlSearchParams:
    def test_all_optional(self):
        p = JqlSearchParams()
        assert p.jql is None
        assert p.project is None
        assert p.issuetype is None
        assert p.status is None
        assert p.priority is None
        assert p.assignee is None
        assert p.labels is None
        assert p.created is None
        assert p.updated is None
        assert p.duedate is None
        assert p.resolved is None
        assert p.order_by is None

    def test_unknown_fields_rejected(self):
        """Unknown parameters (e.g. legacy names, fields/limit/start_at) are
        rejected with the field name in the error."""
        for bad in ({"jql_query": "x"}, {"project_key": "P"}, {"issue_type": "Bug"},
                    {"statuses": "Done"}, {"priorities": "High"}, {"exclude_done": True},
                    {"exclude_status": "Closed"}, {"days_back": 7}, {"created_after": "x"},
                    {"created_before": "x"}, {"updated_before": "x"}, {"fields": "summary"},
                    {"limit": 10}, {"start_at": 0}):
            with pytest.raises(Exception, match=list(bad)[0]):
                JqlSearchParams(project="PROJ", **bad)

    def test_numeric_coerced_to_text(self):
        assert JqlSearchParams(priority=3).priority == ["3"]
        assert JqlSearchParams(jql=42).jql == "42"

    def test_text_list_coercion(self):
        assert JqlSearchParams(labels="a").labels == ["a"]
        assert JqlSearchParams(labels=[1, "b"]).labels == ["1", "b"]

    def test_text_field_rejects_bool(self):
        with pytest.raises(Exception):
            JqlSearchParams(status=True)

    def test_text_field_rejects_dict(self):
        with pytest.raises(Exception):
            JqlSearchParams(labels={"a": 1})
