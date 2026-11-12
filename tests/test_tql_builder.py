"""Tests for the Zephyr TQL contract (tql_builder) and its dedup usage in methods.

Mirrors test_cql_builder.py / test_jql_builder.py: the TQL contract must live
in a single source (TQL_HELP) and the search surfaces must never drift apart.
"""

from __future__ import annotations

import pytest

from atlassian_rag_mcp.config import settings
from atlassian_rag_mcp.jira.plugins.zephyr import get_methods
from atlassian_rag_mcp.jira.plugins.zephyr.tql_builder import (
    FIELDS_OVERRIDE_DESC,
    LIMIT_DESC,
    OFFSET_DESC,
    PROJECT_ID_DESC,
    QUERY_DESC,
    TQL_HELP,
    TQL_TESTCASE_EXTRA,
    TQL_TESTPLAN_EXTRA,
    TQL_TESTRUN_EXTRA,
    TQLBuilder,
    TqlSearchParams,
    build_tql_query,
    sanitize_tql_value,
    tql_looks_advanced,
)

SEARCH_METHODS = ("search_test_cases", "search_test_runs", "search_test_plans")


def _descriptors() -> dict:
    return {m.id: m for m in get_methods()}


def _params(descriptor) -> dict:
    return {p.name: p for p in descriptor.parameters}


class TestTqlHelpContract:
    def test_contains_verified_facts(self):
        assert "verified up to 500" in TQL_HELP
        assert "ORDER BY" in TQL_HELP
        assert "exact match only, no ~" in TQL_HELP
        for field in ("projectId", "folderTreeId", "key", "name"):
            assert field in TQL_HELP

    def test_contains_common_400_fields(self):
        for field in ("status", "statusId", "createdOn", "updatedOn", "now()"):
            assert field in TQL_HELP

    def test_entity_extras_are_distinct(self):
        assert "testCase.id" in TQL_TESTCASE_EXTRA
        assert "estimatedTime" in TQL_TESTCASE_EXTRA
        assert "start" in TQL_TESTRUN_EXTRA and "finish" in TQL_TESTRUN_EXTRA
        assert "createdOn|updatedOn" in TQL_TESTPLAN_EXTRA


class TestSearchMethodDescriptions:
    @pytest.mark.parametrize("mid", SEARCH_METHODS)
    def test_embeds_tql_help_verbatim(self, mid):
        assert TQL_HELP in _descriptors()[mid].description

    def test_entity_extras_attached(self):
        descs = _descriptors()
        assert TQL_TESTCASE_EXTRA in descs["search_test_cases"].description
        assert TQL_TESTRUN_EXTRA in descs["search_test_runs"].description
        assert TQL_TESTPLAN_EXTRA in descs["search_test_plans"].description

    @pytest.mark.parametrize("mid", SEARCH_METHODS)
    def test_no_boilerplate_duplication(self, mid):
        desc = _descriptors()[mid].description
        for phrase in (
            "TQL (Zephyr Scale query language)",
            "verified up to 500",
            "sorted by name ascending",
            "exact match only, no ~",
        ):
            assert desc.count(phrase) == 1, f"{mid}: {phrase!r} repeated"


class TestSharedParameters:
    def test_search_methods_share_param_descriptions(self):
        descs = _descriptors()
        for name in ("project_key", "project_id", "tql", "limit", "offset"):
            trio = [_params(descs[mid])[name].description for mid in SEARCH_METHODS]
            assert trio[0] == trio[1] == trio[2], name

    def test_project_key_desc_is_single_source(self):
        descs = _descriptors()
        with_pk = [
            next(p.description for p in d.parameters if p.name == "project_key")
            for d in descs.values()
            if any(p.name == "project_key" for p in d.parameters)
        ]
        assert len(with_pk) == 10
        assert len(set(with_pk)) == 1

    def test_fields_override_is_single_source(self):
        descs = _descriptors()
        overrides = [
            next(p.description for p in d.parameters if p.name == "fields")
            for d in descs.values()
            if any(p.name == "fields" for p in d.parameters)
            and next(p.description for p in d.parameters if p.name == "fields") == FIELDS_OVERRIDE_DESC
        ]
        assert len(overrides) == 8

    def test_key_params(self):
        descs = _descriptors()
        expected = {
            "test_case_key": ("search_test_cases", 6),
            "test_run_key": ("search_test_runs", 3),
            "test_plan_key": ("search_test_plans", 1),
        }
        for key_name, (source, count) in expected.items():
            keys = [
                (d.id, p)
                for d in descs.values()
                for p in d.parameters
                if p.name == key_name
            ]
            assert len(keys) == count, key_name
            for method_id, p in keys:
                assert p.required
                assert source in p.description
                # no cross-references to other search methods
                for other in expected:
                    if other != key_name:
                        assert expected[other][0] not in p.description

    def test_limit_offset_descs(self):
        descs = _descriptors()
        for mid in SEARCH_METHODS:
            params = _params(descs[mid])
            assert params["limit"].description == LIMIT_DESC
            assert params["offset"].description == OFFSET_DESC
            assert params["tql"].description == QUERY_DESC
            assert params["project_id"].description == PROJECT_ID_DESC


class TestPlaceholderSubstitutionStability:
    def test_fresh_copies_reflect_settings_change(self, monkeypatch):
        monkeypatch.setattr(settings, "jira_project_key", "AAA")
        first = {m.id: m for m in get_methods()}
        monkeypatch.setattr(settings, "jira_project_key", "BBB")
        second = {m.id: m for m in get_methods()}
        pk_first = next(
            p.description
            for p in first["get_folders"].parameters
            if p.name == "project_key"
        )
        pk_second = next(
            p.description
            for p in second["get_folders"].parameters
            if p.name == "project_key"
        )
        assert '"AAA"' in pk_first
        assert '"BBB"' in pk_second

    def test_no_raw_placeholders_after_substitution(self):
        for m in get_methods():
            assert "{project_key}" not in m.description
            assert "{project_keys}" not in m.description
            for p in m.parameters:
                assert "{project_key}" not in p.description
                assert "{project_keys}" not in p.description


class TestBuildTqlQuery:
    def test_auto_build_project_and_folder(self):
        q = build_tql_query(TqlSearchParams(entity="testCase", project_id=12345, folder_id=555))
        assert q == "testCase.projectId IN (12345) AND testCase.folderTreeId IN (555)"

    @pytest.mark.parametrize("entity", ("testCase", "testRun", "testPlan"))
    def test_auto_build_project_only(self, entity):
        q = build_tql_query(TqlSearchParams(entity=entity, project_id=77))
        assert q == f"{entity}.projectId IN (77)"

    def test_auto_build_folder_only(self):
        q = build_tql_query(TqlSearchParams(entity="testRun", folder_id=183))
        assert q == "testRun.folderTreeId IN (183)"

    def test_auto_build_empty(self):
        assert build_tql_query(TqlSearchParams(entity="testCase")) == ""

    def test_raw_query_passthrough_verbatim(self):
        q = 'testCase.projectId IN (12345) AND testCase.key IN ("PROJ-123","PROJ-1234")'
        assert build_tql_query(TqlSearchParams(entity="testCase", tql=q)) == q

    def test_raw_query_with_order_by(self):
        q = 'testPlan.key IN ("PROJ-1") ORDER BY testPlan.createdOn DESC'
        assert build_tql_query(TqlSearchParams(entity="testPlan", tql=q)) == q

    @pytest.mark.parametrize(
        ("entity", "plain"),
        [
            ("testCase", "Verify the BILL attribute value"),
            ("testRun", "Some test run name"),
            ("testPlan", "Release smoke plan"),
        ],
    )
    def test_plain_text_wrapped_as_name_exact(self, entity, plain):
        wrapped = build_tql_query(TqlSearchParams(entity=entity, tql=plain))
        assert wrapped == f'{entity}.name = "{plain}"'

    def test_plain_text_respects_entity(self):
        assert build_tql_query(TqlSearchParams(entity="testPlan", tql="x")) == 'testPlan.name = "x"'
        assert build_tql_query(TqlSearchParams(entity="testRun", tql="x")) == 'testRun.name = "x"'

    def test_plain_text_escapes_quotes_and_backslash(self):
        assert (
            build_tql_query(TqlSearchParams(entity="testCase", tql='a"b'))
            == 'testCase.name = "a\\"b"'
        )

    def test_plain_with_bare_in_stays_plain(self):
        # bare IN without a parenthesis is not an operator -> wrapped, not raw
        assert (
            build_tql_query(TqlSearchParams(entity="testCase", tql="PMP IN PROGRESS"))
            == 'testCase.name = "PMP IN PROGRESS"'
        )

    def test_plain_with_operator_rejected(self):
        # conservative: raw-looking text is never silently wrapped as a name value
        with pytest.raises(ValueError, match="Unknown TQL field: 'bogus'"):
            build_tql_query(TqlSearchParams(entity="testCase", tql="bogus ~~~"))

    def test_unknown_entity_rejected(self):
        with pytest.raises(ValueError, match="Unknown TQL entity"):
            build_tql_query(TqlSearchParams(entity="testFoo", project_id=1))

    def test_unknown_param_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TqlSearchParams(entity="testCase", status="Draft")

    @pytest.mark.parametrize(
        ("query", "entity", "pattern"),
        [
            ("bogus ~~~", "testCase", "Unknown TQL field: 'bogus'"),
            ('testCase.status = "Draft"', "testCase", "Unknown TQL field: 'testCase.status'"),
            ("testCase.statusId IN (76)", "testCase", "Unknown TQL field: 'testCase.statusId'"),
            ('testCase.name ~ "x"', "testCase", "does not support operator '~'"),
            ("testCase.folderTreeId = 5", "testCase", "does not support operator '='"),
            ("testCase.createdOn >= now()", "testCase", "Unknown TQL field: 'testCase.createdOn'"),
            ("now() > 1", "testCase", "Function calls"),
            ("testCase.updatedOn >= \"2020-01-01\"", "testCase", "Unknown TQL field: 'testCase.updatedOn'"),
            ("testRun.id IN (131)", "testRun", "Unknown TQL field: 'testRun.id'"),
            ("testPlan.id IN (2)", "testPlan", "Unknown TQL field: 'testPlan.id'"),
            ("testRun.projectId IN (10202)", "testCase", "Allowed TQL fields for testCase"),
            ("testPlan.key IN (\"P1\") ORDER BY testPlan.status", "testPlan", "Unknown TQL ORDER BY field"),
        ],
    )
    def test_validation_rejects(self, query, entity, pattern):
        with pytest.raises(ValueError, match=pattern):
            build_tql_query(TqlSearchParams(entity=entity, tql=query))

    @pytest.mark.parametrize(
        "query",
        [
            "testCase.projectId IN (12345) AND testCase.folderTreeId IN (555)",
            "testCase.projectId NOT IN (12345)",
            'testCase.key IN ("PROJ-123")',
            'testCase.name = "Verify the value"',
            "testCase.id IN (390, 391)",
            "testCase.estimatedTime >= 0",
            "TESTCASE.PROJECTID IN (12345)",  # case-insensitive
            'testCase.name = "order by name = x"',  # quoted values are not fields
            "testRun.projectId = 12345",
            "testPlan.name = \"x\" ORDER BY testPlan.name ASC",
            "testPlan.projectId IN (1) ORDER BY testPlan.name, testPlan.id DESC",
        ],
    )
    def test_validation_accepts(self, query):
        entity = query.split(".", 1)[0]
        assert build_tql_query(TqlSearchParams(entity=entity, tql=query)) == query

    def test_sanitize_escapes_quotes_and_backslash(self):
        assert sanitize_tql_value('a"b') == 'a\\"b'
        assert sanitize_tql_value("a\\b") == "a\\\\b"
        assert sanitize_tql_value("") == ""


class TestTqlLooksAdvanced:
    @pytest.mark.parametrize(
        "text",
        [
            "testCase.projectId IN (12345)",
            'testCase.name = "x"',
            "testRun.projectId NOT IN (1) AND testRun.key IN (\"P-1\") ORDER BY testRun.name",
            "estimatedTime >= 0",
            "a = b",  # bare operator between words -> conservative raw
            "a ~ b",
            "now() > 1",  # function-call shape -> raw (rejected in validation)
        ],
    )
    def test_detects_raw(self, text):
        assert tql_looks_advanced(text)

    @pytest.mark.parametrize(
        "text",
        [
            "Verify the BILL attribute value",
            "PMP IN PROGRESS",  # bare IN without parenthesis
            "тестовые данные",
            "",
        ],
    )
    def test_detects_plain(self, text):
        assert not tql_looks_advanced(text)

    def test_quoted_values_not_syntax(self):
        # quoted strip: no operator left around the value
        assert not tql_looks_advanced('"order by x" and "= x"')


class TestTqlBuilder:
    def test_fluent_chain(self):
        q = (
            TQLBuilder("testCase")
            .project_id(12345)
            .folder_id(555)
            .key("PROJ-123")
            .name("Verify the value")
            .order_by("name", "ASC")
            .build()
        )
        assert q == (
            'testCase.projectId IN (12345) AND testCase.folderTreeId IN (555) '
            'AND testCase.key IN ("PROJ-123") AND testCase.name = "Verify the value" '
            "ORDER BY testCase.name ASC"
        )

    def test_key_value_sanitized(self):
        q = TQLBuilder("testCase").key('a"b').build()
        assert q == 'testCase.key IN ("a\\"b")'

    def test_name_value_sanitized(self):
        q = TQLBuilder("testCase").name('a"b').build()
        assert q == 'testCase.name = "a\\"b"'

    def test_order_by_allow_list(self):
        with pytest.raises(ValueError, match="Unknown TQL ORDER BY field"):
            TQLBuilder("testCase").order_by("status")
        q = TQLBuilder("testPlan").order_by("createdOn", "desc").build()
        assert q == "ORDER BY testPlan.createdOn DESC"

    def test_order_by_raw_dedup(self):
        q = (
            TQLBuilder("testPlan")
            .raw('testPlan.key IN ("P1")')
            .order_by_raw("ORDER BY testPlan.id ASC")
            .build()
        )
        assert q == 'testPlan.key IN ("P1") ORDER BY testPlan.id ASC'

    def test_raw_order_by_hoisted_to_end(self):
        q = (
            TQLBuilder("testPlan")
            .raw('testPlan.key IN ("P1") ORDER BY testPlan.name DESC')
            .folder_id(9)
            .build()
        )
        assert q == 'testPlan.key IN ("P1") AND testPlan.folderTreeId IN (9) ORDER BY testPlan.name DESC'

    def test_raw_with_bad_field_rejected_on_build(self):
        with pytest.raises(ValueError, match="Unknown TQL field"):
            TQLBuilder("testCase").raw("testCase.status = 1").build()

    def test_unknown_entity(self):
        with pytest.raises(ValueError, match="Unknown TQL entity"):
            TQLBuilder("testFoo")

    def test_empty_build(self):
        assert TQLBuilder("testCase").build() == ""

    def test_entity_case_insensitive(self):
        q = TQLBuilder("TESTCASE").project_id(1).build()
        assert q == "testCase.projectId IN (1)"
