"""CQLBuilder: pure function tests — no mocks needed."""

from __future__ import annotations

import datetime

import pytest

from atlassian_rag_mcp.confluence.cql_builder import (
    CQL_SYNTAX_HELP,
    CQLBuilder,
    CqlSearchParams,
    build_cql_query,
    build_space_cql,
    cql_looks_advanced,
)


class TestBasicFilters:
    def test_type(self):
        result = CQLBuilder().type("page").build()
        assert result == 'type = "page"'

    def test_space(self):
        result = CQLBuilder().space("PROJ").build()
        assert result == 'space = "PROJ"'

    def test_spaces_single(self):
        result = CQLBuilder().type("page").spaces("PROJ").build()
        assert result == 'type = "page" AND space = "PROJ"'

    def test_spaces_multiple(self):
        result = CQLBuilder().type("page").spaces("PROJ", "EEE").build()
        assert result == 'type = "page" AND space IN ("PROJ", "EEE")'

    def test_spaces_none_no_condition(self):
        result = CQLBuilder().type("page").spaces().build()
        assert result == 'type = "page"'

    def test_spaces_sanitizes(self):
        result = CQLBuilder().spaces('A"B').build()
        assert result == 'space = "A\\"B"'

    def test_text_contains(self):
        result = CQLBuilder().text_contains("api docs").build()
        assert result == 'text ~ "api docs"'

    def test_title(self):
        result = CQLBuilder().title("my page").build()
        assert result == 'title ~ "my page"'

    def test_label_single(self):
        result = CQLBuilder().label("archived").build()
        assert result == 'label IN ("archived")'

    def test_label_multi(self):
        result = CQLBuilder().label("a", "b", "c").build()
        assert result == 'label IN ("a", "b", "c")'

    def test_id_in(self):
        result = CQLBuilder().id_in("123", "456").build()
        assert result == "id IN (123,456)"

    def test_creator(self):
        result = CQLBuilder().creator("john").build()
        assert result == 'creator = "john"'

    def test_contributor(self):
        result = CQLBuilder().contributor("jane").build()
        assert result == 'contributor = "jane"'

    def test_creator_current_user(self):
        assert CQLBuilder().creator("me").build() == "creator = currentUser()"
        assert CQLBuilder().creator("currentUser").build() == "creator = currentUser()"
        assert CQLBuilder().creator("currentUser()").build() == "creator = currentUser()"

    def test_contributor_current_user(self):
        assert CQLBuilder().contributor("me").build() == "contributor = currentUser()"
        assert CQLBuilder().contributor("currentUser").build() == "contributor = currentUser()"

    def test_created(self):
        result = CQLBuilder().created("2025-01-01").build()
        assert result == 'created >= "2025-01-01"'

    def test_lastmodified(self):
        result = CQLBuilder().lastmodified("2025-01-01").build()
        assert result == 'lastmodified >= "2025-01-01"'

    def test_created_european_normalized(self):
        assert CQLBuilder().created("15.01.2025").build() == 'created >= "2025-01-15"'

    def test_created_iso_dots_normalized(self):
        assert CQLBuilder().created("2025.01.15").build() == 'created >= "2025-01-15"'

    def test_created_natural_language_today(self):
        today = datetime.date.today().isoformat()
        assert CQLBuilder().created("today").build() == f'created >= "{today}"'

    def test_created_relative(self):
        plus7 = (datetime.date.today() + datetime.timedelta(days=7)).isoformat()
        assert CQLBuilder().created("+7d").build() == f'created >= "{plus7}"'

    def test_lastmodified_natural_language(self):
        plus14 = (datetime.date.today() + datetime.timedelta(weeks=2)).isoformat()
        assert CQLBuilder().lastmodified("+2w").build() == f'lastmodified >= "{plus14}"'

    def test_created_cql_function_unquoted(self):
        # CQL date functions pass through parse_date and must stay UNQUOTED
        # (inside quotes the server cannot parse them).
        assert CQLBuilder().created('now("-4w")').build() == 'created >= now("-4w")'

    def test_lastmodified_cql_function_simple(self):
        assert CQLBuilder().lastmodified("startOfYear()").build() == "lastmodified >= startOfYear()"

    def test_raw_fragment(self):
        result = CQLBuilder().raw('text~"x" OR text~"y"').build()
        assert result == 'text~"x" OR text~"y"'


class TestChaining:
    def test_full_chain(self):
        result = CQLBuilder().type("page").space("FFF").label("archived").title("Onboarding Guide").build()
        assert "type = " in result
        assert "space = " in result
        assert "label IN " in result
        assert 'title ~ "Onboarding Guide"' in result
        parts = result.split(" AND ")
        assert len(parts) == 4

    def test_with_raw_and_space(self):
        result = CQLBuilder().space("TEST").raw('text~"term"').build()
        assert result == 'space = "TEST" AND text~"term"'

    def test_full_chain_with_filters(self):
        result = (
            CQLBuilder()
            .type("page")
            .space("FFF")
            .title("roadmap")
            .label("archived", "done")
            .creator("john")
            .contributor("jane")
            .created("2025-01-01")
            .lastmodified("2025-02-01")
            .build()
        )
        assert 'title ~ "roadmap"' in result
        assert 'label IN ("archived", "done")' in result
        assert 'creator = "john"' in result
        assert 'contributor = "jane"' in result
        assert 'created >= "2025-01-01"' in result
        assert 'lastmodified >= "2025-02-01"' in result
        assert len(result.split(" AND ")) == 8

    def test_order_by_last(self):
        result = CQLBuilder().space("PORJ").text_contains("query").order_by("lastmodified", "DESC").build()
        assert result.endswith("ORDER BY lastmodified DESC")

    def test_order_by_without_conditions(self):
        result = CQLBuilder().order_by("title", "ASC").build()
        assert result == "ORDER BY title ASC"

    def test_order_by_multi_field(self):
        result = CQLBuilder().order_by("created, updated").build()
        assert result == "ORDER BY created, updated DESC"

    def test_order_by_unsafe_rejected(self):
        with pytest.raises(ValueError):
            CQLBuilder().order_by("lastmodified; DROP TABLE")

    def test_build_hoists_order_by_from_raw(self):
        result = CQLBuilder().type("page").raw('text ~ "x" ORDER BY created DESC').creator("y").build()
        assert result == 'type = "page" AND text ~ "x" AND creator = "y" ORDER BY created DESC'

    def test_build_quoted_order_by_not_split(self):
        result = CQLBuilder().type("page").raw('text ~ "order by x"').creator("y").build()
        assert result == 'type = "page" AND text ~ "order by x" AND creator = "y"'


class TestSanitize:
    def test_double_quote_escaped(self):
        result = CQLBuilder().text_contains('say "hello"').build()
        assert 'text ~ "say \\"hello\\""' in result

    def test_backslash_escaped(self):
        result = CQLBuilder().text_contains("a\\b").build()
        assert "text ~ " in result
        assert "\\\\" in result or "a\\\\b" in result

    def test_raw_not_escaped(self):
        value = 'text~"term"'
        result = CQLBuilder().raw(value).build()
        assert result == 'text~"term"'


class TestEdgeCases:
    def test_no_conditions_returns_empty(self):
        assert CQLBuilder().build() == ""

    def test_order_by_overwrites(self):
        result = CQLBuilder().space("X").order_by("title", "ASC").order_by("id", "DESC").build()
        assert result == 'space = "X" ORDER BY id DESC'


class TestBuildSpaceCql:
    def test_empty_list(self):
        assert build_space_cql([]) == ""

    def test_none(self):
        assert build_space_cql(None) == ""

    def test_single(self):
        assert build_space_cql(["PROJ"]) == 'space = "PROJ"'

    def test_multiple(self):
        assert build_space_cql(["PROJ", "EEE"]) == 'space IN ("PROJ", "EEE")'

    def test_strips_and_drops_empty(self):
        assert build_space_cql(["  PROJ", "", " EEE "]) == 'space IN ("PROJ", "EEE")'

    def test_single_after_strip(self):
        assert build_space_cql(["  AAA "]) == 'space = "AAA"'


class TestLooksAdvanced:
    def test_cql_indicators(self):
        assert cql_looks_advanced('type = "page"') is True
        assert cql_looks_advanced("space IN (PROJ)") is True
        assert cql_looks_advanced('title ~ "x"') is True
        assert cql_looks_advanced("creator = currentUser()") is True
        assert cql_looks_advanced('text ~ "ABC"') is True
        assert cql_looks_advanced('text!~ "ABC"') is True
        assert cql_looks_advanced('label = "релиз" AND created >= "2025-01-01"') is True
        assert cql_looks_advanced("id IN (123)") is True
        assert cql_looks_advanced("ancestor = 123") is True
        assert cql_looks_advanced("created >= 2024-01-01") is True
        assert cql_looks_advanced('lastModified <= "2024-12-31"') is True
        assert cql_looks_advanced("contributor != currentUser()") is True
        assert cql_looks_advanced('label != "релиз"') is True
        assert cql_looks_advanced('label NOT IN ("a", "b")') is True
        assert cql_looks_advanced('space.key = "PROJ"') is True
        assert cql_looks_advanced('space.title ~ "Персональный"') is True
        assert cql_looks_advanced('space.type = "global"') is True
        assert cql_looks_advanced('status = "current"') is True
        assert cql_looks_advanced("version.number >= 3") is True
        assert cql_looks_advanced('version.by = "alice"') is True
        assert cql_looks_advanced('restriction.restrictToGroup = "ops"') is True
        assert cql_looks_advanced('historyLastModified >= "2025-01-01"') is True

    def test_plain_text(self):
        assert cql_looks_advanced("plain keywords") is False
        assert cql_looks_advanced("порядок оказания медицинской помощи") is False
        assert cql_looks_advanced("ABC") is False
        assert cql_looks_advanced("версия > 2.0") is False
        assert cql_looks_advanced("пример ~ 100 рублей") is False
        assert cql_looks_advanced("PMP IS NOT ready") is False
        assert cql_looks_advanced("PMP IN PROGRESS") is False
        assert cql_looks_advanced("SMP IS NULL") is False
        assert cql_looks_advanced("what is the status of the account") is False

    def test_conservative_detection_word_operator_word(self):
        """A bare ``=``/``~``/``>``/``<`` between two words is detected as raw CQL
        (conservative: real CQL is never silently wrapped as text); it is then
        rejected by validation unless the field is allowed."""
        assert cql_looks_advanced("a = b") is True
        assert cql_looks_advanced("x < y") is True
        assert cql_looks_advanced('spaceKey = "PROJ"') is True
        assert cql_looks_advanced("v2 > v1") is True


class TestBuildCqlQuery:
    def test_empty_input(self):
        assert build_cql_query(CqlSearchParams()) == ""

    def test_plain_text_wrapped(self):
        assert build_cql_query(CqlSearchParams(cql="api docs")) == 'type = "page" AND text ~ "api docs"'

    def test_plain_text_sanitized(self):
        assert build_cql_query(CqlSearchParams(cql='say "hi"')) == 'type = "page" AND text ~ "say \\"hi\\""'

    def test_raw_cql_passthrough(self):
        assert build_cql_query(CqlSearchParams(cql='title = "x"')) == 'type = "page" AND title = "x"'

    def test_dates_parsed_from_natural_language(self):
        plus7 = (datetime.date.today() + datetime.timedelta(days=7)).isoformat()
        result = build_cql_query(CqlSearchParams(cql="api", created="+7d"))
        assert result == f'type = "page" AND created >= "{plus7}" AND text ~ "api"'

    def test_space(self):
        assert (
            build_cql_query(CqlSearchParams(cql="api docs", space=["DEV"]))
            == 'type = "page" AND space = "DEV" AND text ~ "api docs"'
        )

    def test_filters_only(self):
        assert build_cql_query(CqlSearchParams(creator="alice")) == 'type = "page" AND creator = "alice"'

    def test_space_only(self):
        assert build_cql_query(CqlSearchParams(space=["DEV"])) == 'type = "page" AND space = "DEV"'

    def test_irrelevant_params_rejected(self):
        """vector is not a CQL param — rejected (extra="forbid"), same
        concept as the JQL builder."""
        with pytest.raises(Exception):
            CqlSearchParams(**{"vector": "q"})

    def test_filters_and_text(self):
        assert (
            build_cql_query(CqlSearchParams(cql="api", title="docs", label=["a", "b"]))
            == 'type = "page" AND title ~ "docs" AND label IN ("a", "b") AND text ~ "api"'
        )


class TestCqlFieldValidation:
    """Raw CQL using a field outside the allow-list is rejected client-side
    with a ValueError naming the unknown field and listing all allowed ones."""

    @pytest.mark.parametrize(
        "cql",
        [
            'status = "X"',
            'Status = "X"',
            "version.number >= 3",
            'version.by = "alice"',
            'restriction.restrictToGroup = "G"',
            'historyLastModified >= "2020-01-01"',
            'pageStatus = "X"',
            'spaceKey = "PROJ"',
            'space.bogus = "x"',
            'user.bogus = "x"',
            'languages.ru.bogus = "x"',
            'term.ru = "x"',
            "a = b",
            "P99 < 500ms",
            "v2 > v1",
            "ab = 1",
            'space = "PROJ" AND created is empty',  # no IS operator in CQL
            'space = "ABC" AND created is "empty"',
        ],
    )
    def test_unknown_field_rejected(self, cql):
        with pytest.raises(ValueError, match="Unknown CQL field"):
            build_cql_query(CqlSearchParams(cql=cql))

    def test_error_lists_allowed_fields(self):
        with pytest.raises(ValueError) as exc_info:
            build_cql_query(CqlSearchParams(cql='status = "X"'))
        message = str(exc_info.value)
        assert "'status'" in message
        assert "lastmodified" in message
        assert "space.key" in message
        assert "languages.<lang>.title" in message

    def test_error_shows_field_as_written(self):
        """Case is preserved in the message (validation is case-insensitive)."""
        with pytest.raises(ValueError, match="'Status'"):
            build_cql_query(CqlSearchParams(cql='Status = "X"'))

    @pytest.mark.parametrize(
        "cql",
        [
            'text ~ "ABC"',
            'LastModified <= "2024-12-31"',
            'TITLE ~ "x"',
            'Space.Key = "PROJ"',
            'label not in ("draft", "review")',
            'title IN ("A", "B")',
            "created >= now()",
            'space.key = "PROJ" ORDER BY lastmodified DESC',
            'languages.ru.title ~ "x"',
            'user.fullname = "jdoe"',
            'TermIdioms = "ABC"',
            'siteSearch ~ "x"',
            'refinedSite = "PROJ"',
            "termGlobal >= 0",
            "id IN (123, 456)",
            'watcher.userkey = "jdoe"',
            'container = "123"',
            'favorite = "jdoe"',
            'macro = "jira"',
            'content = "123"',
            "creator = currentUser()",
            'Space.Type = "global"',
            'Parent.Title ~ "x"',
            'RefinedCategory = "PROJ"',
            'User = "jdoe"',
            'Watcher = "jdoe"',
        ],
    )
    def test_allowed_fields_passthrough_verbatim(self, cql):
        """Original case is preserved — the query goes to the server as-is."""
        result = build_cql_query(CqlSearchParams(cql=cql))
        assert result == f'type = "page" AND {cql}'

    def test_plain_text_still_wrapped(self):
        assert build_cql_query(CqlSearchParams(cql="PMP IN PROGRESS")) == 'type = "page" AND text ~ "PMP IN PROGRESS"'
        assert (
            build_cql_query(CqlSearchParams(cql="порядок оказания медицинской помощи"))
            == 'type = "page" AND text ~ "порядок оказания медицинской помощи"'
        )


class TestIdAncestorParentParams:
    def test_id_in_clause(self):
        result = build_cql_query(CqlSearchParams(id=["123", "456"]))
        assert result == 'type = "page" AND id IN (123,456)'

    def test_id_single(self):
        result = build_cql_query(CqlSearchParams(id=["123"]))
        assert result == 'type = "page" AND id IN (123)'

    def test_id_scalar_coerced_to_list(self):
        result = build_cql_query(CqlSearchParams(id="123"))
        assert result == 'type = "page" AND id IN (123)'

    def test_id_numeric_coerced_to_str(self):
        result = build_cql_query(CqlSearchParams(id=[123, 456]))
        assert result == 'type = "page" AND id IN (123,456)'

    def test_ancestor_clause(self):
        result = build_cql_query(CqlSearchParams(ancestor="123"))
        assert result == 'type = "page" AND ancestor = 123'

    def test_parent_clause(self):
        result = build_cql_query(CqlSearchParams(parent="456"))
        assert result == 'type = "page" AND parent = 456'

    def test_new_params_combined_with_text_and_space(self):
        result = build_cql_query(CqlSearchParams(cql="api", space=["DEV"], ancestor="123", parent="456"))
        assert result == ('type = "page" AND space = "DEV" AND ancestor = 123 AND parent = 456 AND text ~ "api"')

    def test_id_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="numeric page id"):
            build_cql_query(CqlSearchParams(id=["12a"]))

    def test_ancestor_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="numeric page id"):
            build_cql_query(CqlSearchParams(ancestor="abc"))

    def test_parent_non_numeric_rejected(self):
        with pytest.raises(ValueError, match="numeric page id"):
            build_cql_query(CqlSearchParams(parent="12345x"))

    def test_fluent_ancestor(self):
        assert CQLBuilder().ancestor("123").build() == "ancestor = 123"

    def test_fluent_parent(self):
        assert CQLBuilder().parent("456").build() == "parent = 456"


class TestCqlSearchParams:
    def test_all_optional(self):
        p = CqlSearchParams()
        assert p.cql is None
        assert p.space is None
        assert p.title is None
        assert p.label is None
        assert p.created is None
        assert p.lastmodified is None
        assert p.id is None
        assert p.ancestor is None
        assert p.parent is None

    def test_scalar_label_coerced_to_list(self):
        assert CqlSearchParams(label="archived").label == ["archived"]

    def test_numeric_coerced_to_text(self):
        assert CqlSearchParams(created=2025).created == "2025"
        assert CqlSearchParams(lastmodified=2025).lastmodified == "2025"

    def test_unknown_field_rejected(self):
        with pytest.raises(Exception):
            CqlSearchParams(**{"cql": "q", "page_version": 3})

    def test_bool_rejected(self):
        with pytest.raises(Exception):
            CqlSearchParams(title=True)

    def test_dict_rejected(self):
        with pytest.raises(Exception):
            CqlSearchParams(label={"a": 1})


class TestCqlSyntaxHelp:
    def test_dates_line_documents_sign_semantics(self):
        """Live check: relative dates work ONLY inside now(); - = the last N days;
        a bare -7d as a date value is rejected (400). No future (+) offsets documented."""
        dates_line = next(
            line for line in CQL_SYNTAX_HELP.splitlines() if line.strip().startswith("- Dates:")
        )
        assert 'now("-7d")' in dates_line
        assert 'now("+7d")' not in dates_line
        assert "+7d" not in dates_line
        assert "- = the last N days" in dates_line
        assert "ONLY inside now()" in dates_line
