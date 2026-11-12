"""FTS query builder: AND/OR keyword joining and raw-query passthrough."""

from __future__ import annotations

import pytest

from atlassian_rag_mcp.confluence.fts_builder import (
    FtsBuilder,
    FtsSearchParams,
    build_fts_query,
    fts_looks_advanced,
    validate_fts_query,
)
from atlassian_rag_mcp.exceptions import FtsError


class TestLooksAdvanced:
    @pytest.mark.parametrize(
        "text",
        [
            "text:docs",
            'text:"a b"',
            "text:(a OR b)",
            "page:123",
            "space:PROJ",
            "chunk_id:c1",
            "title:тест",
            '"quick fox"~2 AND api^3',
            'say "hi" there',
            "a AND b",
            "a OR b",
            "a NOT b",
            "doc*",
            "te*t",
        ],
    )
    def test_advanced_indicators(self, text: str) -> None:
        assert fts_looks_advanced(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "plain keywords",
            "порядок оказания помощи",
            "api docs python",
            "api:docs",
            "foo:barBaz",
            "какие сервисы обновляются",
            "Что такое ABC (полное название)?",
            "Что такое hotfix?",
            "онбординг: план по неделям",
            "/application/{id}/allowed-actions",
            "release.2026.2.3.964-alt",
            "a and b",
            "mytext:foo",
            "FOO.NOT.VALID",
            "te?t",
            "-docs",
            "(docs api)",
            "a^2",
            "api^3",
            "тест^2",
            "ABC.DEF.01^4",
        ],
    )
    def test_plain_text(self, text: str) -> None:
        assert fts_looks_advanced(text) is False


class TestBuildFtsQuery:
    def test_empty_input(self) -> None:
        assert build_fts_query(FtsSearchParams()) == ""
        assert build_fts_query(FtsSearchParams(fts="  ")) == ""

    def test_or_mode_default(self) -> None:
        assert (
            build_fts_query(FtsSearchParams(fts="api docs"))
            == "((title:(api OR docs))^3 OR text:(api OR docs))"
        )

    def test_foreign_vector_query_rejected(self) -> None:
        """vector is not a FTS-builder field — unknown keys are rejected
        (extra="forbid", same concept as the JQL/CQL builders)."""
        with pytest.raises(Exception):
            FtsSearchParams(fts="explicit", vector="fallback")

    def test_single_token_is_not_grouped(self) -> None:
        assert build_fts_query(FtsSearchParams(fts="api")) == "((title:(api))^3 OR text:(api))"

    def test_ru_stemming_and_stopwords(self) -> None:
        q = build_fts_query(FtsSearchParams(fts="Ролевая модель и права пользователей"))
        expected = "((title:(ролев OR модел OR прав OR пользовател))^3 "
        expected += "OR text:(ролев OR модел OR прав OR пользовател))"
        assert q == expected

    def test_all_stopwords_yields_empty(self) -> None:
        assert build_fts_query(FtsSearchParams(fts="что и как")) == ""

    def test_advanced_query_passthrough(self) -> None:
        q = build_fts_query(FtsSearchParams(fts='"quick fox"~2 AND api^3'))
        assert q == '"quick fox"~2 AND api^3'

    def test_phrase_with_special_chars_passthrough(self) -> None:
        assert build_fts_query(FtsSearchParams(fts='text:"docs"')) == 'text:"docs"'

    def test_query_with_quotes_is_advanced_passthrough(self) -> None:
        assert build_fts_query(FtsSearchParams(fts='say "hi" there')) == 'say "hi" there'

    def test_title_field_query_is_raw_passthrough(self) -> None:
        assert build_fts_query(FtsSearchParams(fts="title:тест")) == "title:тест"

    def test_ascii_field_prefix_is_rejected(self) -> None:
        # strict mode: an ASCII ``field:`` that is not one of the allowed
        # fields is an error (previously silently searched as plain text)
        with pytest.raises(FtsError, match="unknown field 'foo'"):
            build_fts_query(FtsSearchParams(fts="foo:barBaz"))

    def test_plain_query_with_punctuation_tokens_quoted(self) -> None:
        """Digits-first "26.3.1" matches no PATTERNS family — no exact branch."""
        expected = "((title:(соста OR релиз OR (26 AND 3 AND 1) OR сервис))^3 "
        expected += "OR text:(соста OR релиз OR (26 AND 3 AND 1) OR сервис))"
        assert (
            build_fts_query(FtsSearchParams(fts="Состав релиза 26.3.1: какие сервисы"))
            == expected
        )

    def test_plain_code_with_braces_is_split(self) -> None:
        """Lowercase "allowed-actions" is not a pattern token — no exact branch."""
        expected = "((title:((application AND id AND allowed AND actions)))^3 "
        expected += "OR text:((application AND id AND allowed AND actions)))"
        assert (
            build_fts_query(FtsSearchParams(fts="/application/{id}/allowed-actions"))
            == expected
        )

    def test_version_like_token_expanded_to_and_parts(self) -> None:
        """The index tokenizer splits dotted strings — a whole raw term never
        matches; a PATTERNS-whitelisted identifier additionally gets a boosted
        pattern branch (the release.2026.2.3.964 token stops at "-alt")."""
        expected = "((title:((release AND 2026 AND 3 AND 1 AND 1028)))^3 "
        expected += "OR text:((release AND 2026 AND 3 AND 1 AND 1028)) "
        expected += 'OR pattern:("release.2026.3.1.1028"^3))'
        assert (
            build_fts_query(FtsSearchParams(fts="release.2026.3.1.1028")) == expected
        )
        expected = "((title:((release AND 2026 AND 2 AND 3 AND 964 AND alt)))^3 "
        expected += "OR text:((release AND 2026 AND 2 AND 3 AND 964 AND alt)) "
        expected += 'OR pattern:("release.2026.2.3.964"^3))'
        assert (
            build_fts_query(FtsSearchParams(fts="release.2026.2.3.964-alt")) == expected
        )
        expected = "((title:((PROJ AND 1 AND 2 AND 3)))^3 OR text:((PROJ AND 1 AND 2 AND 3)) "
        expected += 'OR pattern:("PROJ-1"^3))'
        assert build_fts_query(FtsSearchParams(fts="PROJ-1.2.3")) == expected

    def test_version_like_token_among_other_terms(self) -> None:
        expected = "((title:(api OR (release AND 2026 AND 3 AND 1)))^3 "
        expected += "OR text:(api OR (release AND 2026 AND 3 AND 1)) "
        expected += 'OR pattern:("release.2026.3.1"^3))'
        assert build_fts_query(FtsSearchParams(fts="api release.2026.3.1")) == expected

    def test_dotted_two_parts_also_expanded(self) -> None:
        """Lowercase "v2.0" matches no PATTERNS family (no exact branch), but
        the dotted token is still decomposed into ANDed word parts."""
        expected = "((title:((v2 AND 0)))^3 OR text:((v2 AND 0)))"
        assert build_fts_query(FtsSearchParams(fts="v2.0")) == expected

    def test_underscore_and_hyphen_identifiers_are_split(self) -> None:
        """Neither identifier matches a PATTERNS family (no dot+number, no
        hyphen+digit) — decomposed into stemmed word parts, no exact branch."""
        expected = "((title:((DEMO AND SUP AND 2026)))^3 OR text:((DEMO AND SUP AND 2026)))"
        assert build_fts_query(FtsSearchParams(fts="DEMO_SUP_2026")) == expected
        expected = "((title:((get AND mo AND async)))^3 OR text:((get AND mo AND async)))"
        assert build_fts_query(FtsSearchParams(fts="get-mo-async")) == expected

    def test_reserved_word_part_is_quoted(self) -> None:
        """No trailing number — not a case-code pattern token, no exact branch."""
        expected = '((title:((FOO AND "NOT" AND VALID)))^3 OR text:((FOO AND "NOT" AND VALID)))'
        assert build_fts_query(FtsSearchParams(fts="FOO.NOT.VALID")) == expected

    def test_exact_code_gets_unique_pattern_branch(self) -> None:
        """The whole identifier is one exact pattern token, distinct from
        related codes, so ABC.DEF.01 cannot be diluted by C_ABC.DEF.01 etc."""
        q = build_fts_query(FtsSearchParams(fts="ABC.DEF.01"))
        assert 'pattern:("ABC.DEF.01"^3)' in q
        from atlassian_rag_mcp.storage.fts.normalize import extract_pattern_tokens
        assert extract_pattern_tokens("ABC.DEF.01") != extract_pattern_tokens("C_ABC.DEF.01")
        assert extract_pattern_tokens("ABC.DEF.01") != extract_pattern_tokens("ABC.DEF_GHI.01")
        assert extract_pattern_tokens("ABC.DEF.01") != extract_pattern_tokens("ABC.DEF.02")

    def test_spaces_embedded_as_clause(self) -> None:
        assert (
            build_fts_query(FtsSearchParams(fts="api", spaces=["DEV"]))
            == "((title:(api))^3 OR text:(api)) AND space:DEV"
        )
        assert (
            build_fts_query(FtsSearchParams(fts="api", spaces=["DEV", "OPS"]))
            == "((title:(api))^3 OR text:(api)) AND (space:DEV OR space:OPS)"
        )

    def test_page_ids_embedded_as_clause(self) -> None:
        assert (
            build_fts_query(FtsSearchParams(fts="api", page_ids=["p1", "p2"]))
            == "((title:(api))^3 OR text:(api)) AND (page:p1 OR page:p2)"
        )

    def test_spaces_and_page_ids_combined(self) -> None:
        assert (
            build_fts_query(FtsSearchParams(fts="api", spaces=["DEV"], page_ids=["p1"]))
            == "((title:(api))^3 OR text:(api)) AND space:DEV AND page:p1"
        )

    def test_spaces_only_filters(self) -> None:
        assert build_fts_query(FtsSearchParams(spaces=["DEV"])) == "space:DEV"
        assert build_fts_query(FtsSearchParams(page_ids=["p1", "p2"])) == "(page:p1 OR page:p2)"

    def test_empty_params_without_filters_return_empty(self) -> None:
        assert build_fts_query(FtsSearchParams(fts="  ", spaces=[])) == ""
        assert build_fts_query(FtsSearchParams(spaces=None, page_ids=None)) == ""


class TestPerTermBoost:
    def test_pattern_code_boost_is_pattern_branch_only(self) -> None:
        q = build_fts_query(FtsSearchParams(fts="ABC.DEF.01^4"))
        assert q == "(pattern:(ABC.DEF.01)^4)"

    def test_non_pattern_boost_falls_back_to_title_branch(self) -> None:
        q = build_fts_query(FtsSearchParams(fts="тест^2"))
        assert q == "((title:(тест))^2)"

    def test_mixed_plain_and_boosted_terms(self) -> None:
        q = build_fts_query(
            FtsSearchParams(fts="онбординг ABC.DEF.01^4 тест^2")
        )
        assert q == (
            "((title:(онбординг))^3 OR text:(онбординг)"
            " OR pattern:(ABC.DEF.01)^4 OR (title:(тест))^2)"
        )

    def test_boost_clamped_to_10(self) -> None:
        assert (
            build_fts_query(FtsSearchParams(fts="ABC.DEF.01^99"))
            == "(pattern:(ABC.DEF.01)^10)"
        )
        assert (
            build_fts_query(FtsSearchParams(fts="тест^99"))
            == "((title:(тест))^10)"
        )

    def test_zero_boost_clamped_to_1(self) -> None:
        assert build_fts_query(FtsSearchParams(fts="тест^0")) == "((title:(тест))^1)"

    def test_non_numeric_suffix_searched_as_text(self) -> None:
        q = build_fts_query(FtsSearchParams(fts="тест^abc"))
        assert q == "((title:((тест AND abc)))^3 OR text:((тест AND abc)))"

    def test_mid_token_caret_is_raw_syntax(self) -> None:
        assert build_fts_query(FtsSearchParams(fts="a^2b")) == "a^2b"

    def test_bare_boost_suffix_yields_empty(self) -> None:
        assert build_fts_query(FtsSearchParams(fts="^4")) == ""


class TestStrictValidation:
    @pytest.mark.parametrize(
        "query",
        [
            "titl:abc",
            "chunk_id:c1",
            "foo:barBaz",
            "text2:x",
            "http://example.com",
            "text:",
            "text:   ",
            "page: ",
            "text:(abc",
            "text:(a AND b))",
            '"unbalanced',
            'text:"a b" c"d',
        ],
    )
    def test_invalid_raises(self, query: str) -> None:
        with pytest.raises(FtsError):
            validate_fts_query(query)

    @pytest.mark.parametrize(
        "query",
        [
            "text:(a AND b)",
            "title:x OR text:x",
            '"phrase"~2',
            "space:PROJ page:123",
            "pattern:ABC.DEF.01^4",
            "te*t",
            "тест^2 ABC.DEF.01^4",
            "тема: развёртывание",
            'text:"a:b"',
        ],
    )
    def test_valid_passes(self, query: str) -> None:
        validate_fts_query(query)

    def test_unknown_field_error_names_allowed_fields(self) -> None:
        with pytest.raises(FtsError, match="unknown field 'chunk_id'"):
            validate_fts_query("chunk_id:c1")
        with pytest.raises(FtsError, match="allowed fields: text, title, pattern, page, space"):
            validate_fts_query("chunk_id:c1")

    def test_build_fts_query_raises_on_invalid(self) -> None:
        with pytest.raises(FtsError):
            build_fts_query(FtsSearchParams(fts="chunk_id:c1"))
        with pytest.raises(FtsError):
            build_fts_query(FtsSearchParams(fts="text:"))
        with pytest.raises(FtsError):
            build_fts_query(FtsSearchParams(fts='"unclosed'))

    def test_build_fts_query_plain_cyrillic_colon_stays_plain(self) -> None:
        q = build_fts_query(FtsSearchParams(fts="тема: развёртывание"))
        assert "text:" in q
        assert "title:" in q


class TestFtsSearchParams:
    def test_only_fts_fields(self) -> None:
        p = FtsSearchParams()
        assert p.fts is None
        assert p.spaces is None
        assert p.page_ids is None
        assert not hasattr(p, "vector")

    def test_numeric_coerced_to_text(self) -> None:
        assert FtsSearchParams(fts=42).fts == "42"

    def test_scalar_list_coerced(self) -> None:
        assert FtsSearchParams(page_ids="p1").page_ids == ["p1"]

    def test_foreign_fields_rejected(self) -> None:
        with pytest.raises(Exception):
            FtsSearchParams(**{"fts": "q", "vector": "v", "page_version": 3})

    def test_required_fts_query(self) -> None:
        p = FtsSearchParams(fts="api")
        assert p.fts == "api"

    def test_fulltext_requires_fts_query(self) -> None:
        with pytest.raises(Exception):
            FtsSearchParams(vector="api")

    def test_bool_rejected(self) -> None:
        with pytest.raises(Exception):
            FtsSearchParams(fts=True)

    def test_nested_list_rejected(self) -> None:
        with pytest.raises(Exception):
            FtsSearchParams(page_ids=[[1]])


class TestFtsBuilder:
    def test_terms_or(self) -> None:
        assert FtsBuilder().terms(["a", "b"]).build() == "text:(a OR b)"

    def test_terms_and(self) -> None:
        assert FtsBuilder().terms(["a", "b"], conjunction=True).build() == "text:(a AND b)"

    def test_chain(self) -> None:
        q = FtsBuilder().terms(["api", "docs"]).phrase("exact phrase").space_keys("DEV").build()
        assert q == 'text:(api OR docs) AND text:"exact phrase" AND space:DEV'

    def test_empty_clauses_dropped(self) -> None:
        assert FtsBuilder().terms([]).build() == ""
        assert FtsBuilder().page_id("  ").terms(["ok"]).build() == "text:ok"

    def test_keyword_with_quotes_is_sanitized(self) -> None:
        assert FtsBuilder().terms(['say "hi"']).build() == 'text:"say \\"hi\\""'
