"""Tests for storage/fts/normalize — FTS normalization and pattern tokens."""

from __future__ import annotations

import re

from atlassian_rag_mcp.storage.fts.normalize import (
    PATTERN_RE,
    PATTERNS,
    extract_pattern_tokens,
    normalize_index_text,
    stem_query_string,
    stem_token,
)


class TestStemToken:
    def test_cyrillic_stemmed(self):
        assert stem_token("модели") == "модел"

    def test_latin_passthrough(self):
        assert stem_token("ABC") == "ABC"

    def test_short_cyrillic_passthrough(self):
        assert stem_token("ме") == "ме"


class TestNormalizeIndexText:
    def test_stems_cyrillic_keeps_codes(self):
        out = normalize_index_text("Модель данных ABC.DEF.01")
        assert "модел" in out.split()
        assert "ABC" in out.split()


class TestStemQueryString:
    def test_stems_cyrillic_runs(self):
        assert stem_query_string("text:модели") == "text:модел"


class TestPatterns:
    def test_all_patterns_are_individual_regexes(self):
        """Each family must be a usable, linear regex on its own."""
        for p in PATTERNS:
            re.compile(p)
        assert len(PATTERNS) >= 4

    def test_case_codes(self):
        assert extract_pattern_tokens("10 ABC.DEF.01 УТОЧНИТЬ") == ["ABC.DEF.01"]
        out = extract_pattern_tokens("C_ABC.DEF_GHI.01 и A_B.CD_EF.01")
        assert out == ["C_ABC.DEF_GHI.01", "A_B.CD_EF.01"]

    def test_jira_ids(self):
        out = extract_pattern_tokens("см. PROJ-1002 и PROJ-1003, PROJ-1004")
        assert out == ["PROJ-1002", "PROJ-1003", "PROJ-1004"]

    def test_short_codes(self):
        out = extract_pattern_tokens("файлы A-CODE и B-TYPE")
        assert out == ["A-CODE", "B-TYPE"]

    def test_release_versions(self):
        out = extract_pattern_tokens("релиз release.2026.3.1.1028 готов")
        assert out == ["release.2026.3.1.1028"]


class TestExtractPatternTokens:
    def test_multiple_preserve_order_and_duplicates(self):
        out = extract_pattern_tokens("ABC.DEF.01 и ABC.DEF.02, снова ABC.DEF.01")
        assert out == ["ABC.DEF.01", "ABC.DEF.02", "ABC.DEF.01"]

    def test_trailing_punctuation_not_swallowed(self):
        """A sentence-final dot / list separator must not extend the code."""
        assert extract_pattern_tokens("Отключены: ABC.DEF.01. Добавлены ABC.DEF.02;") == [
            "ABC.DEF.01",
            "ABC.DEF.02",
        ]

    def test_related_codes_are_distinct_terms(self):
        out = set(extract_pattern_tokens(
            "ABC.DEF.01 C_ABC.DEF.01 ABC.DEF_GHI.01 ABC.DEF.02"
        ))
        assert out == {"ABC.DEF.01", "C_ABC.DEF.01", "ABC.DEF_GHI.01", "ABC.DEF.02"}

    def test_noise_is_not_extracted(self):
        text = (
            "OBJECT_IDENTIFIER APP_VERSION BIT_STRING 23.01.2023 1.18.0 v2.0 "
            "GET_INSURED_GROUPS_FOR_APPORTIONMENT_SMO SOAP-ENV"
        )
        assert extract_pattern_tokens(text) == []

    def test_mixed(self):
        out = extract_pattern_tokens("C_ABC.DEF_GHI.01 release.2026.3.1.1028 PROJ-1002 v2.0")
        assert out == ["C_ABC.DEF_GHI.01", "release.2026.3.1.1028", "PROJ-1002"]

    def test_plain_words_are_not_tokens(self):
        assert extract_pattern_tokens("модель данных api docs") == []

    def test_empty(self):
        assert extract_pattern_tokens("") == []

    def test_long_base64_blob_is_fast(self):
        """Confluence macro payloads embed base64-encoded HTML (long runs of
        alphanumerics); the pattern must stay linear (no catastrophic
        backtracking): a 250 KB blob extracts in well under a second."""
        import time

        blob = ("JTNDdGFibGUlMjBjbGFzcyUzRCUyMndyYXBwZWQlMjBjb25mbHVlbmNlVGFibGUlMjIlM0U" * 2000)
        t0 = time.time()
        assert extract_pattern_tokens(blob) == []
        assert time.time() - t0 < 1.0

    def test_query_side_tokens_are_plain_strings(self):
        """No U+0300 encoding — the query-side token equals the raw
        identifier, exactly what the field's regex tokenizer emits (Rust side)."""
        assert PATTERN_RE.findall("ABC.DEF.01") == ["ABC.DEF.01"]
