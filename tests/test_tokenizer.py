"""Tests for utils/tokenizer.py — bundled Qwen3 tokenizer and token split."""

from __future__ import annotations

from atlassian_rag_mcp.utils.tokenizer import (
    count_qwen3_tokens,
    get_qwen3_tokenizer,
    split_overlap_tokens,
)


def _share_aligned_overlap(a: str, b: str) -> bool:
    """True when a's some suffix equals b's prefix on character boundaries."""
    upper = min(len(a), len(b))
    return any(a[-k:] == b[:k] for k in range(2, upper + 1))


class TestBundledTokenizer:
    def test_loads_from_project_file(self):
        tk = get_qwen3_tokenizer()
        assert tk is not None

    def test_cached_after_first_load(self):
        assert get_qwen3_tokenizer() is get_qwen3_tokenizer()

    def test_count_cyrillic(self):
        assert count_qwen3_tokens("Привет мир") == 5

    def test_count_digits(self):
        assert count_qwen3_tokens("12345") == 6


class TestSplitOverlapTokens:
    """split_overlap_tokens with a deterministic fake counter (1 token =
    1 char for ASCII, ~3 bytes = 1 token for Cyrillic)."""

    def test_no_split_when_within_budget(self):
        text = "short text"
        result = split_overlap_tokens(text, len, max_tokens=6000, overlap_tokens=100)
        assert result == [text]

    def test_split_long_ascii(self):
        text = " ".join(["word"] * 2000)
        result = split_overlap_tokens(text, len, max_tokens=100, overlap_tokens=20)
        assert len(result) > 1
        for seg in result:
            assert len(seg) <= 100

    def test_split_cyrillic_by_token_estimate(self):
        def counter(t: str) -> int:
            return len(t.encode("utf-8")) // 3

        text = "абвгдеёж " * 2000
        result = split_overlap_tokens(text, counter, max_tokens=200, overlap_tokens=20)
        assert len(result) > 1
        for seg in result:
            assert counter(seg) <= 200

    def test_overlap_exists(self):
        text = "alpha bravo charlie delta " * 500
        result = split_overlap_tokens(text, len, max_tokens=100, overlap_tokens=15)
        assert len(result) >= 2
        for i in range(len(result) - 1):
            assert _share_aligned_overlap(result[i], result[i + 1]), (
                f"Segments {i} and {i+1} do not share expected overlap"
            )

    def test_unbreakable_unit_cuts_by_char_count(self):
        """Every "token" weighs 1000 > budget 500: no split can help, so
        segments are cut at max_tokens characters and progress is made."""
        def counter(t: str) -> int:
            return 1000 * len(t)

        text = "a" * 1000
        result = split_overlap_tokens(text, counter, max_tokens=500, overlap_tokens=0)
        assert result == [text[:500], text[500:]]
