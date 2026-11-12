"""Chunking pipeline: synthetic HTML → chunks — no mocks needed.

A single unified size limit (`max_tokens`, default 8000 tokens from
CHUNK_MAX_TOKENS, measured by the bundled Qwen3 tokenizer) applies
to the FINAL chunk text for BOTH tables and sections: for sections it is
the heading prefix + plain text, for tables it is the raw HTML markup only
(no heading prefix in table text; heading context is carried in the
heading_hierarchy metadata):
Tables:
  - no text -> skipped
  - final HTML size <= max_tokens -> whole table in one chunk
    (chunk_type='table', block_part_id=None)
  - final HTML size > max_tokens -> one chunk per <tr>
    (chunk_type='table_row', block_part_id=1..N)
Sections:
  - final text size <= max_tokens -> one 'section' chunk
    (chunk_type='section', block_part_id=None)
  - final text size > max_tokens -> one 'section_part' chunk per
    paragraph (chunk_type='section_part', block_part_id=1..N)
All chunks of a logical block share a common `block_index` (1-based,
monotonically increasing across tables AND sections on the page).

Tests pin the chunker's counter to the UTF-8 byte length (a deterministic
stand-in for the tokenizer) so size expectations are exact and the real
tokenizer file is not needed; the explicit `counter=` param covers the
real-tokenizer path.
"""

from __future__ import annotations

import sys

import pytest

from atlassian_rag_mcp.confluence.chunker.html_chunker import HtmlChunker
from atlassian_rag_mcp.confluence.chunker.pipeline import chunking_pipeline


def _byte_len(t: str) -> int:
    return len(t.encode("utf-8"))


@pytest.fixture(autouse=True)
def _pin_chunk_counter():
    """Deterministic counter: UTF-8 byte length (stand-in for the tokenizer)."""
    import atlassian_rag_mcp.confluence.chunker.html_chunker as hc

    saved = (hc._chunk_counter, hc._counter_resolved)
    hc._chunk_counter = _byte_len
    hc._counter_resolved = True
    yield
    hc._chunk_counter, hc._counter_resolved = saved


def _valid_html(body: str) -> str:
    return f"<html><body>{body}</body></html>"


def _small_table() -> str:
    return (
        "<table><tr><th>Name</th><th>Value</th></tr>"
        "<tr><td>Alpha</td><td>One</td></tr>"
        "</table>"
    )


class TestParagraphChunks:
    def test_single_paragraph(self):
        result = chunking_pipeline(
            page_id="42",
            page_title="Test Page",
            html=_valid_html("<p>Content that is long enough to pass the meaningful text check.</p>"),
        )
        assert len(result) == 1
        assert result[0].chunk_type == "section"
        assert result[0].page_id == "42"
        assert result[0].page_title == "Test Page"
        assert result[0].block_index == 1
        assert result[0].block_part_id is None

    def test_empty_html(self):
        result = chunking_pipeline(
            page_id="1", page_title="Empty",
            html=_valid_html(""),
        )
        assert result == []

    def test_only_whitespace(self):
        result = chunking_pipeline(
            page_id="1", page_title="WS",
            html=_valid_html("   \n\n  "),
        )
        assert result == []

    def test_comment_only_html(self):
        """lxml raises ParserError ("Document is empty") on comment-only docs
        — the pipeline must treat it as an empty page, not crash."""
        result = chunking_pipeline(
            page_id="1", page_title="CO",
            html="<!-- only a comment -->",
        )
        assert result == []


class TestHeadingChunks:
    def test_heading_then_paragraph(self):
        html = _valid_html("<h1>Section A</h1><p>Content under the first heading.</p>")
        result = chunking_pipeline(
            page_id="1", page_title="P", html=html,
        )
        assert len(result) == 1
        assert result[0].heading_hierarchy == ["Section A"]
        text = result[0].text
        assert "Section A" in text
        assert "Content under the first heading" in text

    def test_multi_level_headings(self):
        html = _valid_html(
            "<h1>Level One Title</h1><p>Paragraph one content here.</p>"
            "<h2>Level Two Title</h2><p>Paragraph two content here.</p>"
            "<h3>Level Three Title</h3><p>Paragraph three content here.</p>"
        )
        result = chunking_pipeline(
            page_id="1", page_title="P", html=html,
        )
        assert len(result) == 3
        assert result[0].heading_hierarchy == ["Level One Title"]
        assert result[1].heading_hierarchy == ["Level One Title", "Level Two Title"]
        assert result[2].heading_hierarchy == ["Level One Title", "Level Two Title", "Level Three Title"]


class TestTableChunks:
    """Small tables (final text within the byte budget) -> whole table in one 'table' chunk."""

    def test_small_table_single_chunk(self):
        result = chunking_pipeline(
            page_id="1", page_title="P", html=_valid_html(_small_table()),
        )
        assert len(result) == 1
        c = result[0]
        assert c.chunk_type == "table"
        assert c.block_index == 1
        assert c.block_part_id is None
        assert "Name" in c.text and "Alpha" in c.text

    def test_whole_table_chunk_is_cleaned_html(self):
        html = _valid_html(
            '<table class="relative-table wrapped" style="width: 50%">'
            "<tr><th class=\"confluenceTh\">Name</th>"
            "<th><div class=\"content-wrapper\"><p><span>x</span></p></div></th></tr>"
            "<tr><td>Alpha</td><td>One</td></tr>"
            "</table>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        c = result[0]
        assert c.chunk_type == "table"
        assert "<table>" in c.text
        assert "<tr>" in c.text
        assert "class=" not in c.text
        assert "style=" not in c.text

    def test_small_table_chunk_heading_in_metadata_only(self):
        html = _valid_html(f"<h2>Registry Section</h2>{_small_table()}")
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        c = result[0]
        assert c.chunk_type == "table"
        assert c.heading_hierarchy == ["Registry Section"]
        assert not c.text.startswith("## Registry Section")
        assert c.text.lstrip().startswith("<table>")

    def test_empty_table_no_chunks(self):
        html = _valid_html(
            "<table><tr><td></td><td> </td></tr>"
            "<tr><td>\xa0</td><td></td></tr></table>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert result == []

    def test_single_row_table(self):
        html = _valid_html("<table><tr><td>Only one row with enough text.</td></tr></table>")
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        assert result[0].chunk_type == "table"
        assert result[0].block_part_id is None


class TestTableRowwise:
    """Large tables: one chunk per row, each emitted as a <tr>."""

    @staticmethod
    def _big_table(n_rows: int = 320) -> str:
        rows = "".join(
            f"<tr><td>Row number {i} column one value</td>"
            f"<td>Row number {i} column two value</td></tr>"
            for i in range(n_rows)
        )
        return f"<table><tr><th>Name</th><th>Value</th></tr>{rows}</table>"

    def test_large_table_rowwise(self):
        result = chunking_pipeline(
            page_id="1", page_title="P", html=_valid_html(self._big_table()),
        )
        assert len(result) == 321
        for c in result:
            assert c.chunk_type == "table_row"
            assert c.block_index == 1
        assert [c.block_part_id for c in result] == list(range(1, 322))

    def test_first_row_not_special(self):
        result = chunking_pipeline(
            page_id="1", page_title="P", html=_valid_html(self._big_table()),
        )
        first = result[0]
        assert first.chunk_type == "table_row"
        assert first.block_part_id == 1
        assert "Name" in first.text and "Value" in first.text
        assert "{" not in first.text

    def test_each_row_is_tr(self):
        result = chunking_pipeline(
            page_id="1", page_title="P", html=_valid_html(self._big_table()),
        )
        for c in result:
            assert c.text.count("<tr>") == 1
            assert c.text.count("<table>") == 0

    def test_thead_large_rowwise(self):
        rows = "".join(
            f"<tr><td>Data {i} alpha</td><td>Data {i} beta</td></tr>" for i in range(450)
        )
        html = (
            "<table><thead><tr><th>Head A</th><th>Head B</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=_valid_html(html))
        assert len(result) == 451
        assert result[0].block_part_id == 1
        assert "Head A" in result[0].text

    def test_thead_small_single_chunk(self):
        html = (
            '<table><thead><tr><th>Head A</th><th>Head B</th></tr></thead>'
            "<tbody><tr><td>Cell A1</td><td>Cell B1</td></tr></tbody></table>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=_valid_html(html))
        assert len(result) == 1
        assert result[0].chunk_type == "table"
        assert "Head A" in result[0].text and "Cell A1" in result[0].text

    def test_empty_rows_skipped_rowwise(self):
        rows = "".join(
            f"<tr><td>Filled row number {i} here</td><td>More text value</td></tr>"
            if i % 100 != 50
            else "<tr><td></td><td> </td></tr>"
            for i in range(400)
        )
        html = _valid_html(f"<table>{rows}</table>")
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 396
        part_ids = [c.block_part_id for c in result]
        assert 51 not in part_ids and 151 not in part_ids
        assert max(part_ids) == 400
        for c in result:
            assert "Filled" in c.text


class TestTableSizeThreshold:
    """Tables split only when the final chunk text exceeds max_tokens."""

    @staticmethod
    def _row(cells_text: str) -> str:
        return f"<tr><td>{cells_text}</td></tr>"

    def test_table_under_400_is_single_chunk(self):
        text = "Word " * 60
        html = _valid_html(f"<table>{self._row(text)}</table>")
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        assert result[0].chunk_type == "table"
        assert result[0].block_part_id is None

    def test_table_at_400_is_rowwise(self):
        text = "A" * 405
        html = _valid_html(f"<table>{self._row(text)}</table>")
        result = chunking_pipeline(page_id="1", page_title="P", html=html, max_tokens=400)
        assert len(result) == 1
        assert result[0].chunk_type == "table_row"
        assert result[0].block_part_id == 1

    def test_multi_row_under_400_is_single_chunk(self):
        rows = self._row("Short row alpha here.") + self._row("Short row beta here.")
        html = _valid_html(f"<table>{rows}</table>")
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        assert result[0].chunk_type == "table"
        assert "Short row alpha" in result[0].text
        assert "Short row beta" in result[0].text

    def test_multi_row_over_400_is_rowwise(self):
        rows = "".join(self._row(f"Longer row content number {i} with extra words.") for i in range(20))
        html = _valid_html(f"<table>{rows}</table>")
        result = chunking_pipeline(page_id="1", page_title="P", html=html, max_tokens=400)
        assert len(result) == 20
        assert all(c.chunk_type == "table_row" for c in result)
        assert [c.block_part_id for c in result] == list(range(1, 21))


class TestSectionSplit:
    """Headingless content over the size limit → one 'section_part' per paragraph."""

    def test_headingless_long_page_splits_by_paragraph(self):
        paras = "<p>Paragraph text number {} that is long enough.</p>"
        html = _valid_html("".join(paras.format(i) for i in range(150)))
        chunks = chunking_pipeline(page_id="1", page_title="P", html=html, max_tokens=6000)
        assert len(chunks) == 150
        assert all(c.chunk_type == "section_part" for c in chunks)
        assert all(c.block_part_id == i + 1 for i, c in enumerate(chunks))
        assert all(c.block_index == chunks[0].block_index for c in chunks)
        for i in range(150):
            assert f"Paragraph text number {i}" in chunks[i].text

    def test_single_huge_paragraph_single_part(self):
        sentence = "This is a sentence that is long enough to matter. "
        big = sentence * 500
        chunks = chunking_pipeline(
            page_id="1", page_title="P", html=_valid_html(f"<p>{big}</p>"),
        )
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "section_part"
        assert chunks[0].block_part_id == 1
        all_text = chunks[0].text
        assert all_text.count("This is a sentence") == 500

    def test_multiple_paragraphs_split_by_paragraph(self):
        chunks = chunking_pipeline(
            page_id="1", page_title="P",
            html=_valid_html("".join(f"<p>{'word ' * 300}</p>" for _ in range(5))),
            max_tokens=6000,
        )
        assert len(chunks) == 5
        assert all(c.chunk_type == "section_part" for c in chunks)
        assert [c.block_part_id for c in chunks] == [1, 2, 3, 4, 5]


class TestByteLengthBudget:
    """With the pinned counter (UTF-8 byte length) the limit applies to
    the FINAL chunk text in bytes (heading prefix + plain text or raw HTML)
    — the single pipeline size budget shared with the vector layer.
    Regression tests for the old CHUNK_MAX_CHARS (chars of cleaned text) /
    bytes mismatch."""

    def test_cyrillic_table_under_old_char_limit_over_byte_limit_rowwise(self):
        """Cleaned text (1760 chars) passes the old 2000-char limit, but the
        final HTML (~6.5 KB) exceeds the 6000-byte budget — split per row."""
        rows = "".join(
            "<tr><td>абвгдежз</td><td>абвгдежз</td></tr>" for _ in range(110)
        )
        html = _valid_html(f"<table>{rows}</table>")
        assert 110 * 2 * 8 < 2000
        assert len(f"<table>{rows}</table>".encode("utf-8")) > 6000
        result = chunking_pipeline(page_id="1", page_title="P", html=html, max_tokens=6000)
        assert len(result) == 110
        assert all(c.chunk_type == "table_row" for c in result)
        assert [c.block_part_id for c in result] == list(range(1, 111))
        for c in result:
            assert len(c.text.encode("utf-8")) <= 6000

    def test_cyrillic_section_over_byte_limit_splits_by_paragraph(self):
        para = "абвгдежзийклмнопрстуфхцчшщыэюя " * 80
        html = _valid_html(f"<p>{para}</p><p>{para}</p>")
        chunks = chunking_pipeline(page_id="1", page_title="P", html=html, max_tokens=6000)
        assert len(chunks) == 2
        assert all(c.chunk_type == "section_part" for c in chunks)
        assert [c.block_part_id for c in chunks] == [1, 2]
        for c in chunks:
            assert len(c.text.encode("utf-8")) <= 6000

    def test_section_at_exact_byte_limit_is_single_chunk(self):
        html = _valid_html(f"<p>{'a' * 6000}</p>")
        chunks = chunking_pipeline(page_id="1", page_title="P", html=html, max_tokens=6000)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "section"
        assert len(chunks[0].text.encode("utf-8")) == 6000

    def test_section_one_byte_over_limit_splits(self):
        html = _valid_html(f"<p>{'a' * 6001}</p>")
        chunks = chunking_pipeline(page_id="1", page_title="P", html=html, max_tokens=6000)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "section_part"
        assert chunks[0].block_part_id == 1


class TestTokenCounter:
    """A (real-tokenizer-style) counter measuring tokens, not bytes: text
    under the token budget stays whole even when its byte count passes the
    same numeric budget (Cyrillic ~3 bytes/token). The counter is pinned on
    the chunker module (the autouse fixture restores it)."""

    @staticmethod
    def _pin_counter(counter):
        import atlassian_rag_mcp.confluence.chunker.html_chunker as hc

        hc._chunk_counter = counter
        hc._counter_resolved = True

    def test_cyrillic_over_byte_count_under_token_budget_stays_whole(self):
        para = "абвгдежзийклмнопрстуфхцчшщыэюя " * 150
        html = _valid_html(f"<p>{para}</p>")
        assert len(para.encode("utf-8")) > 8000

        def fake_counter(t: str) -> int:
            return len(t.encode("utf-8")) // 3

        self._pin_counter(fake_counter)
        chunks = HtmlChunker().process("1", "P", html, max_tokens=8000)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "section"
        assert fake_counter(chunks[0].text) <= 8000

    def test_dense_counter_single_paragraph_stays_one_chunk(self):
        """~1 token/byte: the token count (15000) overflows the 8000 budget,
        but a single paragraph cannot be split at paragraph boundaries — the
        chunker emits ONE oversized section_part and the vector layer's
        safety net re-splits it before embedding (see
        test_dense_chunk_still_split_despite_tokenizer)."""
        para = "1 2 3 4 5 6 7 8 9 0 " * 1500
        html = _valid_html(f"<p>{para}</p>")

        def fake_counter(t: str) -> int:
            return len(t.encode("utf-8"))

        self._pin_counter(fake_counter)
        chunks = HtmlChunker().process("1", "P", html, max_tokens=8000)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "section_part"
        assert fake_counter(chunks[0].text) > 8000


class TestSectionNoSplitUnderLimit:
    """Headingless short content stays a single 'section' chunk."""

    def test_small_section_single_chunk(self):
        html = _valid_html("<p>Short paragraph well under the limit.</p>")
        chunks = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "section"
        assert chunks[0].block_part_id is None
        assert chunks[0].block_index == 1

    def test_headinged_short_section_single_chunk(self):
        html = _valid_html("<h2>Intro</h2><p>Another short paragraph here.</p>")
        chunks = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "section"
        assert chunks[0].block_part_id is None
        assert "Intro" in chunks[0].text


class TestUniformity:
    """No special-casing: single-column and nested tables follow the same rules."""

    def test_single_column_small_single_chunk(self):
        html = _valid_html(
            "<table><tr><td>Single column row one</td></tr>"
            "<tr><td>Single column row two</td></tr></table>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        assert result[0].chunk_type == "table"
        assert "Single column row one" in result[0].text
        assert "Single column row two" in result[0].text

    def test_single_column_large_rowwise(self):
        rows = "".join(
            f"<tr><td>Long single column row number {i} with extra words</td></tr>"
            for i in range(650)
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=_valid_html(f"<table>{rows}</table>"))
        assert all(c.chunk_type == "table_row" for c in result)
        assert len(result) == 650

    def test_nested_table_small_single_chunk(self):
        html = _valid_html(
            "<table><tr><td>Outer cell"
            "<table><tr><td>Inner cell value</td></tr></table>"
            "</td></tr></table>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        assert result[0].chunk_type == "table"
        assert result[0].text.count("<table>") == 2

    def test_nested_table_large_outer_rowwise(self):
        inner = "<table><tr><td>Inner cell value</td></tr></table>"
        rows = "".join(f"<tr><td>Outer row number {i} {inner}</td></tr>" for i in range(400))
        result = chunking_pipeline(page_id="1", page_title="P", html=_valid_html(f"<table>{rows}</table>"))
        assert all(c.chunk_type == "table_row" for c in result)
        assert len(result) == 400
        for c in result:
            assert c.text.count("<table>") == 1


class TestMetadata:
    def test_block_index_increments_across_tables_and_sections(self):
        html = _valid_html(
            "<p>Introductory text paragraph here.</p>"
            "<table><tr><td>First table cell content.</td></tr></table>"
            "<p>Middle text paragraph content.</p>"
            "<table><tr><td>Second table cell content.</td></tr></table>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        tables = [c for c in result if c.chunk_type == "table"]
        assert [c.block_index for c in tables] == [2, 4]
        assert [c.chunk_type for c in tables] == ["table", "table"]
        assert all(c.block_part_id is None for c in tables)

    def test_chunk_index_sequential_across_types(self):
        html = _valid_html(
            "<p>First paragraph content here.</p>"
            "<table><tr><td>Table cell content here.</td></tr></table>"
            "<p>Second paragraph content here.</p>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert [c.chunk_index for c in result] == [0, 1, 2]

    def test_page_version_and_updated_at(self):
        html = _valid_html("<p>Long enough content for version test.</p>")
        result = chunking_pipeline(
            page_id="1", page_title="P", html=html,
            page_version=3, updated_at="2024-06-15T10:00:00Z",
        )
        assert result[0].page_version == 3
        assert result[0].updated_at == "2024-06-15T10:00:00Z"

    def test_parent_id_passthrough(self):
        html = _valid_html("<p>Long enough content for parent test.</p>")
        result = chunking_pipeline(
            page_id="1", page_title="P", html=html,
            parent_id="99",
        )
        assert result[0].parent_id == "99"


class TestListChunks:
    def test_unordered_list(self):
        html = _valid_html("<ul><li>Item A content here.</li><li>Item B content here.</li></ul>")
        result = chunking_pipeline(
            page_id="1", page_title="P", html=html,
        )
        assert len(result) == 1
        text = result[0].text
        assert "Item A content here" in text
        assert "Item B content here" in text

    def test_ordered_list(self):
        html = _valid_html("<ol><li>First item content.</li><li>Second item content.</li></ol>")
        result = chunking_pipeline(
            page_id="1", page_title="P", html=html,
        )
        assert len(result) >= 1
        text = result[0].text
        assert "First item content" in text
        assert "Second item content" in text


class TestNoPandas:
    def test_pandas_not_loaded_by_pipeline(self):
        saved = sys.modules.pop("pandas", None)
        try:
            chunking_pipeline(
                page_id="1", page_title="P",
                html=_valid_html(self._big()),
            )
            assert "pandas" not in sys.modules
        finally:
            if saved is not None:
                sys.modules["pandas"] = saved

    @staticmethod
    def _big() -> str:
        rows = "".join(f"<tr><td>Row number {i} with enough text.</td></tr>" for i in range(320))
        return f"<table><tr><th>Name</th></tr>{rows}</table>"

    def test_no_pandas_import_in_source(self):
        import atlassian_rag_mcp.confluence.chunker.html_chunker as mod
        src = open(mod.__file__, encoding="utf-8").read()
        assert "import pandas" not in src
        assert "from pandas" not in src


class TestCleanerRemovesScripts:
    def test_script_content_not_in_chunks(self):
        html = _valid_html(
            "<p>Meaningful paragraph content here.</p>"
            "<script>var junk = 'SCRIPT_JUNK_MARKER';</script>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        assert "SCRIPT_JUNK_MARKER" not in result[0].text


class TestContainerBranch:
    def test_div_with_direct_text_and_paragraph(self):
        html = _valid_html(
            "<div>Direct text inside a div container "
            "<p>Paragraph text under a div wrapper.</p></div>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        assert "Direct text" in result[0].text
        assert "Paragraph text" in result[0].text

    def test_section_container_recurse(self):
        html = _valid_html(
            "<section><p>First paragraph inside a section.</p>"
            "<p>Second paragraph inside a section.</p></section>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        assert "First paragraph" in result[0].text
        assert "Second paragraph" in result[0].text

    def test_plain_text_without_document_wrapper(self):
        result = chunking_pipeline(
            page_id="1", page_title="P",
            html="Just a plain text page, long enough to be meaningful.",
        )
        assert len(result) == 1
        assert "plain text page" in result[0].text


class TestEmptyHeading:
    def test_empty_heading_ignored_in_hierarchy(self):
        html = _valid_html(
            "<h1></h1><h2>Real Heading</h2>"
            "<p>Content that is long enough to be chunked.</p>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        assert "Real Heading" in result[0].text
        assert result[0].heading_hierarchy == ["Real Heading"]


class TestNestedLists:
    def test_sup_tag_inside_list_item(self):
        html = _valid_html(
            "<p>List content for a page section.</p>"
            "<ul><li>item with <sup>superscript</sup> inside</li></ul>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        flat = "".join(c.text for c in result)
        assert "superscript" in flat


class TestDispatchGuards:
    def _chunker(self):
        c = HtmlChunker()
        c.process("1", "T", _valid_html(""))
        return c

    def test_non_element_ignored(self):
        c = self._chunker()
        c._process_element("hello")
        assert c._current_paragraphs == []

    def test_comment_ignored(self):
        from lxml import etree
        c = self._chunker()
        c._process_element(etree.Comment("hidden comment"))
        assert c._current_paragraphs == []

    def test_script_tag_skipped(self):
        from lxml.html import fromstring
        c = self._chunker()
        script = fromstring("<html><body><script>var x=1;</script></body></html>").xpath("//script")[0]
        c._process_element(script)
        assert "var x=1" not in c._current_paragraphs

    def test_leaf_element_takes_text(self):
        from lxml.html import fromstring
        c = self._chunker()
        sup = fromstring("<html><body><sup>sup note</sup></body></html>").xpath("//sup")[0]
        c._process_element(sup)
        assert c._current_paragraphs == ["sup note"]


class TestSectionDirectText:
    def test_section_with_direct_text_child(self):
        html = _valid_html(
            "<section>Direct section text here "
            "<p>Paragraph inside a section element.</p></section>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        assert "Direct section text" in result[0].text
        assert "Paragraph inside" in result[0].text

    def test_nested_ul(self):
        html = _valid_html(
            "<p>List content for a page section.</p>"
            "<ul><li>top item with text"
            "<ul><li>nested child item</li></ul></li></ul>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        flat = "".join(c.text for c in result)
        assert "top item with text" in flat
        assert "nested child item" in flat


class TestCodeBlocks:
    def test_pre_preserves_newlines_and_indentation(self):
        html = _valid_html(
            "<p>Intro text for the section.</p>"
            "<pre>def foo():\n    return 42</pre>"
            "<p>Outro text for the section.</p>"
        )
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        chunk = result[0]
        assert chunk.chunk_type == "section"
        assert "def foo():\n    return 42" in chunk.text
        assert "Intro text" in chunk.text
        assert "Outro text" in chunk.text

    def test_pre_cleans_trailing_ws_and_blank_runs(self):
        html = _valid_html("<pre>a  \n\n\n\nb</pre>")
        result = chunking_pipeline(page_id="1", page_title="P", html=html)
        assert len(result) == 1
        assert "a\n\nb" in result[0].text
