"""Tests for Confluence HTML cleaner (clean_confluence_html)."""

from __future__ import annotations

from lxml import etree

from atlassian_rag_mcp.confluence.chunker.html_cleaner import clean_confluence_html


def _clean(html: str) -> str:
    """clean_confluence_html returns an lxml tree (None for empty input); serialize."""
    tree = clean_confluence_html(html)
    return "" if tree is None else etree.tostring(tree, encoding="unicode")


class TestJiraMacros:
    def test_jira_issue_link_replaced_with_key(self):
        html = '<div class="jira-issue"><a class="jira-issue-key">PROJ-123</a></div>'
        result = _clean(html)
        assert "PROJ-123" in result
        assert "jira-issue" not in result

    def test_jira_issue_from_data_wikimarkup(self):
        html = (
            '<span class="refresh-module-id">'
            '<input class="refresh-wiki" data-wikimarkup="&lt;ac:parameter&gt;key=PROJ-456&lt;/ac:parameter&gt;" />'
            "</span>"
        )
        result = _clean(html)
        assert "PROJ-456" in result

    def test_jira_issue_from_data_jira_key(self):
        html = '<span class="jira-issue" data-jira-key="PROJ-789">x</span>'
        result = _clean(html)
        assert "PROJ-789" in result

    def test_jira_issue_other_project_key(self):
        html = (
            '<span class="refresh-module-id">'
            '<input class="refresh-wiki" data-wikimarkup="&lt;ac:parameter&gt;key=PROJ-456&lt;/ac:parameter&gt;" />'
            "</span>"
        )
        result = _clean(html)
        assert "PROJ-456" in result

    def test_jira_macro_without_key_removed(self):
        html = '<span class="jira-issue">no key here</span>'
        result = _clean(html)
        assert "no key here" not in result

    def test_refresh_issues_container_removed(self):
        html = '<div class="refresh-issues-bottom">footer</div>'
        result = _clean(html)
        assert "footer" not in result

    def test_refresh_action_link_removed(self):
        html = '<a class="refresh-action">refresh</a>'
        result = _clean(html)
        assert "refresh" not in result

    def test_refresh_wiki_input_removed(self):
        html = '<input class="refresh-wiki" data-wikimarkup="key=NOPE" />'
        result = _clean(html)
        assert "refresh-wiki" not in result


class TestScriptsAndMarkers:
    def test_scripts_removed(self):
        html = "<p>text</p><script>alert(1)</script><style>x{}</style><noscript>ns</noscript>"
        result = _clean(html)
        assert "alert" not in result
        assert "x{}" not in result
        assert "ns" not in result
        assert "text" in result

    def test_inline_comment_markers_unwrapped(self):
        html = '<span class="inline-comment-marker">commented</span>'
        result = _clean(html)
        assert "commented" in result
        assert "inline-comment-marker" not in result

    def test_terms_marks_unwrapped(self):
        html = '<span class="terms-mark">term</span>'
        result = _clean(html)
        assert "term" in result


class TestTables:
    def test_colspan_rowspan_preserved(self):
        html = '<table><tr><td colspan="2" rowspan="1">cell</td></tr></table>'
        result = _clean(html)
        assert 'colspan="2"' in result
        assert 'rowspan="1"' in result

    def test_other_table_attrs_removed(self):
        html = '<table class="wrapped" style="x" border="1"><tr><td>cell</td></tr></table>'
        result = _clean(html)
        assert "border" not in result
        assert "wrapped" not in result
        assert "cell" in result

    def test_colgroup_removed(self):
        html = "<table><colgroup><col width=\"100\" /></colgroup><tr><td>c</td></tr></table>"
        result = _clean(html)
        assert "colgroup" not in result
        assert "c" in result


class TestInlineFormatting:
    def test_formatting_tags_unwrapped(self):
        html = "<p><b>bold</b> <i>italic</i></p>"
        result = _clean(html)
        assert "<b>" not in result
        assert "<i>" not in result
        assert "bold" in result
        assert "italic" in result

    def test_br_replaced_with_newline(self):
        html = "<p>line1<br>line2</p>"
        result = _clean(html)
        assert "<br" not in result
        assert "line1" in result and "line2" in result

    def test_empty_paragraph_removed(self):
        html = "<p></p><p>  </p><p>keep</p>"
        result = _clean(html)
        assert "keep" in result
        assert result.count("<p>") == 1

    def test_links_unwrapped_href_lost(self):
        html = '<a href="https://x">linktext</a>'
        result = _clean(html)
        assert "linktext" in result
        assert "href" not in result

    def test_strikethrough_marked(self):
        html = "<p>before<s>gone</s>after</p>"
        result = _clean(html)
        assert "~~gone~~" in result

    def test_simple_tags_unwrapped(self):
        html = "<p><time>2020</time> <code>code</code> <sub>sub</sub></p>"
        result = _clean(html)
        assert "<time>" not in result
        assert "<code>" not in result
        assert "2020" in result and "code" in result and "sub" in result

    def test_pre_block_preserved(self):
        html = "<pre>line1\nline2</pre>"
        result = _clean(html)
        assert "<pre" in result
        assert "line1" in result and "line2" in result

    def test_hr_unwrapped(self):
        html = "<p>a</p><hr /><p>b</p>"
        result = _clean(html)
        assert "<hr" not in result

    def test_span_space_between_text_nodes(self):
        html = "<p>foo<span>bar</span>baz</p>"
        result = _clean(html)
        assert "foo" in result and "bar" in result and "baz" in result


class TestAttributes:
    def test_style_attribute_removed(self):
        html = '<p style="color:red">text</p>'
        result = _clean(html)
        assert "color:red" not in result
        assert "text" in result

    def test_classes_removed(self):
        html = '<div class="content-wrapper"><p class="lead">text</p></div>'
        result = _clean(html)
        assert "content-wrapper" not in result
        assert "lead" not in result
        assert "text" in result

    def test_data_attributes_removed(self):
        html = '<p data-ref="r1" data-client-id="c1" title="t">text</p>'
        result = _clean(html)
        assert "data-ref" not in result
        assert "title" not in result
        assert "text" in result
