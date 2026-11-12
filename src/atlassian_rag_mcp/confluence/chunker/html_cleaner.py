"""Cleans Confluence HTML — removes junk, macros, styles, attributes.

Works in-place on an lxml tree (single parse, shared with the chunker)
and returns the cleaned root element.
"""

from __future__ import annotations

import re

from lxml import etree
from lxml.html import document_fromstring, tostring

from atlassian_rag_mcp.log.logger import logger

_JIRA_KEY_RE = re.compile(r'key\s*=\s*([A-Z][A-Z0-9]*-\d+)', re.IGNORECASE)

TABLE_TAGS = {"table", "tr", "td", "th", "thead", "tbody", "tfoot"}
KEEP_TABLE_ATTRS = {"colspan", "rowspan"}


def _cls(token: str) -> str:
    """XPath predicate matching an exact token inside @class."""
    return f"contains(concat(' ', normalize-space(@class), ' '), ' {token} ')"


def clean_confluence_html(html: str) -> etree._Element | None:
    """Cleans Confluence HTML and returns the cleaned lxml tree (root element).

    Returns None for empty or unparseable input (e.g. comment-only documents).
    """
    logger.debug(f"cleaner: input html ({len(html)} chars):\n{html}")
    if not html or not html.strip():
        logger.debug("cleaner: empty input, returning None")
        return None
    try:
        root = document_fromstring(html)
    except etree.ParserError:
        logger.debug("cleaner: unparseable input (ParserError), returning None")
        return None

    _remove_jira_macros(root)
    _remove_by_xpath(root, "//script | //noscript | //style | //colgroup | //col")
    _unwrap_by_xpath(
        root,
        f"//span[{_cls('inline-comment-marker')}]"
        f" | //span[{_cls('terms-mark')}]"
        f" | //div[{_cls('content-wrapper')}]",
    )
    _strip_attributes(root)
    _unwrap_by_xpath(root, "//b | //strong | //i | //em | //u")
    _remove_spans(root)
    _mark_strikethrough(root)
    _replace_br(root)
    _remove_empty_paragraphs(root)
    _unwrap_by_xpath(root, "//a | //time | //div | //code | //sub | //hr | //img")

    cleaned = tostring(root, encoding="unicode")
    logger.debug(
        f"cleaner: cleaned html ({len(cleaned)} chars, text_content={len(root.text_content())} chars):\n{cleaned}"
    )
    return root


def _remove_jira_macros(root: etree._Element) -> None:
    """Removes/replaces Jira macros (div, span with jira-issue classes, etc.).

    A macro nested inside another macro reaches this loop without a parent
    (already removed together with its outer macro) — such nodes are skipped.
    """
    macro_xpath = (
        "//*[name() = 'div' or name() = 'span']["
        f"{_cls('refresh-module-id')} or {_cls('jira-issue')} or {_cls('jim-sortable')}"
        "]"
    )
    for tag in root.xpath(macro_xpath):
        if tag.getparent() is None:
            continue
        key = None

        links = tag.xpath(f".//a[{_cls('jira-issue-key')}]")
        if links:
            key = (links[0].text_content() or "").strip()

        if not key:
            for inp in tag.xpath(f".//input[{_cls('refresh-wiki')}]"):
                markup = inp.get("data-wikimarkup")
                if markup:
                    m = _JIRA_KEY_RE.search(markup)
                    if m:
                        key = m.group(1)
                        break

        if not key:
            key = (tag.get("data-jira-key") or "").strip()

        if key:
            _insert_text_before(tag, key)
        _remove_el(tag)

    _remove_by_xpath(
        root,
        "//*[name() = 'div']["
        f"{_cls('refresh-issues-bottom')} or {_cls('jira-issues')} or {_cls('refresh-issues-loading')}"
        f"] | //*[name() = 'a'][{_cls('refresh-action')}]"
        f" | //*[name() = 'input'][{_cls('refresh-wiki')}]",
    )


def _remove_by_xpath(root: etree._Element, xpath: str) -> None:
    for el in root.xpath(xpath):
        _remove_el(el)


def _unwrap_by_xpath(root: etree._Element, xpath: str) -> None:
    for el in root.xpath(xpath):
        if el.getparent() is not None:
            _unwrap(el)


def _strip_attributes(root: etree._Element) -> None:
    """Clears all attributes, keeping colspan/rowspan on table elements.

    Replaces the old separate style/class/data-*/table-attr passes.
    Must run AFTER _remove_jira_macros — the macro pass still reads
    class and data-* attributes.
    """
    for el in root.iter():
        if not isinstance(el, etree._Element) or not isinstance(el.tag, str):
            continue
        if el.tag in TABLE_TAGS:
            for name in [n for n in el.attrib if n not in KEEP_TABLE_ATTRS]:
                del el.attrib[name]
        else:
            el.attrib.clear()


def _remove_spans(root: etree._Element) -> None:
    """Unwraps span, adding a space between touching tag siblings."""
    for tag in root.xpath("//span"):
        if tag.getparent() is None:
            continue
        parent = tag.getparent()
        prev = tag.getprevious()
        nxt = tag.getnext()

        def _text(node) -> str:
            if node is None:
                return ""
            if isinstance(node, etree._Element) and isinstance(node.tag, str):
                return node.text_content()
            return str(node)

        need_space = False
        if (
            prev is not None
            and nxt is not None
            and any(isinstance(n, etree._Element) and isinstance(n.tag, str) for n in (prev, nxt))
        ):
            p, n = _text(prev), _text(nxt)
            if p and n and not p[-1].isspace() and not n[0].isspace():
                need_space = True

        last = tag[-1] if len(tag) else None
        _unwrap(tag)
        if need_space:
            if last is not None:
                last.tail = (last.tail or "") + " "
            elif tag.text:
                if prev is not None:
                    prev.tail = (prev.tail or "") + " "
                else:
                    parent.text = (parent.text or "") + " "


def _mark_strikethrough(root: etree._Element) -> None:
    """Marks <s> content as ~~s~~, then unwraps the tag."""
    for tag in root.xpath("//s"):
        if tag.getparent() is None:
            continue
        _insert_text_before(tag, "~~")
        tag.tail = "~~" + (tag.tail or "")
        _unwrap(tag)


def _replace_br(root: etree._Element) -> None:
    """Replaces <br> with newline."""
    for tag in root.xpath("//br"):
        if tag.getparent() is None:
            continue
        _insert_text_before(tag, "\n" + (tag.tail or ""))
        tag.getparent().remove(tag)


def _remove_empty_paragraphs(root: etree._Element) -> None:
    """Removes empty <p>, keeps non-empty ones for context search."""
    for tag in root.xpath("//p[not(normalize-space())]"):
        _remove_el(tag)


def _remove_el(el: etree._Element) -> None:
    """Removes el, preserving its tail text (lxml drops it on remove)."""
    parent = el.getparent()
    if parent is None:
        return
    prev = el.getprevious()
    tail = el.tail
    parent.remove(el)
    if tail:
        if prev is not None:
            prev.tail = (prev.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail


def _insert_text_before(el: etree._Element, text: str) -> None:
    """Inserts plain text immediately before el (lxml has no text nodes)."""
    prev = el.getprevious()
    if prev is not None:
        prev.tail = (prev.tail or "") + text
    else:
        el.getparent().text = (el.getparent().text or "") + text


def _unwrap(el: etree._Element) -> None:
    """Replaces el with its content (text and children, in order)."""
    parent = el.getparent()
    prev = el.getprevious()
    last = el[-1] if len(el) else None
    if el.text:
        if prev is not None:
            prev.tail = (prev.tail or "") + el.text
        else:
            parent.text = (parent.text or "") + el.text
    for child in list(el):
        el.addprevious(child)
    if el.tail:
        if last is not None:
            last.tail = (last.tail or "") + el.tail
        elif prev is not None:
            prev.tail = (prev.tail or "") + el.tail
        else:
            parent.text = (parent.text or "") + el.tail
    parent.remove(el)
