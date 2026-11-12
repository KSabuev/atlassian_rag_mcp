"""HTML chunker for Confluence pages — table-aware, unified block schema.

Size measurement uses one shared Qwen3 tokenizer counter (the real tokenizer,
bundled with the project, loaded once; raises on a missing tokenizer file —
no byte fallback).
"""
from __future__ import annotations

import logging
import uuid

from lxml import etree
from lxml.html import tostring

from atlassian_rag_mcp.config.settings import settings
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.utils.tokenizer import count_qwen3_tokens, get_qwen3_tokenizer

from ..models import ChunkPayload, PageContext
from .html_cleaner import clean_confluence_html

_chunk_counter = None
_counter_resolved = False


def _get_chunk_counter():
    global _chunk_counter, _counter_resolved
    if not _counter_resolved:
        get_qwen3_tokenizer()
        _chunk_counter = count_qwen3_tokens
        _counter_resolved = True
    return _chunk_counter


class HtmlChunker:
    """Converts Confluence HTML to markdown-like chunks, preserving tables as HTML.

    A single size budget (`max_tokens`, default ``settings.chunk_max_tokens``,
    measured by the Qwen3 tokenizer) is applied to the FINAL chunk text for
    BOTH sections and tables: for sections it is the heading prefix + plain
    text, for tables it is the raw HTML markup only (heading context lives
    in the ``heading_hierarchy`` metadata). One chunk per logical unit:

      - text between headings -> one `section` chunk, or if it exceeds the
        budget, one `section_part` chunk per paragraph (block_part_id=1..N);
        a single paragraph over the budget is emitted as one oversized chunk
        (re-split by the vector layer's CHUNK_MAX_TOKENS safety net before
        embedding).
      - a table: no text -> skipped; within budget -> one `table` chunk with
        the full <table> (the LLM gets full column context); over budget ->
        one `table_row` chunk per top-level <tr> of this table (rows of
        nested tables stay inside the nested table), each emitted as its own
        <tr>...</tr> (column context preserved via nested tables).
      - a code block (<pre>) becomes its own paragraph in the current section,
        with line structure (newlines, indentation) preserved.

    All chunks of a logical block (a table OR a section) share a common
    `block_index` (1-based, monotonically increasing across the whole page);
    `block_part_id` marks the position inside that block.
    """

    _HEADING_TAGS = ("h1", "h2", "h3", "h4", "h5", "h6")
    _CONTAINERS = ("html", "body", "div", "section", "main", "article", "form")
    _SKIP_TAGS = ("head", "title", "script", "style", "noscript")

    def __init__(self) -> None:
        self._ctx: PageContext | None = None
        self._heading_stack: list[tuple[int, str]] = []
        self._chunk_index: int = 0
        self._block_index: int = 0
        self._max_tokens: int | None = None
        self._current_paragraphs: list[str] = []
        self._chunks: list[ChunkPayload] = []

    def _count(self, text: str) -> int:
        return _get_chunk_counter()(text)

    def process(
        self,
        page_id: str,
        page_title: str,
        html: str,
        page_version: int | None = None,
        updated_at: str | None = None,
        parent_id: str | None = None,
        max_tokens: int | None = None,
    ) -> list[ChunkPayload]:
        """Parse raw Confluence HTML into chunks (cleans first, then chunks the same tree)."""
        self._ctx = PageContext(
            page_id=page_id,
            page_title=page_title,
            page_version=page_version,
            updated_at=updated_at,
            parent_id=parent_id,
        )
        self._heading_stack = []
        self._chunk_index = 0
        self._block_index = 0
        self._max_tokens = max_tokens if max_tokens is not None else settings.chunk_max_tokens
        self._current_paragraphs = []
        self._chunks = []

        root = clean_confluence_html(html)

        if root is not None:
            logger.debug(
                f"chunker: page={page_id} max_tokens={self._max_tokens} "
                f"cleaned tree before chunking:\n{tostring(root, encoding='unicode')}"
            )

        self._process_element(root)
        self._flush_paragraphs()
        return self._chunks

    def _process_element(self, elem) -> None:
        """Recursively process an lxml element, accumulating state.

        Inline content is taken exactly once, at the leaf level: block
        containers pass through their direct text and children, while
        leaf-ish elements (span, a, acronym, code, macro, ...) take their
        text and do not recurse — this prevents double counting.
        """
        if not isinstance(elem, etree._Element) or not isinstance(elem.tag, str):
            return

        name = elem.tag

        if name in self._HEADING_TAGS:
            self._flush_paragraphs()
            self._push_heading(elem)
            return

        if name == "table":
            self._flush_paragraphs()
            self._parse_table(elem)
            return

        if name == "p":
            self._append_text(_clean_text(elem.text_content()))
            return

        if name == "pre":
            self._append_text(_clean_code_text(elem.text_content()))
            return

        if name in ("ul", "ol"):
            lines = _ul_to_markdown(elem)
            if lines:
                self._current_paragraphs.append(lines)
            return

        if name in self._CONTAINERS:
            self._append_text(_clean_text(elem.text))
            for child in elem:
                self._process_element(child)
                self._append_text(_clean_text(child.tail or ""))
            return

        if name in self._SKIP_TAGS:
            return

        self._append_text(_clean_text(elem.text_content()))

    def _append_text(self, text: str) -> None:
        if text:
            self._current_paragraphs.append(text)

    def _parse_table(self, table: etree._Element) -> None:
        table_text = _clean_text(table.text_content())

        if not table_text:
            return

        _, hierarchy_list = self._build_heading_info()

        self._block_index += 1
        bindex = self._block_index

        table_html = _element_html(table)

        if self._count(table_html) <= self._max_tokens:
            self._emit_block_chunk(
                text=table_html,
                chunk_type="table",
                block_index=bindex,
                block_part_id=None,
                hierarchy_list=hierarchy_list,
            )
            return

        rows = [tr for tr in table.xpath(".//tr") if _nearest_table(tr) is table]
        part_id = 1
        for tr in rows:
            if not _clean_text(tr.text_content()):
                part_id += 1
                continue
            self._emit_block_chunk(
                text=_element_html(tr),
                chunk_type="table_row",
                block_index=bindex,
                block_part_id=part_id,
                hierarchy_list=hierarchy_list,
            )
            part_id += 1

    def _emit(self, chunk: ChunkPayload) -> None:
        self._chunks.append(chunk)
        self._chunk_index += 1

    def _emit_block_chunk(
        self,
        text: str,
        chunk_type: str,
        block_index: int,
        block_part_id: int | None,
        hierarchy_list: list[str],
    ) -> None:
        chunk = ChunkPayload(
            chunk_id=str(uuid.uuid4()),
            text=text,
            chunk_type=chunk_type,
            block_index=block_index,
            block_part_id=block_part_id,
            chunk_index=self._chunk_index,
            heading_hierarchy=hierarchy_list,
            **self._ctx.model_dump(),
        )
        self._emit(chunk)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"chunker: emitted chunk_id={chunk.chunk_id} chunk_type={chunk_type} "
                f"block_index={block_index} block_part_id={block_part_id} "
                f"chunk_index={chunk.chunk_index} heading_hierarchy={hierarchy_list} "
                f"tokens={self._count(text)}:\n{text}"
            )

    def _flush_paragraphs(self) -> None:
        if not self._current_paragraphs:
            return

        paragraphs, self._current_paragraphs = self._current_paragraphs, []
        heading_md, hierarchy_list = self._build_heading_info()
        prefix = f"{heading_md}\n\n" if heading_md else ""

        combined = "\n\n".join(paragraphs)

        self._block_index += 1
        bindex = self._block_index

        if self._count(prefix + combined) <= self._max_tokens:
            self._emit_block_chunk(
                text=prefix + combined,
                chunk_type="section",
                block_index=bindex,
                block_part_id=None,
                hierarchy_list=hierarchy_list,
            )
            return

        part_id = 1
        for p in paragraphs:
            self._emit_block_chunk(
                text=prefix + p,
                chunk_type="section_part",
                block_index=bindex,
                block_part_id=part_id,
                hierarchy_list=hierarchy_list,
            )
            part_id += 1

    def _push_heading(self, elem: etree._Element) -> None:
        """Push a heading onto the stack, first popping headings of the same
        or higher level (lower number = higher level)."""
        level = int(elem.tag[1])
        text = _clean_text(elem.text_content())
        if not text:
            return
        while self._heading_stack and self._heading_stack[-1][0] >= level:
            self._heading_stack.pop()
        self._heading_stack.append((level, text))

    def _build_heading_info(self) -> tuple[str, list[str]]:
        """Return (markdown heading of the current section, hierarchy list)."""
        if not self._heading_stack:
            return "", []
        level, text = self._heading_stack[-1]
        return f"{'#' * min(level, 6)} {text}", [t for _, t in self._heading_stack]


def _element_html(el: etree._Element) -> str:
    """Serialize el to HTML without its tail text (lxml's tostring includes it)."""
    tail = el.tail
    el.tail = None
    try:
        return tostring(el, encoding="unicode")
    finally:
        el.tail = tail


def _nearest_table(el: etree._Element) -> etree._Element | None:
    """The closest ancestor table of el (walking above any implicit tbody)."""
    node = el.getparent()
    while node is not None:
        if isinstance(node, etree._Element) and node.tag == "table":
            return node
        node = node.getparent()
    return None


def _clean_text(text: str | None) -> str:
    """Normalize whitespace in a text node."""
    return " ".join((text or "").split())


def _clean_code_text(text: str | None) -> str:
    """Normalize code-block text, preserving line structure (newlines and
    indentation): trailing whitespace per line is stripped, blank runs of 3+
    lines collapse to two, leading/trailing empty lines are dropped."""
    lines = [(line or "").rstrip() for line in (text or "").splitlines()]
    out: list[str] = []
    for line in lines:
        if line == "" and out and out[-1] == "":
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


def _ul_to_markdown(ul: etree._Element) -> str:
    """Convert <ul>/<ol> to '- item' lines."""
    lines: list[str] = []

    def _walk(node: etree._Element, depth: int = 0) -> None:
        for li in node.xpath("./li"):
            text_parts: list[str] = []
            if li.text:
                text_parts.append(_clean_text(li.text))
            for child in li:
                if not isinstance(child, etree._Element) or not isinstance(child.tag, str) or child.tag in ("ul", "ol"):
                    continue
                text_parts.append(_clean_text(child.text_content()))
                if child.tail:
                    text_parts.append(_clean_text(child.tail))
            line = " ".join(p for p in text_parts if p)
            if line:
                lines.append("  " * depth + f"- {line}")
            for sub in li.xpath("./ul | ./ol"):
                _walk(sub, depth + 1)

    _walk(ul)
    return "\n".join(lines)
