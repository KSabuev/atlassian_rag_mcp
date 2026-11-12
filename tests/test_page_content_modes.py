"""Tests for the Confluence page content registry (confluence/page_content/)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atlassian_rag_mcp.confluence.models import Page, PageContent
from atlassian_rag_mcp.confluence.page_content import (
    VALID_MODES,
    ChunksMode,
    FilesMode,
    HtmlMode,
    InfoMode,
    TableMode,
    execute_page_content,
)

EXPECTED_MODES = {"info", "html", "chunks", "table", "files", "attachments"}


@pytest.fixture
def svc():
    s = MagicMock()
    s.search_deps.confluence_url = "https://c.example.com/"
    s.index_queue.get_in_work_versions = AsyncMock(return_value={})
    s.vector_service.get_chunks_by_filter = AsyncMock(return_value=[
        {"chunk_id": "c1", "text": "one", "chunk_type": "section",
         "page_id": "123", "page_title": "P", "chunk_index": 0},
    ])
    s.confluence_client.list_attachments = AsyncMock(return_value=[
        {"title": "f.xml", "media_type": "application/xml", "size": 10,
         "version_number": 1, "created": "2024-01-01",
         "download_url": "https://c.example.com/download/attachments/123/f.xml"},
    ])
    return s


@pytest.fixture
def page():
    return PageContent(
        page_id="123", page_title="P", version=1, last_updated="2024-01-01",
        html="<p>text</p>",
    )


def test_valid_modes_all_expected():
    assert set(VALID_MODES) == EXPECTED_MODES


def test_registry_classes_are_subclasses_of_mode():
    for mode in (InfoMode, HtmlMode, ChunksMode, TableMode, FilesMode):
        assert hasattr(mode, "name")
        assert hasattr(mode, "execute")


async def test_unknown_mode_returns_error(svc, page):
    svc.confluence_client.get_page = AsyncMock(return_value=page)
    result = await execute_page_content("bogus", "123", svc)
    assert result.error is not None
    assert "bogus" in result.error


async def test_info_returns_metadata(svc, page):
    svc.confluence_client.get_page = AsyncMock(return_value=page)
    result = await execute_page_content("info", "123", svc)
    assert result.error is None
    assert result.page_title == "P"
    assert result.version == 1
    assert result.chunks == []
    assert result.html is None


async def test_html_returns_cleaned_content(svc, page):
    svc.confluence_client.get_page = AsyncMock(return_value=page)
    result = await execute_page_content("html", "123", svc)
    assert result.error is None
    assert "text" in result.html


async def test_html_empty_page_returns_empty_string(svc):
    empty = PageContent(page_id="123", page_title="P", version=1, html="")
    svc.confluence_client.get_page = AsyncMock(return_value=empty)
    result = await execute_page_content("html", "123", svc)
    assert result.error is None
    assert result.html == ""


async def test_chunks_returns_chunks(svc, page):
    svc.confluence_client.get_page = AsyncMock(return_value=page)
    result = await execute_page_content("chunks", "123", svc)
    assert result.error is None
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "one"


async def test_table_requires_block_index(svc, page):
    svc.confluence_client.get_page = AsyncMock(return_value=page)
    result = await execute_page_content("table", "123", svc)
    assert result.error is not None
    assert "block_index" in result.error


async def test_table_with_block_index(svc, page):
    svc.confluence_client.get_page = AsyncMock(return_value=page)
    svc.vector_service.get_chunks_by_filter = AsyncMock(return_value=[
        {"chunk_id": "r1", "text": "row1", "chunk_type": "table_row",
         "page_id": "123", "page_title": "P", "chunk_index": 0,
         "block_index": 3, "block_part_id": 1},
    ])
    result = await execute_page_content("table", "123", svc, block_index=3)
    assert result.error is None
    assert len(result.chunks) == 1
    svc.vector_service.get_chunks_by_filter.assert_awaited_once_with(
        page_id="123", block_index=3, limit=1000, offset=0, in_work_versions={},
    )


async def test_files_returns_attachments(svc, page):
    svc.confluence_client.get_page = AsyncMock(return_value=page)
    result = await execute_page_content("files", "123", svc)
    assert result.error is None
    assert len(result.files) == 1
    assert result.files[0].title == "f.xml"


async def test_page_not_found_propagates_error(svc):
    svc.confluence_client.get_page = AsyncMock(return_value=Page(error="Page 999 not found"))
    result = await execute_page_content("info", "999", svc)
    assert result.error is not None
    assert "not found" in result.error


async def test_handler_exception_returns_error(svc):
    svc.confluence_client.get_page = AsyncMock(side_effect=ConnectionError("no network"))
    result = await execute_page_content("info", "123", svc)
    assert result.error is not None
    assert "no network" in result.error


async def test_files_api_error_returns_error(svc, page):
    svc.confluence_client.get_page = AsyncMock(return_value=page)
    svc.confluence_client.list_attachments = AsyncMock(side_effect=RuntimeError("attachments down"))
    result = await execute_page_content("files", "123", svc)
    assert result.error is not None
    assert "attachments down" in result.error


async def test_url_built_from_confluence_url(svc, page):
    svc.confluence_client.get_page = AsyncMock(return_value=page)
    result = await execute_page_content("info", "123", svc)
    assert result.page_url == "https://c.example.com/pages/viewpage.action?pageId=123"
