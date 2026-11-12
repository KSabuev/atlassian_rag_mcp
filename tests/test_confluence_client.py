"""Unit tests for ConfluenceClient — atlassian lib mocked."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from atlassian.errors import ApiValueError
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import HTTPError as RequestsHTTPError

from atlassian_rag_mcp.confluence.client import ConfluenceClient
from atlassian_rag_mcp.exceptions.errors import AuthenticationError, CqlError


@pytest.fixture
def client():
    raw = MagicMock()
    raw.url = "https://c.example.com"
    return ConfluenceClient(confluence=raw)


def _raw_response(items):
    return {"results": items}


class TestListAttachments:
    async def test_normalizes_raw_response(self, client):
        client.confluence.get_attachments_from_content = MagicMock(
            return_value=_raw_response([
                {
                    "title": "file.xml",
                    "mediaType": "application/xml",
                    "size": 1234,
                    "version": {"number": 3, "when": "2024-06-01T10:00:00.000+03:00"},
                },
            ])
        )
        items = await client.list_attachments("123")
        assert items == [{
            "title": "file.xml",
            "media_type": "application/xml",
            "size": 1234,
            "version_number": 3,
            "created": "2024-06-01T10:00:00.000+03:00",
            "download_url": "https://c.example.com/download/attachments/123/file.xml",
        }]

    async def test_download_url_is_urlencoded(self, client):
        client.confluence.get_attachments_from_content = MagicMock(
            return_value=_raw_response([
                {"title": "xml образец (v1).xml", "mediaType": "application/xml", "size": 10,
                 "version": {"number": 1, "when": "2024-01-01"}},
            ])
        )
        items = await client.list_attachments("42")
        assert items[0]["download_url"] == (
            "https://c.example.com/download/attachments/42/"
            "xml%20%D0%BE%D0%B1%D1%80%D0%B0%D0%B7%D0%B5%D1%86%20%28v1%29.xml"
        )

    async def test_empty_attachment_list(self, client):
        client.confluence.get_attachments_from_content = MagicMock(
            return_value=_raw_response([])
        )
        assert await client.list_attachments("42") == []

    async def test_none_raw_response(self, client):
        client.confluence.get_attachments_from_content = MagicMock(return_value=None)
        assert await client.list_attachments("42") == []

    async def test_missing_version_fields(self, client):
        client.confluence.get_attachments_from_content = MagicMock(
            return_value=_raw_response([{"title": "a.txt"}])
        )
        items = await client.list_attachments("42")
        assert items[0]["version_number"] is None
        assert items[0]["created"] is None
        assert items[0]["media_type"] is None

    async def test_limit_passed_to_library(self, client):
        client.confluence.get_attachments_from_content = MagicMock(
            return_value=_raw_response([])
        )
        await client.list_attachments("42", limit=7)
        args, kwargs = client.confluence.get_attachments_from_content.call_args
        assert kwargs["limit"] == 7
        assert kwargs["expand"] == "version"

    async def test_no_base_url_download_is_none(self, client):
        client.confluence.url = ""
        client.confluence.get_attachments_from_content = MagicMock(
            return_value=_raw_response([{"title": "a.txt"}])
        )
        items = await client.list_attachments("42")
        assert items[0]["download_url"] is None

    async def test_error_propagates(self, client):
        client.confluence.get_attachments_from_content = MagicMock(
            side_effect=RuntimeError("404 Not Found")
        )
        with pytest.raises(RuntimeError, match="404 Not Found"):
            await client.list_attachments("999")


class TestGetPage:
    async def test_returns_page_result(self, client):
        client.confluence.get_page_by_id = MagicMock(return_value={
            "id": "42",
            "title": "Registry",
            "version": {"number": 7, "when": "2026-01-01T00:00:00.000Z"},
            "ancestors": [{"id": "10"}, {"id": "20"}],
            "body": {"view": {"value": "<p>html</p>"}},
        })
        res = await client.get_page("42")
        assert res.error is None
        assert res.page_id == "42"
        assert res.page_title == "Registry"
        assert res.version == 7
        assert res.last_updated == "2026-01-01T00:00:00.000Z"
        assert res.parent_id == "20"
        assert res.html == "<p>html</p>"

    async def test_not_found_returns_error(self, client):
        client.confluence.get_page_by_id = MagicMock(return_value=None)
        res = await client.get_page("999")
        assert res.error is not None
        assert "999" in res.error

    async def test_no_ancestors_parent_none(self, client):
        client.confluence.get_page_by_id = MagicMock(return_value={
            "id": "1", "title": "Root", "version": {"number": 1},
            "body": {"view": {"value": "x"}},
        })
        res = await client.get_page("1")
        assert res.parent_id is None


class TestCql:
    async def test_returns_cql_result(self, client):
        client.confluence.cql = MagicMock(return_value={
            "results": [{"id": "1"}],
            "totalSize": 1,
        })
        res = await client.cql("type=page")
        assert res.total == 1
        assert res.results == [{"id": "1"}]
        assert res.query == "type=page"

    def _sdk_400(self, body_message: str) -> ApiValueError:
        """Mimic the atlassian SDK: ApiValueError with requests.HTTPError in reason."""
        resp = MagicMock(status_code=400)
        resp.json = MagicMock(return_value={"message": body_message})
        return ApiValueError(
            "The query cannot be parsed",
            reason=RequestsHTTPError("Bad Request", response=resp),
        )

    async def test_cql_error_carries_query_and_api_message(self, client):
        client.confluence.cql = MagicMock(
            side_effect=self._sdk_400("Could not parse cql : type=page AND bogus ~~~"),
        )
        with pytest.raises(CqlError) as ei:
            await client.cql("type=page AND bogus ~~~")
        msg = str(ei.value)
        assert "type=page AND bogus ~~~" in msg
        assert "(HTTP 400)" in msg
        assert "Could not parse cql" in msg

    async def test_cql_error_without_response_body(self, client):
        client.confluence.cql = MagicMock(
            side_effect=RequestsConnectionError(
                "HTTPSConnectionPool(host='127.0.0.1', port=9): Connection refused",
            ),
        )
        with pytest.raises(CqlError) as ei:
            await client.cql("type=page")
        msg = str(ei.value)
        assert "type=page" in msg
        assert "Connection refused" in msg


class TestSpaces:
    async def test_get_all_spaces(self, client):
        client.confluence.get_all_spaces = MagicMock(return_value={
            "results": [{"key": "A"}, {"key": "B"}],
            "size": 2,
        })
        res = await client.get_all_spaces()
        assert res.total == 2
        assert len(res.spaces) == 2

    async def test_get_space(self, client):
        client.confluence.get_space = MagicMock(return_value={
            "key": "PROJ", "name": "Project M", "homepage": {"id": "HP"},
        })
        res = await client.get_space("PROJ")
        assert res.key == "PROJ"
        assert res.name == "Project M"
        assert res.homepage_id == "HP"


class TestValidate:
    def test_ok_when_spaces_returned(self, client):
        client.confluence.get_all_spaces = MagicMock(return_value={"results": []})
        client.validate()

    def test_raises_when_api_returns_none(self, client):
        client.confluence.get_all_spaces = MagicMock(return_value=None)
        with pytest.raises(AuthenticationError):
            client.validate()

    def test_raises_on_connection_error(self, client):
        client.confluence.get_all_spaces = MagicMock(
            side_effect=RequestsConnectionError("refused")
        )
        with pytest.raises(AuthenticationError, match="Cannot connect"):
            client.validate("https://c.example.com")

    def test_raises_on_generic_error(self, client):
        client.confluence.get_all_spaces = MagicMock(side_effect=ValueError("boom"))
        with pytest.raises(AuthenticationError):
            client.validate()
