"""Tests for Jira plugins — ZephyrClient and create_plugins factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from atlassian import Jira

from atlassian_rag_mcp.exceptions import ZephyrError
from atlassian_rag_mcp.jira.plugins import create_plugins
from atlassian_rag_mcp.jira.plugins.zephyr import ZephyrClient


@pytest.fixture
def mock_client():
    def _make(get_json=None, get=None):
        client = MagicMock()
        client.get = get or AsyncMock(return_value=MagicMock(status_code=200))
        client.aclose = AsyncMock()
        return client
    return _make


@pytest.fixture
def zephyr(mock_client):
    zp = ZephyrClient(MagicMock(), base_url="http://jira:8080", username="u", password="p")
    zp._client = mock_client()
    zp._available = True
    zp._checked = True
    return zp


def _http_response(status_code, json_body=None, text=""):
    """MagicMock response whose raise_for_status raises a real HTTPStatusError."""
    resp = MagicMock(status_code=status_code)
    if json_body is not None:
        resp.json.return_value = json_body
    else:
        resp.json.side_effect = ValueError("not json")
    resp.text = text
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "boom", request=MagicMock(), response=resp
    )
    return resp


class TestZephyrValidate:
    async def test_available_returns_true(self, mock_client):
        zp = ZephyrClient(MagicMock(), base_url="http://j:8080", username="u", password="p")
        zp._client = mock_client(get=AsyncMock(return_value=MagicMock(status_code=200)))
        assert await zp.validate() is True

    async def test_not_installed_404_returns_false(self, mock_client):
        zp = ZephyrClient(MagicMock(), base_url="http://j:8080", username="u", password="p")
        zp._client = mock_client(get=AsyncMock(return_value=_http_response(404)))
        assert await zp.validate() is False

    async def test_server_error_403_raises(self, mock_client):
        zp = ZephyrClient(MagicMock(), base_url="http://j:8080", username="u", password="p")
        zp._client = mock_client(get=AsyncMock(return_value=_http_response(403)))
        with pytest.raises(ZephyrError, match="HTTP 403"):
            await zp.validate()


class TestGetProjects:
    async def test_returns_projects(self, zephyr):
        zephyr._client.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[{"id": 1, "key": "P1"}]),
        )
        result = await zephyr.get_projects()
        assert result == [{"id": 1, "key": "P1"}]

    async def test_not_available_returns_empty(self, zephyr):
        zephyr._available = False
        result = await zephyr.get_projects()
        assert result == []

    async def test_http_error_raises(self, zephyr):
        zephyr._client.get.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=MagicMock(status_code=500)
        )
        with pytest.raises(ZephyrError, match=r"GET /rest/tests/1\.0/project \(HTTP 500\)"):
            await zephyr.get_projects()


class TestGetTestCase:
    async def test_returns_test_case(self, zephyr):
        zephyr._client.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"id": "123", "key": "P1-123", "name": "Test"}),
        )
        result = await zephyr.get_test_case("P1-123")
        assert result is not None
        assert result["key"] == "P1-123"

    async def test_404_returns_none(self, zephyr):
        zephyr._client.get.return_value = MagicMock(status_code=404)
        result = await zephyr.get_test_case("P1-999")
        assert result is None

    async def test_not_available_returns_none(self, zephyr):
        zephyr._available = False
        result = await zephyr.get_test_case("P1-123")
        assert result is None


class TestSearchTestCases:
    async def test_search_by_project_key(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "key": "P1", "name": "P1"}])),
            MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "results": [{"id": "1", "key": "P1-1"}],
                    "total": 1, "startAt": 0, "maxResults": 25,
                }),
            ),
        ]
        result = await zephyr.search_test_cases(project_key="P1")
        assert result["total"] == 1
        assert len(result["results"]) == 1

    async def test_search_with_custom_query(self, zephyr):
        zephyr._client.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"results": [], "total": 0}),
        )
        result = await zephyr.search_test_cases(tql='testCase.key IN ("P1-1")')
        assert result["total"] == 0

    async def test_not_available_returns_empty(self, zephyr):
        zephyr._available = False
        result = await zephyr.search_test_cases(project_key="P1")
        assert result == {"results": [], "total": 0, "startAt": 0, "maxResults": 25}


class TestGetFolders:
    async def test_returns_folders(self, zephyr):
        zephyr._client.get = AsyncMock(side_effect=[
            MagicMock(
                status_code=200,
                json=MagicMock(return_value=[{"id": 10202, "key": "P1", "name": "P1"}]),
            ),
            MagicMock(
                status_code=200,
                json=MagicMock(return_value=[{"id": 1, "name": "Root", "parentId": None}]),
            ),
        ])
        result = await zephyr.get_folders("P1")
        assert len(result) == 1
        assert result[0]["name"] == "Root"

    async def test_not_available_returns_empty(self, zephyr):
        zephyr._available = False
        result = await zephyr.get_folders("P1")
        assert result == []


class TestSearchTestRuns:
    async def test_search_returns_runs(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "key": "P1", "name": "P1"}])),
            MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "results": [{"id": "1", "key": "P1-R1", "name": "Run 1"}],
                    "total": 1,
                }),
            ),
        ]
        result = await zephyr.search_test_runs(project_key="P1")
        assert result["total"] == 1

    async def test_not_available_returns_empty(self, zephyr):
        zephyr._available = False
        result = await zephyr.search_test_runs(project_key="P1")
        assert result == {"results": [], "total": 0, "startAt": 0, "maxResults": 25}


class TestGetTestRun:
    async def test_returns_test_run(self, zephyr):
        zephyr._client.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"id": "1", "key": "P1-R1", "name": "Run 1"}),
        )
        result = await zephyr.get_test_run("P1-R1")
        assert result is not None
        assert result["key"] == "P1-R1"

    async def test_404_returns_none(self, zephyr):
        zephyr._client.get.return_value = MagicMock(status_code=404)
        result = await zephyr.get_test_run("P1-999")
        assert result is None

    async def test_not_available_returns_none(self, zephyr):
        zephyr._available = False
        result = await zephyr.get_test_run("P1-R1")
        assert result is None


class TestGetTestRunItems:
    async def test_returns_items(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value={"id": 5})),
            MagicMock(status_code=200, json=MagicMock(return_value={
                "testRunItems": [{"id": 10, "index": 1}], "total": 1,
            })),
        ]
        result = await zephyr.get_test_run_items("P1-R1")
        assert result["total"] == 1
        assert result["items"][0]["id"] == 10

    async def test_unknown_run_returns_empty(self, zephyr):
        zephyr._client.get.return_value = MagicMock(status_code=404)
        result = await zephyr.get_test_run_items("P1-NOPE")
        assert result == {"items": [], "total": 0}

    async def test_not_available_returns_empty(self, zephyr):
        zephyr._available = False
        result = await zephyr.get_test_run_items("P1-R1")
        assert result == {"items": [], "total": 0}


class TestGetTestRunResults:
    async def test_returns_results(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value={"id": 5})),
            MagicMock(status_code=200, json=MagicMock(return_value={
                "testRunItems": [{"id": 10}], "total": 1,
            })),
            MagicMock(status_code=200, json=MagicMock(return_value=[
                {"id": 100, "attachments": [{"id": 1, "name": "f.txt"}]},
            ])),
        ]
        result = await zephyr.get_test_run_results("P1-R1")
        assert len(result) == 1
        assert result[0]["attachments"][0]["name"] == "f.txt"

    async def test_aggregates_all_items(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value={"id": 5})),
            MagicMock(status_code=200, json=MagicMock(return_value={
                "testRunItems": [{"id": 10}, {"id": 11}], "total": 2,
            })),
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 100}])),
            MagicMock(status_code=200, json=MagicMock(return_value=[])),
        ]
        result = await zephyr.get_test_run_results("P1-R1")
        assert len(result) == 1

    async def test_with_item_id(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value={"id": 5})),
            MagicMock(status_code=200, json=MagicMock(return_value=[])),
        ]
        await zephyr.get_test_run_results("P1-R1", item_id=42)
        _, kwargs = zephyr._client.get.call_args
        assert kwargs["params"]["itemId"] == 42

    async def test_not_available_returns_empty(self, zephyr):
        zephyr._available = False
        result = await zephyr.get_test_run_results("P1-R1")
        assert result == []


class TestGetTestResult:
    async def test_returns_result(self, zephyr):
        zephyr._client.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"id": 100}),
        )
        result = await zephyr.get_test_result(100)
        assert result == {"id": 100}

    async def test_404_returns_none(self, zephyr):
        zephyr._client.get.return_value = MagicMock(status_code=404)
        assert await zephyr.get_test_result(999) is None


class TestGetTestCaseVersions:
    async def test_returns_versions(self, zephyr):
        zephyr._client.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value=[{"majorVersion": 2}, {"majorVersion": 1}]),
        )
        result = await zephyr.get_test_case_versions("P1-123")
        assert len(result) == 2

    async def test_unknown_key_returns_empty(self, zephyr):
        zephyr._client.get.return_value = MagicMock(
            status_code=200, json=MagicMock(return_value=[])
        )
        assert await zephyr.get_test_case_versions("P1-NOPE") == []


class TestGetTestCaseHistory:
    async def test_returns_history(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value={"id": 7})),
            MagicMock(status_code=200, json=MagicMock(return_value={"data": [{"id": 1}], "total": 1})),
        ]
        result = await zephyr.get_test_case_history("P1-123")
        assert result["total"] == 1
        assert result["items"][0]["id"] == 1

    async def test_not_available_returns_empty(self, zephyr):
        zephyr._available = False
        assert await zephyr.get_test_case_history("P1-123") == {"items": [], "total": 0}


class TestGetTestCaseTraceLinks:
    async def test_issue_links(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value={"id": 7})),
            MagicMock(status_code=200, json=MagicMock(return_value=[{"issueId": 456}])),
        ]
        result = await zephyr.get_test_case_trace_links("P1-123")
        assert result[0]["issueId"] == 456

    async def test_invalid_link_type(self, zephyr):
        result = await zephyr.get_test_case_trace_links("P1-123", link_type="bogus")
        assert result is None


class TestGetTestCaseComments:
    async def test_returns_comments(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value={"id": 7})),
            MagicMock(status_code=200, json=MagicMock(return_value=[{"text": "hi"}])),
        ]
        result = await zephyr.get_test_case_comments("P1-123")
        assert result[0]["text"] == "hi"


class TestGetTestCaseAttachments:
    async def test_returns_attachments(self, zephyr):
        zephyr._client.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"attachments": [{"id": 1, "name": "f.txt"}]}),
        )
        result = await zephyr.get_test_case_attachments("P1-123")
        assert result[0]["id"] == 1

    async def test_404_returns_none(self, zephyr):
        zephyr._client.get.return_value = MagicMock(status_code=404)
        assert await zephyr.get_test_case_attachments("P1-NOPE") is None


class TestSearchTestPlans:
    async def test_search_returns_plans(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "key": "P1", "name": "P1"}])),
            MagicMock(status_code=200, json=MagicMock(return_value={
                "results": [{"id": "1", "key": "P1-P1"}], "total": 1,
            })),
        ]
        result = await zephyr.search_test_plans(project_key="P1")
        assert result["total"] == 1


class TestGetTestPlan:
    async def test_returns_plan(self, zephyr):
        zephyr._client.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"id": "1", "key": "P1-P1"}),
        )
        result = await zephyr.get_test_plan("P1-P1")
        assert result["key"] == "P1-P1"

    async def test_404_returns_none(self, zephyr):
        zephyr._client.get.return_value = MagicMock(status_code=404)
        assert await zephyr.get_test_plan("P1-NOPE") is None


class TestStatusesAndPriorities:
    async def test_test_case_statuses(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "key": "P1", "name": "P1"}])),
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "name": "Done"}])),
        ]
        result = await zephyr.get_test_case_statuses("P1")
        assert result[0]["name"] == "Done"

    async def test_test_case_priorities(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "key": "P1", "name": "P1"}])),
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "name": "Critical"}])),
        ]
        result = await zephyr.get_test_case_priorities("P1")
        assert result[0]["name"] == "Critical"

    async def test_test_result_statuses(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "key": "P1", "name": "P1"}])),
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "name": "Passed"}])),
        ]
        result = await zephyr.get_test_result_statuses("P1")
        assert result[0]["name"] == "Passed"


class TestCustomFields:
    async def test_returns_fields(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "key": "P1", "name": "P1"}])),
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 10011, "name": "CF"}])),
        ]
        result = await zephyr.get_custom_fields("P1")
        assert result[0]["id"] == 10011

    async def test_invalid_field_type(self, zephyr):
        result = await zephyr.get_custom_fields("P1", field_type="bogus")
        assert result == []


class TestProjectPermissions:
    async def test_returns_permissions(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "key": "P1", "name": "P1"}])),
            MagicMock(status_code=200, json=MagicMock(return_value={"testcasePermission": {}})),
        ]
        result = await zephyr.get_project_permissions("P1")
        assert "testcasePermission" in result


class TestDatasets:
    async def test_returns_datasets(self, zephyr):
        zephyr._client.get.side_effect = [
            MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "key": "P1", "name": "P1"}])),
            MagicMock(status_code=200, json=MagicMock(return_value={"data": [{"id": 1, "name": "DS"}]})),
        ]
        result = await zephyr.get_datasets("P1")
        assert result[0]["name"] == "DS"


class TestConfluencePages:
    async def test_returns_pages(self, zephyr):
        zephyr._client.get.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"data": [{"id": 123, "title": "T"}]}),
        )
        result = await zephyr.get_confluence_pages([123])
        assert result[0]["title"] == "T"

    async def test_empty_ids_returns_empty(self, zephyr):
        assert await zephyr.get_confluence_pages([]) == []


class TestDownloadAttachment:
    async def test_returns_content(self, zephyr):
        import base64
        zephyr._client.get.return_value = MagicMock(
            status_code=200,
            content=b"hello",
            headers={
                "content-type": "application/octet-stream",
                "content-disposition": 'attachment; filename="f.txt"',
            },
        )
        result = await zephyr.download_attachment(1)
        assert result["name"] == "f.txt"
        assert result["size"] == 5
        assert base64.b64decode(result["content_base64"]) == b"hello"
        assert result["truncated"] is False

    async def test_truncation(self, zephyr):
        zephyr._client.get.return_value = MagicMock(
            status_code=200, content=b"x" * 100, headers={}
        )
        result = await zephyr.download_attachment(1, max_bytes=50)
        assert result["truncated"] is True
        assert result["size"] == 100

    async def test_404_returns_none(self, zephyr):
        zephyr._client.get.return_value = MagicMock(status_code=404)
        assert await zephyr.download_attachment(999) is None

    async def test_401_returns_none(self, zephyr):
        zephyr._client.get.return_value = MagicMock(status_code=401)
        assert await zephyr.download_attachment(999) is None


class TestClose:
    async def test_aclose_closes_http_client(self, zephyr):
        await zephyr.aclose()
        zephyr._client.aclose.assert_awaited_once()


class TestErrorPropagation:
    """Backend failures surface as ZephyrError with endpoint + server text."""

    async def test_search_400_carries_server_message(self, zephyr):
        zephyr._client.get.return_value = _http_response(400, json_body={
            "code": 400,
            "error": "InvalidTqlException",
            "messages": ["Invalid TQL Query: no viable alternative at input 'bogusAND'"],
        })
        with pytest.raises(ZephyrError) as exc_info:
            await zephyr.search_test_cases(tql='testCase.key IN ("P1-1")')
        msg = str(exc_info.value)
        assert "GET /rest/tests/1.0/testcase/search (HTTP 400)" in msg
        assert "Invalid TQL Query: no viable alternative at input 'bogusAND'" in msg

    async def test_search_500_raises(self, zephyr):
        zephyr._client.get.return_value = _http_response(500)
        with pytest.raises(ZephyrError, match=r"GET /rest/tests/1\.0/testcase/search \(HTTP 500\)"):
            await zephyr.search_test_cases(tql='testCase.key IN ("P1-1")')

    @pytest.mark.parametrize(
        "bad_query",
        [
            "bogus ~~~",  # unknown bare field
            'testCase.status = "Draft"',  # unsupported field (400 on server)
            'testCase.name ~ "x"',  # fuzzy is not supported
            "testCase.createdOn >= now()",  # function calls / date filters are not supported
        ],
    )
    async def test_invalid_tql_rejected_client_side(self, zephyr, bad_query):
        """Unknown fields never reach the server (client-side allow-list)."""
        zephyr._client.get = AsyncMock()
        with pytest.raises(ValueError, match="TQL|operator"):
            await zephyr.search_test_cases(tql=bad_query)
        zephyr._client.get.assert_not_called()

    async def test_get_test_case_500_raises(self, zephyr):
        zephyr._client.get.return_value = _http_response(500)
        with pytest.raises(ZephyrError, match=r"GET /rest/tests/1\.0/testcase/P1-123 \(HTTP 500\)"):
            await zephyr.get_test_case("P1-123")

    async def test_availability_500_raises(self, mock_client):
        zp = ZephyrClient(MagicMock(), base_url="http://j:8080", username="u", password="p")
        zp._client = mock_client(get=AsyncMock(return_value=_http_response(500)))
        with pytest.raises(ZephyrError, match="HTTP 500"):
            await zp.is_available()

    async def test_project_resolution_error_propagates(self, zephyr):
        zephyr._client.get.side_effect = httpx.HTTPStatusError(
            "err", request=MagicMock(), response=MagicMock(status_code=500)
        )
        with pytest.raises(ZephyrError, match=r"GET /rest/tests/1\.0/project"):
            await zephyr.get_test_case_statuses("P1")

    async def test_confluence_pages_500_raises(self, zephyr):
        zephyr._client.get.return_value = _http_response(500)
        with pytest.raises(ZephyrError, match=r"GET /rest/tests/1\.0/confluence"):
            await zephyr.get_confluence_pages([123])

    async def test_confluence_pages_404_skipped(self, zephyr):
        zephyr._client.get.return_value = _http_response(404)
        assert await zephyr.get_confluence_pages([123]) == []

    async def test_download_500_raises(self, zephyr):
        zephyr._client.get.return_value = _http_response(500)
        with pytest.raises(ZephyrError, match=r"GET /rest/tests/1\.0/attachment/999"):
            await zephyr.download_attachment(999)

    async def test_connect_error_shows_os_cause(self, zephyr):
        err = httpx.ConnectError("All connection attempts failed")
        err.__cause__ = ConnectionRefusedError(111, "Connect call failed ('127.0.0.1', 9)")
        zephyr._client.get = AsyncMock(side_effect=err)
        with pytest.raises(ZephyrError, match=r"127\.0\.0\.1.*9"):
            await zephyr.get_projects()


class TestCreatePlugins:
    def test_unknown_plugin_skipped(self):
        jira = MagicMock()
        result = create_plugins(jira, "nonexistent")
        assert result == {}

    def test_known_plugin_created(self):
        jira = MagicMock(spec=Jira)
        result = create_plugins(jira, "zephyr", base_url="http://j:8080", username="u", password="p")
        assert "zephyr" in result
        assert isinstance(result["zephyr"], ZephyrClient)

    def test_empty_string_returns_empty(self):
        jira = MagicMock()
        result = create_plugins(jira, "")
        assert result == {}

    def test_multiple_plugins(self):
        jira = MagicMock(spec=Jira)
        result = create_plugins(jira, "zephyr,zephyr", base_url="http://j:8080", username="u", password="p")
        assert "zephyr" in result


class TestTransportRetry:
    async def test_retry_on_remote_protocol_error(self, zephyr):
        ok = MagicMock(status_code=200, json=MagicMock(return_value=[{"id": 1, "key": "P1"}]))
        zephyr._client.get = AsyncMock(side_effect=[httpx.RemoteProtocolError("drop"), ok])
        result = await zephyr.get_projects()
        assert result == [{"id": 1, "key": "P1"}]
        assert zephyr._client.get.call_count == 2

    async def test_retry_exhausted_raises(self, zephyr):
        zephyr._client.get = AsyncMock(side_effect=httpx.RemoteProtocolError("drop"))
        with pytest.raises(ZephyrError, match="GET /rest/tests/1.0/project"):
            await zephyr.get_projects()
        assert zephyr._client.get.call_count == 2

    async def test_status_error_is_not_retried(self, zephyr):
        zephyr._client.get = AsyncMock(side_effect=httpx.HTTPStatusError(
            "err", request=MagicMock(), response=MagicMock(status_code=500),
        ))
        with pytest.raises(ZephyrError, match="HTTP 500"):
            await zephyr.get_projects()
        assert zephyr._client.get.call_count == 1

    async def test_connect_error_is_not_retried(self, zephyr):
        zephyr._client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(ZephyrError, match="GET /rest/tests/1.0/project"):
            await zephyr.get_projects()
        assert zephyr._client.get.call_count == 1

    async def test_availability_failure_is_not_latched(self, mock_client):
        zp = ZephyrClient(MagicMock(), base_url="http://j:8080", username="u", password="p")
        zp._client = mock_client(get=AsyncMock(side_effect=[
            httpx.ConnectError("unreachable"),
            MagicMock(status_code=200),
        ]))
        with pytest.raises(ZephyrError, match="GET /rest/tests/1.0/project"):
            await zp.is_available()
        assert zp._checked is False
        assert await zp.is_available() is True
        assert zp._checked is True
