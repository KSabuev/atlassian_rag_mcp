"""Tests for the shared transport retry helpers (http_retry module)."""

from __future__ import annotations

import http.client
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import requests
import urllib3.exceptions

import atlassian_rag_mcp.confluence.attachments.download as download_mod
from atlassian_rag_mcp.confluence.attachments.download import download_attachment
from atlassian_rag_mcp.confluence.attachments.errors import AttachmentTooLarge
from atlassian_rag_mcp.utils.http_retry import (
    is_transient_sdk_error,
    post_with_retry,
    sdk_transport_retry,
)


def _raise_wrapped(inner_factory, wrapper_factory) -> BaseException:
    """Build an exception the way requests wraps urllib3 errors:
    the inner is raised inside an except block, so it lands in __context__."""
    try:
        try:
            raise inner_factory()
        except Exception as e:
            raise wrapper_factory(e)
    except Exception as exc:
        return exc


def _midstream_chunked() -> BaseException:
    return _raise_wrapped(
        lambda: urllib3.exceptions.ProtocolError("Connection broken"),
        requests.exceptions.ChunkedEncodingError,
    )


def _midstream_connection() -> BaseException:
    return _raise_wrapped(
        lambda: urllib3.exceptions.ProtocolError("reset by peer"),
        requests.exceptions.ConnectionError,
    )


def _refused() -> BaseException:
    return _raise_wrapped(
        lambda: urllib3.exceptions.NewConnectionError(None, "connection refused"),
        requests.exceptions.ConnectionError,
    )





class TestIsTransientSdkError:
    def test_midstream_chunked_encoding_is_retried(self):
        assert is_transient_sdk_error(_midstream_chunked()) is True

    def test_midstream_connection_error_is_retried(self):
        assert is_transient_sdk_error(_midstream_connection()) is True

    def test_incomplete_read_is_retried(self):
        exc = _raise_wrapped(
            lambda: http.client.IncompleteRead(10, 100),
            requests.exceptions.ConnectionError,
        )
        assert is_transient_sdk_error(exc) is True

    def test_urllib3_incomplete_read_wrapper_is_retried(self):
        exc = _raise_wrapped(
            lambda: urllib3.exceptions.IncompleteRead(10, 100),
            requests.exceptions.ConnectionError,
        )
        assert is_transient_sdk_error(exc) is True

    def test_connection_refused_is_not_retried(self):
        assert is_transient_sdk_error(_refused()) is False

    def test_timeouts_are_not_retried(self):
        assert is_transient_sdk_error(requests.exceptions.ReadTimeout("slow")) is False
        assert is_transient_sdk_error(requests.exceptions.ConnectTimeout("slow")) is False

    def test_status_errors_are_not_retried(self):
        assert is_transient_sdk_error(requests.exceptions.HTTPError("503")) is False

    def test_unrelated_errors_are_not_retried(self):
        assert is_transient_sdk_error(ValueError("boom")) is False
        assert is_transient_sdk_error(KeyError("x")) is False





class TestPostWithRetry:
    async def test_retry_then_success(self):
        post = AsyncMock(side_effect=[httpx.RemoteProtocolError("drop"), "OK"])
        client = MagicMock()
        client.post = post
        resp = await post_with_retry(client, "http://x", headers={}, json={"a": 1})
        assert resp == "OK"
        assert post.call_count == 2

    async def test_read_and_write_errors_are_retried(self):
        for err in (httpx.ReadError("r"), httpx.WriteError("w")):
            post = AsyncMock(side_effect=[err, "OK"])
            client = MagicMock()
            client.post = post
            resp = await post_with_retry(client, "http://x", headers={}, json={})
            assert resp == "OK"
            assert post.call_count == 2

    async def test_retry_exhausted_reraises_original(self):
        post = AsyncMock(side_effect=httpx.RemoteProtocolError("drop"))
        client = MagicMock()
        client.post = post
        with pytest.raises(httpx.RemoteProtocolError):
            await post_with_retry(client, "http://x", headers={}, json={})
        assert post.call_count == 2

    async def test_timeout_is_not_retried(self):
        post = AsyncMock(side_effect=httpx.ReadTimeout("slow"))
        client = MagicMock()
        client.post = post
        with pytest.raises(httpx.ReadTimeout):
            await post_with_retry(client, "http://x", headers={}, json={})
        assert post.call_count == 1

    async def test_connect_error_is_not_retried(self):
        post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        client = MagicMock()
        client.post = post
        with pytest.raises(httpx.ConnectError):
            await post_with_retry(client, "http://x", headers={}, json={})
        assert post.call_count == 1

    async def test_timeout_kwarg_forwarded(self):
        post = AsyncMock(return_value="OK")
        client = MagicMock()
        client.post = post
        await post_with_retry(client, "http://x", headers={}, json={}, timeout=10.0)
        post.assert_awaited_once_with("http://x", headers={}, json={}, timeout=10.0)





class TestSdkTransportRetry:
    def test_retry_then_success(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] == 1:
                raise _midstream_chunked()
            return "ok"

        assert sdk_transport_retry()(flaky)() == "ok"
        assert calls["n"] == 2

    def test_retry_exhausted_reraises_original(self):
        calls = {"n": 0}

        def always_fails():
            calls["n"] += 1
            raise _midstream_chunked()

        with pytest.raises(requests.exceptions.ChunkedEncodingError):
            sdk_transport_retry()(always_fails)()
        assert calls["n"] == 2

    def test_connection_refused_is_not_retried(self):
        calls = {"n": 0}

        def always_fails():
            calls["n"] += 1
            raise _refused()

        with pytest.raises(requests.exceptions.ConnectionError):
            sdk_transport_retry()(always_fails)()
        assert calls["n"] == 1

    def test_timeout_is_not_retried(self):
        calls = {"n": 0}

        def always_fails():
            calls["n"] += 1
            raise requests.exceptions.ReadTimeout("slow")

        with pytest.raises(requests.exceptions.ReadTimeout):
            sdk_transport_retry()(always_fails)()
        assert calls["n"] == 1

    def test_status_error_is_not_retried(self):
        calls = {"n": 0}

        def always_fails():
            calls["n"] += 1
            raise requests.exceptions.HTTPError("503")

        with pytest.raises(requests.exceptions.HTTPError):
            sdk_transport_retry()(always_fails)()
        assert calls["n"] == 1





class _Outcome:
    def __init__(self, status_code: int = 200, chunks: list | None = None, exc: BaseException | None = None):
        self.status_code = status_code
        self.chunks = chunks or []
        self.exc = exc


class _FakeStreamClient:
    """Stands in for httpx.AsyncClient: yields canned stream outcomes per call."""

    def __init__(self, outcomes: list[_Outcome]):
        self.outcomes = outcomes
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def stream(self, method: str, url: str):
        self.calls += 1
        outcome = self.outcomes[self.calls - 1]

        class _Resp:
            status_code = outcome.status_code

            async def aiter_bytes(self, size):
                if outcome.exc is not None:
                    raise outcome.exc
                for chunk in outcome.chunks:
                    yield chunk

        class _Ctx:
            async def __aenter__(self):
                return _Resp()

            async def __aexit__(self, *args):
                return False

        return _Ctx()


class TestDownloadAttachmentRetry:
    async def test_retry_on_midstream_drop(self, monkeypatch):
        fake = _FakeStreamClient(
            [
                _Outcome(exc=httpx.RemoteProtocolError("drop")),
                _Outcome(chunks=[b"da", b"ta"]),
            ]
        )
        monkeypatch.setattr(download_mod.httpx, "AsyncClient", lambda *a, **k: fake)
        data = await download_attachment("http://c/dl", max_bytes=1024)
        assert data == b"data"
        assert fake.calls == 2

    async def test_exhausted_reraises(self, monkeypatch):
        fake = _FakeStreamClient(
            [
                _Outcome(exc=httpx.ReadError("drop")),
                _Outcome(exc=httpx.ReadError("drop")),
            ]
        )
        monkeypatch.setattr(download_mod.httpx, "AsyncClient", lambda *a, **k: fake)
        with pytest.raises(httpx.ReadError):
            await download_attachment("http://c/dl", max_bytes=1024)
        assert fake.calls == 2

    async def test_too_large_is_not_retried(self, monkeypatch):
        fake = _FakeStreamClient([_Outcome(chunks=[b"x" * 100])])
        monkeypatch.setattr(download_mod.httpx, "AsyncClient", lambda *a, **k: fake)
        with pytest.raises(AttachmentTooLarge):
            await download_attachment("http://c/dl", max_bytes=10)
        assert fake.calls == 1

    async def test_http_status_error_is_not_retried(self, monkeypatch):
        fake = _FakeStreamClient([_Outcome(status_code=500, chunks=[b"err"])])
        monkeypatch.setattr(download_mod.httpx, "AsyncClient", lambda *a, **k: fake)
        with pytest.raises(RuntimeError, match="HTTP 500"):
            await download_attachment("http://c/dl", max_bytes=1024)
        assert fake.calls == 1





def _flaky_session(calls: dict, ok_response):
    def flaky_session_request(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _midstream_chunked()
        return ok_response

    return flaky_session_request


class TestFactoryWrapping:
    def test_confluence_client_retries_midstream_drops(self):
        from atlassian_rag_mcp.confluence import create_confluence_client

        client = create_confluence_client("http://t:8000", "u", "p", timeout=5)
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.headers = {}
        ok_response.text = "{}"
        calls = {"n": 0}
        client.confluence._session.request = _flaky_session(calls, ok_response)
        result = client.confluence.request("GET", "rest/api/content/1")
        assert result is ok_response
        assert calls["n"] == 2

    def test_jira_client_retries_midstream_drops(self):
        from atlassian_rag_mcp.jira import create_jira_client

        client = create_jira_client("http://t:8000", "u", "p", timeout=5)
        ok_response = MagicMock()
        ok_response.status_code = 200
        ok_response.headers = {}
        ok_response.text = "{}"
        calls = {"n": 0}
        client.jira._session.request = _flaky_session(calls, ok_response)
        result = client.jira.request("GET", "rest/api/2/issue/PROJ-1")
        assert result is ok_response
        assert calls["n"] == 2
