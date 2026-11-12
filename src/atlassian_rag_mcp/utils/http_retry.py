"""Shared tenacity retry configuration for transient transport errors.

Single retry policy for the whole project:
- Retry ONLY transient transport errors: a connection dropped mid-response
  (the vLLM API server is known to drop connections under burst load and
  recover within seconds; Jira/Confluence do the same under load).
- Do NOT retry: timeouts, connection refused (a stuck/down server stays
  stuck), DNS failures and HTTP status errors.
- 2 attempts (1 base + 1 retry), fixed 1s delay, original exception
  re-raised after the last attempt.
"""

from __future__ import annotations

import http.client

import httpx
import requests
import urllib3.exceptions
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from atlassian_rag_mcp.log.logger import logger

HTTPX_RETRYABLE_ERRORS = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)

_REQUESTS_MIDSTREAM_ERRORS = (
    urllib3.exceptions.ProtocolError,
    http.client.IncompleteRead,
    urllib3.exceptions.ResponseError,
)


def is_transient_sdk_error(exc: BaseException) -> bool:
    """True when an atlassian SDK (requests) call failed due to a mid-response drop.

    Walks the exception chain (``__cause__``/``__context__`` and
    ``MaxRetryError.reason``) looking for urllib3 mid-stream errors.
    Connection refused, DNS failures, timeouts and HTTP status errors return
    False.
    """
    if not isinstance(
        exc,
        (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError),
    ):
        return False

    cur: BaseException | None = exc
    for _ in range(10):
        if cur is None:
            return False
        if isinstance(cur, _REQUESTS_MIDSTREAM_ERRORS):
            return True
        if isinstance(cur, urllib3.exceptions.MaxRetryError):
            cur = cur.reason
        else:
            cur = cur.__cause__ or cur.__context__
    return False


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(f"Transient transport error, retrying: {exc}")


def httpx_transport_retry() -> "retry":
    """Tenacity configuration for async httpx calls (1 base attempt + 1 retry)."""
    return retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1.0),
        retry=retry_if_exception_type(HTTPX_RETRYABLE_ERRORS),
        before_sleep=_log_retry,
        reraise=True,
    )


def sdk_transport_retry() -> "retry":
    """Tenacity configuration for sync atlassian SDK calls (1 base attempt + 1 retry)."""
    return retry(
        stop=stop_after_attempt(2),
        wait=wait_fixed(1.0),
        retry=retry_if_exception(is_transient_sdk_error),
        before_sleep=_log_retry,
        reraise=True,
    )


@httpx_transport_retry()
async def post_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    json: dict,
    timeout: float | None = None,
) -> httpx.Response:
    """POST with a single retry on transient transport errors."""
    return await client.post(url, headers=headers, json=json, timeout=timeout)
