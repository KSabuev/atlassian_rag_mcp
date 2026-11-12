"""CQL query service — sync SDK operations for Confluence content search."""

from __future__ import annotations

from typing import Any

from atlassian import Confluence

from atlassian_rag_mcp.confluence.models import CqlResult
from atlassian_rag_mcp.exceptions import CqlError
from atlassian_rag_mcp.log.logger import logger


class CqlService:
    """Sync service for Confluence CQL queries (mirrors Jira SearchService)."""

    def __init__(self, confluence: Confluence) -> None:
        self.confluence = confluence

    def cql(
        self, cql: str, start: int = 0, limit: int = 50, expand: str = "content.version",
    ) -> CqlResult:
        """Execute a CQL query against Confluence.

        Raises CqlError with the failing query and the server's own error text
        (the raw SDK exception only says 'The query cannot be parsed').
        """
        logger.debug(f"cql={cql[:80]} start={start} limit={limit}")
        try:
            raw = self.confluence.cql(cql=cql, start=start, limit=limit, expand=expand)
        except Exception as e:
            raise CqlError(self._describe(e, cql)) from e
        return CqlResult(
            results=raw.get("results", []),
            total=raw.get("totalSize", 0),
            query=cql,
        )

    @staticmethod
    def _describe(e: Exception, cql: str) -> str:
        """Build a human-readable error: the query, the HTTP code (when known)
        and the API's own message.

        The atlassian SDK wraps HTTP 400 as ApiValueError('The query cannot be
        parsed') with the original requests.HTTPError in ``reason``; the actual
        server message (which echoes the failing CQL) lives in the response body.
        """
        response: Any = getattr(e, "response", None)
        if response is None:
            response = getattr(getattr(e, "reason", None), "response", None)
        if response is not None:
            try:
                message = (response.json() or {}).get("message") or ""
            except Exception:
                message = (getattr(response, "text", "") or "").strip()
            if message:
                code = getattr(response, "status_code", None)
                suffix = f" (HTTP {code})" if code is not None else ""
                return f"CQL query failed: {cql}{suffix} — {message[:300]}"
        return f"CQL query failed: {cql} — {e}"
