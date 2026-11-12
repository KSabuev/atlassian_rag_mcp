"""Search service for Jira JQL operations."""

from __future__ import annotations

from atlassian import Jira

from atlassian_rag_mcp.exceptions import JqlError
from atlassian_rag_mcp.jira.constants import SEARCH_FIELDS
from atlassian_rag_mcp.jira.jql_builder import JQLBuilder
from atlassian_rag_mcp.jira.models import SearchResult
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.registry import error_message


def jql_error_message(e: Exception, jql: str) -> str:
    """Build a human-readable JQL failure: the query, the HTTP code (when known)
    and the server's own error text (the Jira SDK already puts it into the
    exception message, e.g. "Ошибка в запросе JQL: ... (строка 1, символ 15)")."""
    code = getattr(getattr(e, "response", None), "status_code", None)
    suffix = f" (HTTP {code})" if code is not None else ""
    return f"JQL query failed: {jql}{suffix} — {error_message(e)}"


class SearchService:
    """Service for Jira search operations."""

    def __init__(self, jira: Jira) -> None:
        self.jira = jira

    def search_issues(
        self,
        jql: str,
        fields: str = SEARCH_FIELDS,
        limit: int = 25,
        start_at: int = 0,
    ) -> SearchResult:
        """Search issues using JQL (Jira Query Language).

        Raises JqlError with the failing JQL and the server's own error text
        (the raw SDK exception only says e.g. 'Ошибка в запросе JQL: ...'
        without the query itself).
        """
        logger.info(f"JQL -> Jira: {jql}")
        try:
            response = self.jira.jql(
                jql,
                fields=fields,
                start=start_at,
                limit=limit,
            )
        except Exception as e:
            raise JqlError(jql_error_message(e, jql)) from e
        if not isinstance(response, dict):
            return SearchResult(start_at=start_at, max_results=limit, error="Invalid response from Jira API")

        issues = response.get("issues", [])
        total = response.get("total", 0)

        return SearchResult(issues=issues, total=total, start_at=start_at, max_results=limit)

    def get_project_issues(
        self,
        project_key: str,
        fields: str = SEARCH_FIELDS,
        limit: int = 25,
        start_at: int = 0,
    ) -> SearchResult:
        """Get issues for a specific project."""
        jql = JQLBuilder().project(project_key).order_by("created", "DESC").build()
        return self.search_issues(jql, fields=fields, limit=limit, start_at=start_at)

