"""JiraClient — main client composing IssueService, SearchService."""

from __future__ import annotations

from typing import Any

from atlassian import Jira
from requests.exceptions import ConnectionError as RequestsConnectionError

from atlassian_rag_mcp.exceptions import AuthenticationError
from atlassian_rag_mcp.jira.issues import IssueService
from atlassian_rag_mcp.jira.plugins import create_plugins
from atlassian_rag_mcp.jira.search import SearchService
from atlassian_rag_mcp.log.logger import logger


def _is_cloud(url: str) -> bool:
    """Check if Jira URL is a Cloud instance."""
    return "atlassian.net" in url


class JiraClient:
    """Main Jira client composing IssueService, SearchService."""

    def __init__(
        self,
        jira: Jira,
        project_key: str | None = None,
        plugins: dict[str, Any] | None = None,
    ) -> None:
        self.jira = jira
        self.project_key = project_key
        self.issues = IssueService(self.jira)
        self.search_service = SearchService(self.jira)
        self.plugins: dict[str, Any] = plugins if plugins is not None else create_plugins(jira)

    def validate(self, url: str = "") -> None:
        """Validate credentials by making a simple Jira API call."""
        try:
            logger.debug("Testing Jira credentials...")
            user = self.jira.myself()
            if user:
                display_name = user.get("displayName", "Unknown")
                logger.info(f"Jira OK — user: {display_name}")
            else:
                raise AuthenticationError(
                    "Jira API returned empty user — check JIRA_URL and credentials."
                )
        except RequestsConnectionError as e:
            raise AuthenticationError(
                f"Cannot connect to Jira at {url}. "
                f"Check JIRA_URL and network connectivity."
            ) from e
        except AuthenticationError:
            raise
        except Exception as e:
            raise AuthenticationError(
                f"Jira validation failed: {e}"
            ) from e

