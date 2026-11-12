"""Service for Jira issue operations."""

from __future__ import annotations

from typing import Any

from atlassian import Jira

from atlassian_rag_mcp.exceptions import JqlError
from atlassian_rag_mcp.jira.constants import EXPAND_FIELDS, ISSUE_FIELDS, SEARCH_FIELDS
from atlassian_rag_mcp.jira.models import (
    IssueResult,
    IssueTreeNode,
    IssueTreeResult,
    ListResult,
    SearchResult,
    TransitionResult,
    WorklogResult,
)
from atlassian_rag_mcp.jira.search import jql_error_message
from atlassian_rag_mcp.log.logger import logger


def build_issue_tree(pairs: list[dict], root_key: str) -> dict:
    """Build a nested tree from the SDK's flat {parent: child} edge list.

    The topmost node (never appearing as a child) becomes the root; if the
    requested root_key is topmost, it stays on top. Cycles are broken by
    skipping already-visited keys.
    """
    children: dict[str, list[str]] = {}
    for pair in pairs:
        for parent, child in pair.items():
            children.setdefault(parent, []).append(child)

    all_children = {c for kids in children.values() for c in kids}
    candidates = [k for k in children if k not in all_children]
    if root_key in candidates:
        top = root_key
    elif candidates:
        top = candidates[0]
    else:
        top = root_key

    def build(key: str, visited: set[str]) -> dict:
        visited.add(key)
        kids = []
        for c in children.get(key, []):
            if c not in visited:
                kids.append(build(c, visited))
        return {"key": key, "children": kids}

    def count(node: dict) -> int:
        return 1 + sum(count(k) for k in node.get("children", []))

    tree = build(top, set())
    edges = [{"from": p, "to": c} for pair in pairs for p, c in pair.items()]
    return IssueTreeResult(
        tree=IssueTreeNode.model_validate(tree),
        edges=edges,
        node_count=count(tree),
    ).model_dump()

class IssueService:
    """Service for Jira issue operations."""

    def __init__(self, jira: Jira) -> None:
        self.jira = jira

    def get_issue(
        self,
        issue_key: str,
        fields: str = ISSUE_FIELDS,
        expand: str = EXPAND_FIELDS,
        comment_limit: int = 10,
    ) -> IssueResult:
        """Get a Jira issue by key."""
        issue = self.jira.issue(
            issue_key,
            fields=fields,
            expand=expand,
        )
        if not issue:
            return IssueResult(issue_key=issue_key, error=f"Issue {issue_key} not found")

        if comment_limit > 0:
            try:
                comments_resp = self.jira.issue_get_comments(issue_key)
                comments = comments_resp.get("comments", []) if isinstance(comments_resp, dict) else []
                issue["comments"] = comments[:comment_limit]
            except Exception as e:
                logger.warning(f"Could not fetch comments for {issue_key}: {e}")
                issue["comments"] = []

        return IssueResult(issue_key=issue_key, fields=issue)

    def get_issue_types(self, project_key: str) -> ListResult:
        """Get issue types available for a project."""
        meta = self.jira.issue_createmeta(
            project=project_key,
            expand="projects.issuetypes",
        )
        projects = meta.get("projects", [])
        if not projects:
            return ListResult()
        items = projects[0].get("issuetypes", [])
        return ListResult(items=items, total=len(items))

    def get_priorities(self) -> ListResult:
        """Get all Jira priorities."""
        items = self.jira.get_all_priorities()
        if not isinstance(items, list):
            return ListResult()
        return ListResult(items=items, total=len(items))

    def get_resolutions(self) -> ListResult:
        """Get all Jira resolutions."""
        items = self.jira.get_all_resolutions()
        if not isinstance(items, list):
            return ListResult()
        return ListResult(items=items, total=len(items))

    def get_project_statuses(self, project_key: str) -> ListResult:
        """Get unique statuses used in a project."""
        data = self.jira.get_status_for_project(project_key)
        if not isinstance(data, list):
            return ListResult()
        seen: set[str] = set()
        items: list[dict[str, Any]] = []
        for entry in data:
            for s in entry.get("statuses", []):
                sid = s.get("id")
                if sid and sid not in seen:
                    seen.add(sid)
                    items.append({"id": sid, "name": s.get("name", "")})
        return ListResult(items=items, total=len(items))

    def get_transitions(self, issue_key: str) -> TransitionResult:
        """Get available status transitions for an issue."""
        transitions = self.jira.get_issue_transitions(issue_key)
        if not isinstance(transitions, list):
            return TransitionResult(issue_key=issue_key)
        return TransitionResult(issue_key=issue_key, transitions=transitions)

    def get_worklog(self, issue_key: str) -> WorklogResult:
        """Get worklog entries for an issue."""
        worklog = self.jira.issue_get_worklog(issue_key)
        if isinstance(worklog, dict):
            items = worklog.get("worklogs", [])
            return WorklogResult(issue_key=issue_key, worklogs=items)
        return WorklogResult(issue_key=issue_key)

    def get_comments(self, issue_key: str, limit: int = 10, start_at: int = 0) -> ListResult:
        """Get comments for an issue. SDK returns all comments at once,
        so limit/start_at are applied client-side."""
        resp = self.jira.issue_get_comments(issue_key)
        if isinstance(resp, dict):
            items = resp.get("comments", [])
            total = resp.get("total", len(items))
            if limit > 0:
                items = items[start_at:start_at + limit]
            return ListResult(items=items, total=total)
        return ListResult()

    def get_changelog(self, issue_key: str) -> IssueResult:
        """Get the changelog (history of field changes) for an issue."""
        issue = self.jira.issue(issue_key, expand="changelog")
        if not issue:
            return IssueResult(issue_key=issue_key, error=f"Issue {issue_key} not found")
        return IssueResult(issue_key=issue_key, fields={"changelog": issue.get("changelog", {})})

    def get_projects(self) -> ListResult:
        """Get all Jira projects accessible to the current user."""
        projects = self.jira.get_all_projects()
        if not isinstance(projects, list):
            return ListResult()
        return ListResult(items=projects, total=len(projects))

    def get_issue_tree(self, issue_key: str, depth: int | None = None) -> IssueResult:
        """Get the issue context graph: subtasks and linked issues (incl. parents/epics)."""
        pairs = self.jira.get_issue_tree_recursive(issue_key, depth=depth)
        if not isinstance(pairs, list):
            return IssueResult(issue_key=issue_key, error="Invalid response from Jira API")
        return IssueResult(issue_key=issue_key, fields=build_issue_tree(pairs, issue_key))

    def get_issue_labels(self, issue_key: str) -> ListResult:
        """Get labels of an issue (normalized to {"name": ...} dicts)."""
        resp = self.jira.get_issue_labels(issue_key)
        items = resp if isinstance(resp, list) else []
        items = [item if isinstance(item, dict) else {"name": item} for item in items]
        return ListResult(items=items, total=len(items))

    def get_issue_attachments(self, issue_key: str) -> ListResult:
        """Get issue attachments with metadata and content URLs."""
        ids = self.jira.get_attachments_ids_from_issue(issue_key)
        items: list[dict[str, Any]] = []
        for entry in ids or []:
            aid = entry.get("id") if isinstance(entry, dict) else entry
            if aid is None:
                continue
            try:
                detail = self.jira.get_attachment(aid)
            except Exception as e:
                logger.warning(f"Could not fetch attachment {aid}: {e}")
                continue
            if isinstance(detail, dict):
                items.append(detail)
            else:
                items.append({"id": str(aid)})
        return ListResult(items=items, total=len(items))

    def get_issue_remote_links(
        self,
        issue_key: str,
        global_id: str | None = None,
        internal_id: str | None = None,
    ) -> ListResult:
        """Get remote links on an issue (links to external systems)."""
        resp = self.jira.get_issue_remote_links(issue_key, global_id=global_id, internal_id=internal_id)
        items = resp if isinstance(resp, list) else []
        return ListResult(items=items, total=len(items))

    def get_issue_watchers(self, issue_key: str) -> ListResult:
        """Get watchers of an issue."""
        resp = self.jira.issue_get_watchers(issue_key)
        items = resp if isinstance(resp, list) else []
        return ListResult(items=items, total=len(items))

    def issue_exists(self, issue_key: str) -> bool:
        """Check whether an issue exists."""
        return bool(self.jira.issue_exists(issue_key))

    def count_issues(self, jql: str) -> int:
        """Issue count for a JQL query.

        Cloud: approximate_issue_count. DC/Server: search with limit=1 and read total.
        """
        if getattr(self.jira, "cloud", False):
            try:
                n = self.jira.approximate_issue_count(jql)
                return int(n) if n is not None else 0
            except Exception:
                pass
        try:
            resp = self.jira.jql(jql, fields="key", limit=1)
        except Exception as e:
            raise JqlError(jql_error_message(e, jql)) from e
        if isinstance(resp, dict):
            return int(resp.get("total", 0))
        return 0

    def get_boards(
        self,
        project_key: str | None = None,
        board_name: str | None = None,
        board_type: str | None = None,
        limit: int = 50,
        start_at: int = 0,
    ) -> ListResult:
        """Get agile boards (optionally filtered by project)."""
        resp = self.jira.get_all_agile_boards(
            board_name=board_name, project_key=project_key,
            board_type=board_type, start=start_at, limit=limit,
        )
        values = resp.get("values", []) if isinstance(resp, dict) else []
        items = [
            {
                "id": b.get("id"),
                "name": b.get("name"),
                "type": b.get("type"),
                "project_key": (b.get("project") or {}).get("key") if isinstance(b, dict) else None,
                "location": b.get("location") if isinstance(b, dict) else None,
            }
            for b in values
        ]
        total = resp.get("size", len(values)) if isinstance(resp, dict) else len(values)
        return ListResult(items=items, total=total)

    def resolve_board_id(self, project_key: str) -> str | None:
        """Resolve a board ID for a project. Prefers scrum boards (sprints/epics
        are only available there); falls back to the first board."""
        boards = self.get_boards(project_key=project_key, limit=50)
        if not boards.items:
            return None
        for b in boards.items:
            if b.get("type") == "scrum":
                return b.get("id")
        return boards.items[0].get("id")

    def get_epics(self, board_id: str, done: bool = False, limit: int = 50, start_at: int = 0) -> ListResult:
        """Get epics of a board."""
        resp = self.jira.get_epics(board_id, done=done, start=start_at, limit=limit)
        values = resp.get("values", []) if isinstance(resp, dict) else []
        total = resp.get("total", len(values)) if isinstance(resp, dict) else len(values)
        return ListResult(items=values, total=total)

    def get_issues_for_epic(
        self,
        epic_id: str,
        fields: str = SEARCH_FIELDS,
        limit: int = 50,
        start_at: int = 0,
    ) -> SearchResult:
        """Get issues belonging to an epic. Board-independent endpoint
        (the board-scoped one 404s on some DC instances); limit/start_at
        are applied client-side."""
        resp = self.jira.epic_issues(epic_id, fields=fields)
        issues = resp.get("issues", []) if isinstance(resp, dict) else []
        total = resp.get("total", len(issues)) if isinstance(resp, dict) else 0
        if limit > 0:
            issues = issues[start_at:start_at + limit]
        return SearchResult(issues=issues, total=total, start_at=start_at, max_results=limit)

    def get_sprints(self, board_id: str, state: str | None = None, limit: int = 50, start_at: int = 0) -> ListResult:
        """Get sprints of a board (optionally filtered by state)."""
        resp = self.jira.get_all_sprints_from_board(board_id, state=state, start=start_at, limit=limit)
        values = resp.get("values", []) if isinstance(resp, dict) else []
        total = resp.get("total", len(values)) if isinstance(resp, dict) else len(values)
        return ListResult(items=values, total=total)

    def get_sprint_issues(self, sprint_id: str, start_at: int = 0, limit: int = 25) -> SearchResult:
        """Get issues of a sprint."""
        resp = self.jira.get_sprint_issues(sprint_id, start=start_at, limit=limit)
        issues = resp.get("issues", []) if isinstance(resp, dict) else []
        total = resp.get("total", len(issues)) if isinstance(resp, dict) else 0
        return SearchResult(issues=issues, total=total, start_at=start_at, max_results=limit)

    def get_project(self, project_key: str, expand: str | None = None) -> dict:
        """Get details of a single project."""
        project = self.jira.get_project(project_key, expand=expand)
        if not project:
            raise ValueError(f"Project {project_key} not found")
        return project

    def get_project_components(self, project_key: str) -> ListResult:
        """Get components of a project."""
        resp = self.jira.get_project_components(project_key)
        items = resp if isinstance(resp, list) else []
        return ListResult(items=items, total=len(items))

    def get_project_versions(self, project_key: str, expand: str | None = None) -> ListResult:
        """Get versions of a project."""
        resp = self.jira.get_project_versions(project_key, expand=expand)
        items = resp if isinstance(resp, list) else []
        return ListResult(items=items, total=len(items))

    def get_all_statuses(self) -> ListResult:
        """Get all statuses of the Jira instance."""
        resp = self.jira.get_all_statuses()
        items = resp if isinstance(resp, list) else []
        return ListResult(items=items, total=len(items))

    def get_fields(self) -> ListResult:
        """Get all Jira fields (id, name, type, custom)."""
        resp = self.jira.get_all_fields()
        if isinstance(resp, dict):
            items = [
                {
                    "id": f.get("id"),
                    "key": f.get("key"),
                    "name": f.get("name"),
                    "custom": f.get("custom"),
                    "schema": (f.get("schema") or {}).get("type") if isinstance(f, dict) else None,
                }
                for f in resp.values() if isinstance(f, dict)
            ]
        else:
            items = resp if isinstance(resp, list) else []
        return ListResult(items=items, total=len(items))

    def get_last_issue_number(self, project_key: str) -> int:
        """Get the last (highest) issue number of a project.

        DC/Server returns an issue key string (e.g. 'PROJ-1005');
        Cloud returns a dict with 'issueNumber'. The SDK crashes on projects
        without issues (empty 'issues' list) — those yield 0.
        """
        try:
            resp = self.jira.get_project_issuekey_last(project_key)
        except (KeyError, IndexError, TypeError):
            return 0
        if isinstance(resp, str):
            tail = resp.rsplit("-", 1)[-1]
            return int(tail) if tail.isdigit() else 0
        if isinstance(resp, dict):
            return int(resp.get("issueNumber", 0))
        return 0

    def get_users(self, limit: int = 50, start_at: int = 0) -> ListResult:
        """Get all users (paginated).

        Cloud: /user listing. DC/Server: 'username=.' search (listing is not available).
        """
        if getattr(self.jira, "cloud", False):
            users = self.jira.users_get_all(start=start_at, limit=limit)
        else:
            users = self.jira.user_find_by_user_string(username=".", start=start_at, limit=limit)
        items = users if isinstance(users, list) else []
        return ListResult(items=items, total=len(items))

    def find_user(
        self,
        query: str | None = None,
        username: str | None = None,
        account_id: str | None = None,
        include_inactive_users: bool = False,
        limit: int = 50,
        start_at: int = 0,
    ) -> ListResult:
        """Find users by query/username/account_id.

        Cloud: fuzzy query / account_id. DC/Server: exact username
        (query is mapped to username search).
        """
        if getattr(self.jira, "cloud", False):
            users = self.jira.user_find_by_user_string(
                username=username, query=query, account_id=account_id,
                include_inactive_users=include_inactive_users, start=start_at, limit=limit,
            )
        else:
            users = self.jira.user_find_by_user_string(
                username=username or query,
                include_inactive_users=include_inactive_users, start=start_at, limit=limit,
            )
        items = users if isinstance(users, list) else []
        return ListResult(items=items, total=len(items))

    def get_groups(self, query: str | None = None, exclude: str | None = None, limit: int = 20) -> ListResult:
        """Get groups (optionally filtered by name query)."""
        resp = self.jira.get_groups(query=query, exclude=exclude, limit=limit)
        items = resp if isinstance(resp, list) else []
        items = [g if isinstance(g, dict) else {"name": g} for g in items]
        return ListResult(items=items, total=len(items))

    def get_project_users(self, project_key: str, limit: int = 50, start_at: int = 0) -> ListResult:
        """Get users assignable in a project."""
        users = self.jira.get_all_assignable_users_for_project(project_key, start=start_at, limit=limit)
        items = users if isinstance(users, list) else []
        return ListResult(items=items, total=len(items))
