"""Jira response models for MCP tool outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class IssueResult(BaseModel):
    issue_key: str
    fields: dict[str, Any] = {}
    error: str | None = None


class SearchResult(BaseModel):
    issues: list[dict[str, Any]] = []
    total: int = 0
    start_at: int = 0
    max_results: int = 0
    error: str | None = None


class TransitionResult(BaseModel):
    issue_key: str
    transitions: list[dict[str, Any]] = []
    error: str | None = None


class WorklogResult(BaseModel):
    issue_key: str
    worklogs: list[dict[str, Any]] = []
    error: str | None = None


class ListResult(BaseModel):
    items: list[dict[str, Any]] = []
    total: int = 0
    error: str | None = None


class JiraMethodResult(BaseModel):
    """Generic result of a Jira method executed via jira_execute_method."""
    method: str = ""
    data: Any = None
    error: str | None = None


class IssueTreeNode(BaseModel):
    """A node of the issue-link tree (parent/child issue keys)."""
    key: str
    children: list["IssueTreeNode"] = []


class IssueTreeResult(BaseModel):
    """Issue-link tree with its flat edges and total node count."""
    tree: IssueTreeNode
    edges: list[dict[str, str]] = []
    node_count: int = 0


IssueTreeNode.model_rebuild()
