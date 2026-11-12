"""Shared Jira field-list defaults — single source of truth."""

from __future__ import annotations

ISSUE_FIELDS = "summary,status,assignee,description,priority,issuetype,created,updated,labels,reporter"
SEARCH_FIELDS = "summary,status,assignee,priority,issuetype,created,updated"
EXPAND_FIELDS = "renderedFields,changelog"
