"""Custom exceptions for atlassian_rag_mcp."""

from __future__ import annotations


class BaseError(RuntimeError):
    """Raised when a backend service fails (connection, availability, validation)."""


class AuthenticationError(BaseError):
    """Raised when any API authentication fails (401/403)."""


class CqlError(BaseError):
    """Raised when a CQL query fails on the Confluence API side (syntax, 4xx/5xx, network).

    The message always carries the failing query plus the API's own error text,
    so callers (and MCP clients) can see exactly what was sent and what the
    server answered — the raw SDK exception only says 'The query cannot be parsed'.
    """


class FtsError(BaseError):
    """Raised when an FTS (Tantivy FTS) query is rejected by the query parser.

    Mirror of CqlError/JqlError: the message carries the failing query plus the
    parser's own error text (e.g. "Syntax Error: text:(((", "Field does not
    exist: 'foo'"), so callers (and MCP clients) can see exactly what was
    sent and why it was rejected. Index/IO failures are NOT FtsQueryError —
    they are logged and reported as an empty result set.
    """


class JqlError(BaseError):
    """Raised when a JQL query fails on the Jira API side (syntax, 4xx/5xx, network).

    Mirror of CqlError: the message carries the failing JQL plus the server's
    own error text (the Jira SDK already puts it into the exception message,
    e.g. "Ошибка в запросе JQL: ... (строка 1, символ 15)").
    """


class ZephyrError(BaseError):
    """Raised when a Zephyr Scale API request fails (4xx/5xx, network).

    Mirror of CqlError/JqlError: the message carries the requested endpoint
    plus the server's own error text (the raw httpx exception only says
    'Client error ... for url ...' and hides the response body), so callers
    can see exactly what was requested and what the server answered.
    """
