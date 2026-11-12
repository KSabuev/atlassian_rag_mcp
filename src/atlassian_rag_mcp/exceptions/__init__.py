"""Custom exceptions for the application."""

from __future__ import annotations

from atlassian_rag_mcp.exceptions.errors import (
    AuthenticationError,
    BaseError,
    CqlError,
    FtsError,
    JqlError,
    ZephyrError,
)

__all__ = ["AuthenticationError", "BaseError", "CqlError", "FtsError", "JqlError", "ZephyrError"]
