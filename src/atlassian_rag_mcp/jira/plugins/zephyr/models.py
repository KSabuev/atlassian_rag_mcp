"""Zephyr response models for MCP tool outputs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ZephyrMethodResult(BaseModel):
    """Generic result of a Zephyr method executed via zephyr_execute_method."""
    method: str = ""
    data: Any = None
    error: str | None = None
