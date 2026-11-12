"""Health check models."""

from __future__ import annotations

from pydantic import BaseModel


class ServiceStatus(BaseModel):
    status: str
    dimension: int | None = None
    model: str | None = None
    error: str | None = None


class HealthCheckResult(BaseModel):
    status: str
    services: dict[str, ServiceStatus]
