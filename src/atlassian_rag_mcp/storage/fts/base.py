"""Abstract FTS index interface — backend-agnostic full-text search."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class FTSIndex(ABC):
    """Abstract interface for full-text search index backends."""

    async def start(self) -> None:
        ...

    @abstractmethod
    async def index_batch(self, docs: list[dict[str, Any]]) -> bool:
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    async def delete_by_filter(self, filter_condition: dict) -> bool:
        ...

    @abstractmethod
    async def clear(self) -> bool:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    @abstractmethod
    async def validate(self) -> None:
        ...
