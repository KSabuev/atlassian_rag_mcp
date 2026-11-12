"""Abstract VectorDB interface — weak coupling for switching between vector storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorDB(ABC):
    """Abstract interface for vector storage backends (LanceDB, Qdrant)."""

    @abstractmethod
    async def upsert_points(self, points: list[dict[str, Any]]) -> bool:
        ...

    @abstractmethod
    async def search(
        self,
        vector: list[float],
        limit: int = 25,
        query_filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    async def scroll_points(
        self,
        filter_condition: dict,
        limit: int = 10000,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        ...

    @abstractmethod
    async def delete_by_filter(self, filter_condition: dict) -> bool:
        ...

    @abstractmethod
    async def create_collection(
        self, vector_size: int | None = None, distance: str = "Cosine"
    ) -> bool:
        ...

    @abstractmethod
    async def collection_exists(self) -> bool:
        ...

    @abstractmethod
    async def get_collection_info(self) -> dict[str, Any] | None:
        ...

    @abstractmethod
    async def start(self) -> None:
        """Lazy initialization (I/O). Call after construction."""
        ...

    @abstractmethod
    async def validate(self) -> None:
        ...

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...

    async def get_page_versions_map(self) -> dict[str, int]:
        versions: dict[str, int] = {}
        offset: str | None = None
        page_size = 1000
        while True:
            points, next_offset = await self.scroll_points(
                filter_condition={}, limit=page_size, offset=offset
            )
            if not points:
                break
            for p in points:
                pay = p.get("payload", {})
                pid = pay.get("page_id", "")
                if not pid:
                    continue
                ver = pay.get("page_version") or 0
                if pid not in versions or versions[pid] < ver:
                    versions[pid] = ver
            if not next_offset:
                break
            offset = next_offset
        return versions
