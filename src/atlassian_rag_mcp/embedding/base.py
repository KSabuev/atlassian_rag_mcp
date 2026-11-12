"""Abstract embedding provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    embedding_dim: int
    model_name: str

    @abstractmethod
    async def validate(self) -> None:
        ...

    @abstractmethod
    async def get_embedding(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    async def get_embedding_dimension(self) -> int:
        ...

    @abstractmethod
    async def close(self) -> None:
        ...
