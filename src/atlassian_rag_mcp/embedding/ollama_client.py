"""Ollama embedding provider — local embeddings via Ollama API."""

from __future__ import annotations

import httpx

from atlassian_rag_mcp.embedding.base import EmbeddingProvider
from atlassian_rag_mcp.exceptions import BaseError
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.utils.http_retry import post_with_retry


class OllamaClient(EmbeddingProvider):
    """Ollama embedding provider via /api/embed."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        model: str,
        dim: int = 2560,
    ) -> None:
        self._client = client
        self.base_url = url.rstrip("/")
        self.model_name = model
        self.embedding_dim: int = dim

    async def validate(self) -> None:
        try:
            response = await post_with_retry(
                self._client,
                f"{self.base_url}/api/embed",
                headers={"Content-Type": "application/json"},
                json={"model": self.model_name, "input": ["health check probe"]},
                timeout=10.0,
            )
            if response.status_code == 404:
                raise BaseError(
                    f"Ollama embedding model '{self.model_name}' not found at {self.base_url}. "
                    f"Pull it first: ollama pull {self.model_name}"
                )
            if response.status_code != 200:
                raise BaseError(
                    f"Ollama at {self.base_url} returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
            data = response.json()
            embeddings = data.get("embeddings", [])
            if not embeddings:
                raise BaseError("Ollama returned empty embeddings")
            dim = len(embeddings[0])
            self.embedding_dim = dim
            logger.info(f"Embedding OK — model={self.model_name}, dim={dim}, provider=ollama")
        except httpx.ConnectError:
            raise BaseError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Check EMBEDDING_URL."
            )
        except httpx.TimeoutException:
            raise BaseError(
                f"Ollama at {self.base_url} timed out after 10s. "
                f"Check network connectivity."
            )
        except BaseError:
            raise
        except Exception as e:
            raise BaseError(f"Ollama validation failed: {e}") from e

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await post_with_retry(
                self._client,
                f"{self.base_url}/api/embed",
                headers={"Content-Type": "application/json"},
                json={"model": self.model_name, "input": texts},
            )
            if response.status_code == 200:
                return response.json().get("embeddings", [])
            raise BaseError(
                f"Ollama API error: {response.status_code} - {response.text[:500]}"
            )
        except Exception as e:
            logger.error(f"Ollama request failed: {e}")
            raise

    async def get_embedding(self, text: str) -> list[float]:
        results = await self._embed([text])
        return results[0] if results else []

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def get_embedding_dimension(self) -> int:
        return self.embedding_dim

    async def close(self) -> None:
        await self._client.aclose()
