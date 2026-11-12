"""OpenAI-compatible embedding provider — v1/embeddings API."""

from __future__ import annotations

import time

import httpx

from atlassian_rag_mcp.embedding.base import EmbeddingProvider
from atlassian_rag_mcp.exceptions import AuthenticationError, BaseError
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.utils.http_retry import post_with_retry


class OpenAICompatClient(EmbeddingProvider):
    """OpenAI-compatible embedding provider (vLLM, OpenAI API, Azure, NVIDIA)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        url: str,
        api_key: str | None,
        model: str,
        dim: int = 2560,
    ) -> None:
        self._client = client
        self.base_url = url
        self.api_key = api_key
        self.model_name = model
        self.embedding_dim: int = dim

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def validate(self) -> None:
        try:
            payload = {
                "input": "health check probe",
                "model": self.model_name,
                "encoding_format": "float",
            }
            response = await post_with_retry(
                self._client,
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json=payload,
                timeout=10.0,
            )
            if response.status_code == 401:
                raise AuthenticationError(
                    f"Embedding service at {self.base_url} rejected authentication. "
                    f"Check EMBEDDING_API_KEY."
                )
            if response.status_code != 200:
                raise BaseError(
                    f"Embedding service at {self.base_url} "
                    f"returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
            data = response.json()
            embedding = data.get("data", [{}])[0].get("embedding", [])
            dim = len(embedding)
            self.embedding_dim = dim
            logger.info(f"Embedding OK — model={self.model_name}, dim={dim}, provider=openai_compat")
        except httpx.ConnectError:
            raise BaseError(
                f"Cannot connect to embedding service at {self.base_url}. "
                f"Check EMBEDDING_URL."
            )
        except httpx.TimeoutException:
            raise BaseError(
                f"Embedding service at {self.base_url} timed out after 10s. "
                f"Check network connectivity."
            )
        except BaseError:
            raise
        except Exception as e:
            raise BaseError(
                f"Embedding validation failed: {e}"
            ) from e

    async def get_embedding(self, text: str) -> list[float]:
        payload = {
            "input": text,
            "model": self.model_name,
            "encoding_format": "float",
        }
        try:
            response = await post_with_retry(
                self._client,
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json=payload,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("data", [{}])[0].get("embedding", [])
            raise BaseError(
                f"Embedding API error: {response.status_code} - {response.text[:500]}"
            )
        except Exception as e:
            logger.error(f"Embedding request failed: {e}")
            raise

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        t0 = time.monotonic()
        payload = {
            "input": texts,
            "model": self.model_name,
            "encoding_format": "float",
        }
        try:
            logger.debug(f"batch={len(texts)}")
            response = await post_with_retry(
                self._client,
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json=payload,
            )
            elapsed = time.monotonic() - t0
            if response.status_code == 200:
                data = response.json()
                results = [{} for _ in texts]
                for item in data.get("data", []):
                    idx = item.get("index", 0)
                    if 0 <= idx < len(results):
                        results[idx] = item.get("embedding", [])
                return [r if isinstance(r, list) else [] for r in results]
            raise BaseError(
                f"Batch embedding API error ({elapsed*1000:.0f}ms): "
                f"{response.status_code} - {response.text}"
            )
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(f"Batch embedding request failed after {elapsed*1000:.0f}ms: {e}")
            raise

    async def get_embedding_dimension(self) -> int:
        return self.embedding_dim

    async def close(self) -> None:
        await self._client.aclose()
