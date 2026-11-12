"""Embedding providers — factory and type dispatching."""

from __future__ import annotations

import httpx

from atlassian_rag_mcp.embedding.base import EmbeddingProvider


def create_embedding_client(
    url: str = "",
    api_key: str = "",
    model: str = "",
    dim: int = 2560,
    binding: str = "openai_compat",
    http_client: httpx.AsyncClient | None = None,
    http_timeout: float = 300.0,
) -> EmbeddingProvider:
    """Factory: create an EmbeddingProvider by binding (openai_compat/ollama)."""
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=http_timeout)

    if binding == "ollama":
        from atlassian_rag_mcp.embedding.ollama_client import OllamaClient
        return OllamaClient(client=http_client, url=url, model=model, dim=dim)

    from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient
    return OpenAICompatClient(
        client=http_client, url=url, api_key=api_key, model=model, dim=dim,
    )


__all__ = ["EmbeddingProvider", "create_embedding_client"]
