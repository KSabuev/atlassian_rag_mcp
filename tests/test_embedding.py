"""Unit test for embedding providers — httpx.AsyncClient.post mocked, no real service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from atlassian_rag_mcp.embedding import create_embedding_client
from atlassian_rag_mcp.embedding.base import EmbeddingProvider
from atlassian_rag_mcp.exceptions import AuthenticationError, BaseError


def _make_async_post(status_code=200, json_data=None):
    """Return an AsyncMock that returns a canned response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data or {}
    mock_resp.text = "error" if status_code != 200 else ""
    return AsyncMock(return_value=mock_resp)


def _make_openai_json(embedding=None, dim=2560):
    emb = embedding if embedding is not None else [0.1] * dim
    return {"data": [{"index": 0, "embedding": emb}]}


def _make_ollama_json(embeddings=None, dim=2560):
    emb_list = embeddings if embeddings is not None else [[0.1] * dim]
    return {"embeddings": emb_list}


class TestOpenAICompatValidate:
    async def test_ok(self):
        mock_post = _make_async_post(json_data=_make_openai_json(dim=2560))
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=2560,
            )
            await client.validate()
            assert client.embedding_dim == 2560

    async def test_http_401_raises_authentication_error(self):
        mock_post = _make_async_post(status_code=401)
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="bad",
                model="m",
                dim=2560,
            )
            with pytest.raises(AuthenticationError):
                await client.validate()

    async def test_http_500_raises_backend_error(self):
        mock_post = _make_async_post(status_code=500)
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=2560,
            )
            with pytest.raises(BaseError):
                await client.validate()

    async def test_connection_error_raises_backend_error(self):
        mock_post = AsyncMock(side_effect=httpx.ConnectError("no route"))
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=2560,
            )
            with pytest.raises(BaseError, match="Cannot connect"):
                await client.validate()

    async def test_timeout_raises_backend_error(self):
        mock_post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=2560,
            )
            with pytest.raises(BaseError, match="timed out"):
                await client.validate()

    async def test_autodetects_dimension(self):
        mock_post = _make_async_post(json_data=_make_openai_json(dim=128))
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=0,
            )
            await client.validate()
            assert client.embedding_dim == 128


class TestOpenAICompatGetEmbedding:
    async def test_returns_vector(self):
        mock_post = _make_async_post(
            json_data=_make_openai_json(embedding=[0.1, 0.2, 0.3, 0.4], dim=4)
        )
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=4,
            )
            vec = await client.get_embedding("hello")
            assert vec == [0.1, 0.2, 0.3, 0.4]

    async def test_error_raises(self):
        mock_post = _make_async_post(status_code=500)
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=4,
            )
            with pytest.raises(BaseError, match="Embedding API error: 500"):
                await client.get_embedding("hello")

    async def test_request_exception_raises(self):
        mock_post = AsyncMock(side_effect=httpx.ConnectError("no route"))
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=4,
            )
            with pytest.raises(httpx.ConnectError):
                await client.get_embedding("hello")

    async def test_retry_on_remote_protocol_error(self):
        ok_resp = _make_async_post(
            json_data=_make_openai_json(embedding=[0.1, 0.2, 0.3, 0.4], dim=4)
        )
        mock_post = AsyncMock(
            side_effect=[httpx.RemoteProtocolError("Server disconnected"), ok_resp.return_value]
        )
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=4,
            )
            vec = await client.get_embedding("hello")
            assert vec == [0.1, 0.2, 0.3, 0.4]
            assert mock_post.call_count == 2

    async def test_retry_exhausted_raises(self):
        mock_post = AsyncMock(side_effect=httpx.RemoteProtocolError("Server disconnected"))
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=4,
            )
            with pytest.raises(httpx.RemoteProtocolError):
                await client.get_embedding("hello")
            assert mock_post.call_count == 2

    async def test_no_retry_on_timeout(self):
        mock_post = AsyncMock(side_effect=httpx.ReadTimeout("slow"))
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=4,
            )
            with pytest.raises(httpx.ReadTimeout):
                await client.get_embedding("hello")
            assert mock_post.call_count == 1

    async def test_batch_error_raises(self):
        mock_post = _make_async_post(status_code=500)
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=4,
            )
            with pytest.raises(BaseError, match="Batch embedding API error"):
                await client.get_embeddings(["a", "b"])

    async def test_get_embeddings_batch(self):
        json_data = {
            "data": [
                {"index": 0, "embedding": [0.1, 0.0, 0.0]},
                {"index": 1, "embedding": [0.0, 0.2, 0.0]},
            ]
        }
        mock_post = _make_async_post(json_data=json_data)
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

            client = OpenAICompatClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                api_key="k",
                model="m",
                dim=3,
            )
            vecs = await client.get_embeddings(["a", "b"])
            assert len(vecs) == 2
            assert vecs[0] == [0.1, 0.0, 0.0]
            assert vecs[1] == [0.0, 0.2, 0.0]

    async def test_get_embeddings_empty_input(self):
        from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient

        client = OpenAICompatClient(
            client=httpx.AsyncClient(),
            url="http://t:8000",
            api_key="k",
            model="m",
            dim=4,
        )
        assert await client.get_embeddings([]) == []


class TestOllamaValidate:
    async def test_ok(self):
        mock_post = _make_async_post(json_data=_make_ollama_json(dim=2560))
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.ollama_client import OllamaClient

            client = OllamaClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                model="m",
                dim=2560,
            )
            await client.validate()
            assert client.embedding_dim == 2560

    async def test_model_not_found(self):
        mock_post = _make_async_post(status_code=404)
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.ollama_client import OllamaClient

            client = OllamaClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                model="missing",
                dim=2560,
            )
            with pytest.raises(BaseError, match="not found"):
                await client.validate()

    async def test_empty_embeddings(self):
        mock_post = _make_async_post(json_data=_make_ollama_json(embeddings=[]))
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.ollama_client import OllamaClient

            client = OllamaClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                model="m",
                dim=2560,
            )
            with pytest.raises(BaseError, match="empty"):
                await client.validate()

    async def test_autodetects_dimension(self):
        mock_post = _make_async_post(json_data=_make_ollama_json(dim=64))
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.ollama_client import OllamaClient

            client = OllamaClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                model="m",
                dim=0,
            )
            await client.validate()
            assert client.embedding_dim == 64


class TestOllamaGetEmbedding:
    async def test_returns_vector(self):
        mock_post = _make_async_post(
            json_data=_make_ollama_json(embeddings=[[0.5, 0.6, 0.7]])
        )
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.ollama_client import OllamaClient

            client = OllamaClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                model="m",
                dim=3,
            )
            vec = await client.get_embedding("hello")
            assert vec == [0.5, 0.6, 0.7]

    async def test_retry_on_remote_protocol_error(self):
        ok = _make_async_post(json_data=_make_ollama_json(embeddings=[[0.1, 0.2, 0.3]]))
        mock_post = AsyncMock(side_effect=[httpx.RemoteProtocolError("drop"), ok.return_value])
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.ollama_client import OllamaClient

            client = OllamaClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                model="m",
                dim=3,
            )
            vec = await client.get_embedding("hello")
            assert vec == [0.1, 0.2, 0.3]
            assert mock_post.call_count == 2

    async def test_no_retry_on_timeout(self):
        mock_post = AsyncMock(side_effect=httpx.ReadTimeout("slow"))
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.ollama_client import OllamaClient

            client = OllamaClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                model="m",
                dim=3,
            )
            with pytest.raises(httpx.ReadTimeout):
                await client.get_embedding("hello")
            assert mock_post.call_count == 1

    async def test_error_raises(self):
        mock_post = _make_async_post(status_code=500)
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.ollama_client import OllamaClient

            client = OllamaClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                model="m",
                dim=3,
            )
            with pytest.raises(BaseError, match="Ollama API error: 500"):
                await client.get_embedding("hello")

    async def test_batch_error_raises(self):
        mock_post = _make_async_post(status_code=500)
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.ollama_client import OllamaClient

            client = OllamaClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                model="m",
                dim=3,
            )
            with pytest.raises(BaseError, match="Ollama API error: 500"):
                await client.get_embeddings(["a", "b"])

    async def test_get_embeddings_batch(self):
        mock_post = _make_async_post(
            json_data=_make_ollama_json(embeddings=[[1.0, 0.0], [0.0, 1.0]])
        )
        with patch.object(httpx.AsyncClient, "post", mock_post):
            from atlassian_rag_mcp.embedding.ollama_client import OllamaClient

            client = OllamaClient(
                client=httpx.AsyncClient(),
                url="http://t:8000",
                model="m",
                dim=2,
            )
            vecs = await client.get_embeddings(["a", "b"])
            assert vecs == [[1.0, 0.0], [0.0, 1.0]]

    async def test_get_embeddings_empty_input(self):
        from atlassian_rag_mcp.embedding.ollama_client import OllamaClient

        client = OllamaClient(
            client=httpx.AsyncClient(),
            url="http://t:8000",
            model="m",
            dim=4,
        )
        assert await client.get_embeddings([]) == []


class TestFactory:
    def test_create_openai_compat(self):
        client = create_embedding_client(
            binding="openai_compat", url="http://t:8000", api_key="k",
            model="m", dim=2560, http_client=httpx.AsyncClient(),
        )
        from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient
        assert isinstance(client, OpenAICompatClient)

    def test_create_ollama(self):
        client = create_embedding_client(
            binding="ollama", url="http://t:8000", model="m", dim=2560,
            http_client=httpx.AsyncClient(),
        )
        from atlassian_rag_mcp.embedding.ollama_client import OllamaClient
        assert isinstance(client, OllamaClient)

    def test_unknown_binding_falls_back_to_openai(self):
        client = create_embedding_client(
            binding="unknown", url="http://t:8000", api_key="k",
            model="m", dim=2560, http_client=httpx.AsyncClient(),
        )
        from atlassian_rag_mcp.embedding.openai_compat_client import OpenAICompatClient
        assert isinstance(client, OpenAICompatClient)

    def test_returns_embedding_provider(self):
        client = create_embedding_client(
            binding="openai_compat", url="http://t:8000", api_key="k",
            model="m", dim=2560, http_client=httpx.AsyncClient(),
        )
        assert isinstance(client, EmbeddingProvider)

    async def test_get_embedding_dimension(self):
        client = create_embedding_client(
            binding="openai_compat", url="http://t:8000", api_key="k",
            model="m", dim=2560, http_client=httpx.AsyncClient(),
        )
        assert await client.get_embedding_dimension() == 2560

    async def test_close(self):
        client = create_embedding_client(
            binding="openai_compat", url="http://t:8000", api_key="k",
            model="m", dim=2560, http_client=httpx.AsyncClient(),
        )
        await client.close()
        assert client._client.is_closed

    async def test_factory_creates_client(self):
        client = create_embedding_client(
            binding="openai_compat", url="http://t:8000", api_key="k",
            model="m", dim=2560, http_client=httpx.AsyncClient(),
        )
        assert not hasattr(client, "max_token_size")
        await client.close()
        assert client._client.is_closed
