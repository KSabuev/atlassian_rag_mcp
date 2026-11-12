"""Tests for VectorService — high-level indexing and search operations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from atlassian_rag_mcp.confluence.models import ChunkPayload
from atlassian_rag_mcp.confluence.services.vector_service import VectorService
from atlassian_rag_mcp.exceptions import BaseError


@pytest.fixture
def vector_db():
    db = MagicMock()
    db.create_collection = AsyncMock()
    db.upsert_points = AsyncMock(return_value=True)
    db.delete_by_filter = AsyncMock(return_value=True)
    db.scroll_points = AsyncMock(return_value=([], None))
    db.close = AsyncMock()
    return db


@pytest.fixture
def embedding():
    emb = MagicMock()
    emb.get_embedding_dimension = AsyncMock(return_value=256)
    emb.get_embeddings = AsyncMock()
    emb.close = AsyncMock()
    return emb


@pytest.fixture
def fts_index():
    idx = MagicMock()
    idx.index_batch = AsyncMock(return_value=True)
    idx.delete_by_filter = AsyncMock(return_value=True)
    return idx


@pytest.fixture
def svc(vector_db, embedding, fts_index):
    return VectorService(
        vector_db=vector_db,
        embedding_client=embedding,
        fts_index=fts_index,
        batch_size=50,
    )


def _byte_len(t: str) -> int:
    """Deterministic counter (UTF-8 byte length) for size-split tests."""
    return len(t.encode("utf-8"))


def make_chunk(chunk_id="c1", text="some text", chunk_type="section",
               chunk_index=0, block_index=None, block_part_id=None,
               heading_hierarchy=None):
    return ChunkPayload(
        chunk_id=chunk_id,
        text=text,
        chunk_type=chunk_type,
        chunk_index=chunk_index,
        page_id="p1",
        page_title="Test",
        block_index=block_index,
        block_part_id=block_part_id,
        heading_hierarchy=heading_hierarchy or [],
    )


class TestInit:
    async def test_stores_dependencies(self, vector_db, embedding, fts_index):
        svc = VectorService(
            vector_db=vector_db,
            embedding_client=embedding,
            fts_index=fts_index,
            batch_size=10,
        )
        assert svc.vector_db is vector_db
        assert svc.embedding is embedding
        assert svc.fts_index is fts_index
        assert svc.batch_size == 10


class TestEnsureCollection:
    async def test_creates_collection_with_embedding_dim(self, svc, vector_db, embedding):
        await svc._ensure_collection()
        embedding.get_embedding_dimension.assert_awaited_once()
        vector_db.create_collection.assert_awaited_once_with(vector_size=256)

    async def test_create_collection_failure_raises(self, svc, vector_db, embedding):
        vector_db.create_collection = AsyncMock(return_value=False)
        with pytest.raises(BaseError, match="Failed to create vector collection"):
            await svc._ensure_collection()


class TestIndexPage:
    async def test_index_page_single_batch(self, svc, vector_db, embedding, fts_index):
        embedding.get_embeddings = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        chunks = [make_chunk("c1", "hello"), make_chunk("c2", "world")]
        result = await svc.index_page(
            page_id="p1", chunks=chunks,
        )
        assert result is True
        embedding.get_embeddings.assert_awaited_once_with(["hello", "world"])
        vector_db.upsert_points.assert_awaited_once()
        fts_index.index_batch.assert_awaited_once()

    async def test_index_page_multi_batch_embedding(self, svc, vector_db, embedding):
        svc.batch_size = 2

        async def embed_side_effect(texts):
            return [[i * 0.1, i * 0.2] for i in range(len(texts))]

        embedding.get_embeddings = AsyncMock(side_effect=embed_side_effect)
        chunks = [make_chunk(f"c{i}", f"text{i}") for i in range(5)]
        result = await svc.index_page(
            page_id="p1", chunks=chunks,
        )
        assert result is True
        assert embedding.get_embeddings.call_count == 3
        vector_db.upsert_points.assert_awaited_once()

    async def test_index_page_upsert_failure_raises(self, svc, vector_db, embedding):
        embedding.get_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
        vector_db.upsert_points = AsyncMock(return_value=False)
        chunks = [make_chunk("c1", "hello")]
        with pytest.raises(RuntimeError, match="Failed to upsert points for page p1"):
            await svc.index_page(
                page_id="p1", chunks=chunks,
            )

    async def test_index_page_embedding_error_raises(self, svc, embedding):
        embedding.get_embeddings = AsyncMock(side_effect=RuntimeError("API down"))
        chunks = [make_chunk("c1", "hello")]
        with pytest.raises(RuntimeError, match="API down"):
            await svc.index_page(
                page_id="p1", chunks=chunks,
            )

    async def test_index_page_empty_chunks(self, svc, embedding, vector_db, fts_index):
        result = await svc.index_page(
            page_id="p1", chunks=[],
        )
        assert result is True
        embedding.get_embeddings.assert_not_called()
        vector_db.upsert_points.assert_not_called()
        fts_index.index_batch.assert_awaited_once_with([])

    async def test_index_page_fts_failure_raises(self, svc, vector_db, embedding, fts_index):
        embedding.get_embeddings = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        fts_index.index_batch = AsyncMock(return_value=False)
        chunks = [make_chunk("c1", "hello"), make_chunk("c2", "world")]
        with pytest.raises(RuntimeError, match="FTS index failed for page p1"):
            await svc.index_page(
                page_id="p1", chunks=chunks,
            )

    async def test_index_page_vector_count_mismatch_raises(self, svc, embedding):
        embedding.get_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
        chunks = [make_chunk("c1", "a"), make_chunk("c2", "b")]
        with pytest.raises(RuntimeError, match="Vector count mismatch for page p1"):
            await svc.index_page(
                page_id="p1", chunks=chunks,
            )

    async def test_index_page_empty_vectors_raises(self, svc, embedding):
        embedding.get_embeddings = AsyncMock(return_value=[[]])
        chunks = [make_chunk("c1", "hello")]
        with pytest.raises(RuntimeError, match="Embedding returned empty vectors for page p1"):
            await svc.index_page(
                page_id="p1", chunks=chunks,
            )


class TestEmbedTextNormalization:
    """The embedding input is the HTML-stripped text; the stored payload keeps the full HTML."""

    async def test_table_chunk_embeds_clean_text(self, svc, vector_db, embedding):
        html = '<tr><td colspan="1">PROJ-1001</td><td>Иванов Иван Иванович</td></tr>'
        embedding.get_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
        chunks = [make_chunk("c1", html, chunk_type="table_row")]
        result = await svc.index_page(page_id="p1", chunks=chunks)
        assert result is True
        embedding.get_embeddings.assert_awaited_once_with(
            ["PROJ-1001 Иванов Иван Иванович"]
        )
        points = vector_db.upsert_points.await_args.args[0]
        assert points[0]["payload"]["text"] == html

    async def test_section_plain_text_passes_through(self, svc, embedding):
        embedding.get_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
        chunks = [make_chunk("c1", "hello world")]
        await svc.index_page(page_id="p1", chunks=chunks)
        embedding.get_embeddings.assert_awaited_once_with(["hello world"])

    async def test_oversized_table_chunk_splits_clean_text(self, svc, vector_db, embedding):
        svc._max_tokens = 30
        svc._token_counter = _byte_len
        html = (
            "<table><tr><td>" + "слово " * 10 + "</td></tr>"
            "<tr><td>" + "файл " * 10 + "</td></tr></table>"
        )

        async def embed_side_effect(texts):
            return [[0.1, 0.2] for _ in texts]

        embedding.get_embeddings = AsyncMock(side_effect=embed_side_effect)
        chunks = [make_chunk("c1", html, chunk_type="table_row")]
        result = await svc.index_page(page_id="p1", chunks=chunks)
        assert result is True
        texts = [t for call in embedding.get_embeddings.await_args_list for t in call.args[0]]
        assert len(texts) > 1
        assert all("<" not in t and ">" not in t for t in texts)
        points = vector_db.upsert_points.await_args.args[0]
        assert len(points) == len(texts)
        assert all(p["payload"]["text"] == html for p in points)


class TestDeletePage:
    async def test_delete_deletes_both_stores(self, svc, vector_db, fts_index):
        result = await svc.delete_page("p42")
        assert result is True
        expected_filter = {"must": [{"key": "page_id", "match": {"value": "p42"}}]}
        fts_index.delete_by_filter.assert_awaited_once_with(expected_filter)
        vector_db.delete_by_filter.assert_awaited_once_with(expected_filter)

    async def test_delete_fts_failure_returns_false(self, svc, vector_db, fts_index):
        fts_index.delete_by_filter = AsyncMock(return_value=False)
        result = await svc.delete_page("p42")
        assert result is False
        vector_db.delete_by_filter.assert_awaited_once()

    async def test_delete_vdb_failure_returns_false(self, svc, vector_db, fts_index):
        vector_db.delete_by_filter = AsyncMock(return_value=False)
        result = await svc.delete_page("p42")
        assert result is False
        fts_index.delete_by_filter.assert_awaited_once()


class TestGetChunksByFilter:
    async def test_returns_enriched_chunks(self, svc, vector_db):
        vector_db.scroll_points = AsyncMock(return_value=([
            {
                "id": "c1",
                "payload": {
                    "chunk_id": "c1",
                    "text": "hello",
                    "chunk_type": "section",
                    "chunk_index": 0,
                    "page_id": "p1",
                    "page_title": "Test",
                    "parent_id": "",
                    "page_version": 1,
                    "updated_at": "2024-01-01",
                    "block_index": -1,
                    "block_part_id": -1,
                    "heading_hierarchy": [],
                },
            },
        ], None))
        result = await svc.get_chunks_by_filter(page_id="p1")
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"
        assert result[0]["text"] == "hello"

    async def test_no_conditions_returns_empty(self, svc):
        result = await svc.get_chunks_by_filter()
        assert result == []

    async def test_scroll_with_offset_and_limit(self, svc, vector_db):
        vector_db.scroll_points = AsyncMock(return_value=([], None))
        result = await svc.get_chunks_by_filter(page_id="p1", limit=5, offset=10)
        assert result == []


class TestClose:
    async def test_closes_vector_db_and_embedding(self, svc, vector_db, embedding):
        await svc.close()
        vector_db.close.assert_awaited_once()
        embedding.close.assert_awaited_once()


class TestPrepareTokenSplit:
    """Verify that _prepare expands oversized chunks into sub-chunks."""

    async def test_short_chunk_not_split(self, svc, embedding):
        """FTS receives the exact same canonical payload as the vector point."""
        embedding.get_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
        chunks = [make_chunk("c1", "hello")]
        points, fts_docs = await svc._prepare("p1", chunks)
        assert len(points) == 1
        assert points[0]["id"] == "c1"
        assert points[0]["payload"]["text"] == "hello"
        assert points[0]["payload"].get("full_text") is None
        assert points[0]["payload"].get("parent_chunk_id") is None
        assert len(fts_docs) == 1
        assert fts_docs == [p["payload"] for p in points]

    async def test_oversized_chunk_splitted(self, embedding):
        """Point ids are valid, distinct UUIDs (Qdrant requirement)."""
        import uuid
        long_text = "the quick brown fox " * 30
        svc = VectorService(
            vector_db=MagicMock(),
            embedding_client=embedding,
            fts_index=MagicMock(),
            batch_size=50,
            max_tokens=50,
            overlap_tokens=10,
        )
        svc._token_counter = _byte_len
        def embed_side_effect(texts):
            return [[0.1, 0.2]] * len(texts)

        embedding.get_embeddings = AsyncMock(side_effect=embed_side_effect)
        chunks = [make_chunk("c1", long_text)]
        points, fts_docs = await svc._prepare("p1", chunks)
        assert len(points) > 1
        pids = []
        for p in points:
            u = uuid.UUID(p["id"])
            pids.append(str(u))
        assert len(set(pids)) == len(pids)
        for p in points:
            assert p["payload"]["chunk_id"] == "c1"
            assert p["payload"]["text"] == long_text
            assert p["payload"].get("full_text") is None
            assert p["payload"].get("parent_chunk_id") is None
        assert len(fts_docs) == 1
        assert fts_docs[0]["chunk_id"] == "c1"
        assert fts_docs[0]["text"] == long_text

    async def test_default_counter_is_bundled_qwen3(self, embedding):
        """No explicit counter: the bundled Qwen3 tokenizer is used, so a
        10000-byte (~3300 token) Cyrillic chunk fits the default 8000 budget
        and is embedded whole."""
        long_text = ("абвгд " * 5000).encode("utf-8")[:10000].decode(
            "utf-8", errors="ignore"
        )
        svc = VectorService(
            vector_db=MagicMock(),
            embedding_client=embedding,
            fts_index=MagicMock(),
            batch_size=50,
        )
        def embed_side_effect(texts):
            return [[0.1, 0.2]] * len(texts)

        embedding.get_embeddings = AsyncMock(side_effect=embed_side_effect)
        points, _fts_docs = await svc._prepare("p1", [make_chunk("c1", long_text)])
        assert len(points) == 1

    async def test_whole_chunk_embedded_when_real_tokens_fit(self, embedding):
        """~21000 bytes: far over any byte budget, but the real token count
        (~3 chars/token) fits the 8000-token budget — ONE vector, no split."""
        long_text = "the quick brown fox jumps over " * 700
        from atlassian_rag_mcp.utils.text import html_to_text

        def counter(t: str) -> int:
            return len(t) // 3

        svc = VectorService(
            vector_db=MagicMock(),
            embedding_client=embedding,
            fts_index=MagicMock(),
            batch_size=50,
            max_tokens=8000,
            overlap_tokens=200,
        )
        svc._token_counter = counter
        def embed_side_effect(texts):
            return [[0.1, 0.2]] * len(texts)

        embedding.get_embeddings = AsyncMock(side_effect=embed_side_effect)
        points, _fts_docs = await svc._prepare("p1", [make_chunk("c1", long_text)])
        assert len(points) == 1
        assert points[0]["id"] == "c1"
        sent = embedding.get_embeddings.await_args.args[0]
        assert sent == [html_to_text(long_text)]

    async def test_dense_chunk_still_split_despite_tokenizer(self, embedding):
        """Token-dense text (~1 token/byte): the real token count (15000)
        overflows the 8000-token budget — split at token boundaries."""
        long_text = ("1 2 3 4 5 6 7 8 9 0 " * 1500)

        def counter(t: str) -> int:
            return len(t.encode("utf-8"))

        svc = VectorService(
            vector_db=MagicMock(),
            embedding_client=embedding,
            fts_index=MagicMock(),
            batch_size=50,
            max_tokens=8000,
            overlap_tokens=200,
        )
        svc._token_counter = counter
        def embed_side_effect(texts):
            return [[0.1, 0.2]] * len(texts)

        embedding.get_embeddings = AsyncMock(side_effect=embed_side_effect)
        points, _fts_docs = await svc._prepare("p1", [make_chunk("c1", long_text)])
        assert len(points) >= 2
        seen = []
        for call in embedding.get_embeddings.await_args_list:
            seen.extend(call.args[0])
        for t in seen:
            assert len(t.encode("utf-8")) <= 8000

    async def test_mix_short_and_long(self, embedding):
        long_text = "alpha bravo " * 30
        svc = VectorService(
            vector_db=MagicMock(),
            embedding_client=embedding,
            fts_index=MagicMock(),
            batch_size=50,
            max_tokens=50,
            overlap_tokens=10,
        )
        svc._token_counter = _byte_len
        def embed_side_effect(texts):
            return [[0.1]] * len(texts)

        embedding.get_embeddings = AsyncMock(side_effect=embed_side_effect)
        chunks = [
            make_chunk("c1", "short"),
            make_chunk("c2", long_text),
        ]
        points, fts_docs = await svc._prepare("p1", chunks)
        c1_points = [p for p in points if p["payload"]["chunk_id"] == "c1"]
        c2_points = [p for p in points if p["payload"]["chunk_id"] == "c2"]
        assert len(c1_points) == 1
        assert c1_points[0]["id"] == "c1"
        assert c1_points[0]["payload"]["text"] == "short"
        assert len(c2_points) >= 2
        assert all(p["payload"]["text"] == long_text for p in c2_points)
        assert all(p["payload"].get("full_text") is None for p in c2_points)


class TestDedupeByChunkId:
    def test_dedupe_shared_chunk_id(self):
        from atlassian_rag_mcp.storage.utils import dedupe_chunks_by_id
        chunks = [
            {"chunk_id": "base", "text": "FULL"},
            {"chunk_id": "base", "text": "FULL"},
            {"chunk_id": "c3", "text": "standalone"},
        ]
        result = dedupe_chunks_by_id(chunks)
        assert len(result) == 2
        assert [c["chunk_id"] for c in result] == ["base", "c3"]
        assert result[0]["text"] == "FULL"

    def test_unique_ids_kept(self):
        from atlassian_rag_mcp.storage.utils import dedupe_chunks_by_id
        chunks = [
            {"chunk_id": "a", "text": "A"},
            {"chunk_id": "b", "text": "B"},
        ]
        assert len(dedupe_chunks_by_id(chunks)) == 2

    def test_empty(self):
        from atlassian_rag_mcp.storage.utils import dedupe_chunks_by_id
        assert dedupe_chunks_by_id([]) == []


class TestGetChunksByFilterMerge:
    async def test_composite_chunks_deduped(self, svc, vector_db):
        vector_db.scroll_points = AsyncMock(return_value=([
            {
                "id": "point-a",
                "payload": {
                    "chunk_id": "base", "text": "ORIGINAL FULL TEXT",
                    "chunk_type": "section", "chunk_index": 0, "page_id": "p1",
                    "page_title": "T", "parent_id": "", "page_version": 1,
                    "block_index": 1, "block_part_id": None, "heading_hierarchy": [],
                },
            },
            {
                "id": "point-b",
                "payload": {
                    "chunk_id": "base", "text": "ORIGINAL FULL TEXT",
                    "chunk_type": "section", "chunk_index": 0, "page_id": "p1",
                    "page_title": "T", "parent_id": "", "page_version": 1,
                    "block_index": 1, "block_part_id": None, "heading_hierarchy": [],
                },
            },
            {
                "id": "point-c",
                "payload": {
                    "chunk_id": "c3", "text": "standalone",
                    "chunk_type": "section", "chunk_index": 1, "page_id": "p1",
                    "page_title": "T", "parent_id": "", "page_version": 1,
                    "block_index": 2, "block_part_id": None, "heading_hierarchy": [],
                },
            },
        ], None))
        result = await svc.get_chunks_by_filter(page_id="p1")
        assert len(result) == 2
        by_id = {c["chunk_id"]: c for c in result}
        assert by_id["base"]["text"] == "ORIGINAL FULL TEXT"
        assert by_id["c3"]["text"] == "standalone"


class TestCounter:
    """The bundled Qwen3 tokenizer is the counter by default."""

    async def test_default_counter_counts_real_tokens(self, vector_db, embedding, fts_index):
        svc = VectorService(
            vector_db=vector_db,
            embedding_client=embedding,
            fts_index=fts_index,
            batch_size=50,
        )
        assert svc._token_counter("Привет мир") == 5
