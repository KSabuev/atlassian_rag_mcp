"""Shared test suite for VectorDB — runs same tests on Qdrant and LanceDB."""

from __future__ import annotations

import math
import tempfile
import uuid
from dataclasses import dataclass

import pytest

from atlassian_rag_mcp.storage.vectordb.lancedb_client import LanceDBClient


@dataclass
class QdrantConfig:
    host: str = "127.0.0.1"
    port: int = 6333
    collection: str = "atlassian_chunks"
    api_key: str | None = None
    vector_size: int = 2560


def _cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2)


def _make_point_lance(
    num: int,
    vector: list[float],
    page_id: str = "p1",
    page_title: str = "Test Page",
    chunk_type: str = "section",
    chunk_index: int = 0,
    page_version: int = 1,
    parent_id: str = "",
    block_index: int = -1,
    block_part_id: int = -1,
    updated_at: str = "2024-01-01T00:00:00",
    heading_hierarchy: list | None = None,
    text: str = "Sample chunk text",
):
    """Point builder for LanceDB (string IDs)."""
    chunk_id = f"c{num}"
    if heading_hierarchy is None:
        heading_hierarchy = []
    return {
        "id": chunk_id,
        "vector": vector,
        "payload": {
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "text": text,
            "page_id": page_id,
            "page_title": page_title,
            "page_version": page_version,
            "parent_id": parent_id,
            "block_index": block_index,
            "block_part_id": block_part_id,
            "chunk_index": chunk_index,
            "updated_at": updated_at,
            "heading_hierarchy": heading_hierarchy,
        },
    }


def _make_point_qdrant(
    num: int,
    vector: list[float],
    page_id: str = "p1",
    page_title: str = "Test Page",
    chunk_type: str = "section",
    chunk_index: int = 0,
    page_version: int = 1,
    parent_id: str = "",
    block_index: int = -1,
    block_part_id: int = -1,
    updated_at: str = "2024-01-01T00:00:00",
    heading_hierarchy: list | None = None,
    text: str = "Sample chunk text",
):
    """Point builder for Qdrant (integer IDs)."""
    chunk_id = f"c{num}"
    if heading_hierarchy is None:
        heading_hierarchy = []
    return {
        "id": num,
        "vector": vector,
        "payload": {
            "chunk_id": chunk_id,
            "chunk_type": chunk_type,
            "text": text,
            "page_id": page_id,
            "page_title": page_title,
            "page_version": page_version,
            "parent_id": parent_id,
            "block_index": block_index,
            "block_part_id": block_part_id,
            "chunk_index": chunk_index,
            "updated_at": updated_at,
            "heading_hierarchy": heading_hierarchy,
        },
    }


def _qdrant_available() -> bool:
    try:
        from qdrant_client import QdrantClient

        from atlassian_rag_mcp.config import settings
        client = QdrantClient(
            host=settings.qdrant_host or "127.0.0.1",
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=False,
            timeout=2,
        )
        client.get_collections()
        client.close()
        return True
    except Exception:
        return False


class VectorDBTestSuite:
    """Shared async test methods for VectorDB backends.

    Each concrete subclass must provide:
      - `store` fixture  (connected VectorDB instance)
      - `make_point` method returning a valid point dict
    """

    def make_point(self, num, vector, **kwargs) -> dict:
        raise NotImplementedError

    def _make_sub_points(self, base_num: int, logical_id: str, vector, **kwargs) -> list[dict]:
        """Two sub-points of a split composite: one logical chunk_id, distinct point ids."""
        subs = []
        for i in range(2):
            p = self.make_point(base_num * 10 + i, vector, **kwargs)
            p["payload"]["chunk_id"] = logical_id
            if isinstance(p["id"], int):
                p["id"] = base_num * 1000 + i
            else:
                p["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{logical_id}_{i}"))
            subs.append(p)
        return subs

    async def test_delete_old_keeps_composite_sub_points(self, store):
        """Re-index simulation: the post-write cleanup excludes the fresh
        chunks by LOGICAL chunk_id. Sub-points of a split composite (point id
        != chunk_id) must survive the cleanup — regression guard for the
        backend storing point ids in the chunk identity column.
        """
        v = [1.0, 0.0, 0.0, 0.0]
        old_points = [
            self.make_point(1, v, page_id="p1", page_version=1),
            *self._make_sub_points(2, "old-comp", v, page_id="p1", page_version=1),
        ]
        await store.upsert_points(old_points)

        new_points = [
            self.make_point(11, v, page_id="p1", page_version=2),
            *self._make_sub_points(12, "new-comp", v, page_id="p1", page_version=2),
        ]
        await store.upsert_points(new_points)
        new_logical_ids = [p["payload"]["chunk_id"] for p in new_points]

        fc = {
            "must": [{"key": "page_id", "match": {"value": "p1"}}],
            "must_not": [{"key": "chunk_id", "match": {"any": new_logical_ids}}],
        }
        await store.delete_by_filter(fc)

        results, _ = await store.scroll_points(
            filter_condition={"must": [{"key": "page_id", "match": {"value": "p1"}}]}
        )
        surviving = [r["payload"]["chunk_id"] for r in results]
        assert sorted(surviving) == sorted(new_logical_ids)
        assert len(surviving) == 3

    async def test_upsert_and_search(self, store):
        v = [1.0, 0.0, 0.0, 0.0]
        await store.upsert_points([self.make_point(1, v)])
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        assert len(results) == 1
        assert results[0]["score"] > 0.99

    async def test_search_ranking(self, store):
        points = [
            self.make_point(1, [1.0, 0.0, 0.0, 0.0]),
            self.make_point(2, [0.9, 0.1, 0.0, 0.0]),
            self.make_point(3, [0.0, 1.0, 0.0, 0.0]),
        ]
        await store.upsert_points(points)
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0], limit=3)
        assert len(results) == 3

    async def test_search_with_filter_value(self, store):
        points = [
            self.make_point(1, [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            self.make_point(2, [0.9, 0.0, 0.0, 0.0], page_id="p2"),
        ]
        await store.upsert_points(points)
        qf = {"must": [{"key": "page_id", "match": {"value": "p1"}}]}
        results = await store.search(
            vector=[1.0, 0.0, 0.0, 0.0], limit=5, query_filter=qf
        )
        assert len(results) == 1

    async def test_search_with_filter_any(self, store):
        points = [
            self.make_point(1, [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            self.make_point(2, [0.9, 0.0, 0.0, 0.0], page_id="p2"),
            self.make_point(3, [0.8, 0.0, 0.0, 0.0], page_id="p3"),
        ]
        await store.upsert_points(points)
        qf = {"must": [{"key": "page_id", "match": {"any": ["p1", "p3"]}}]}
        results = await store.search(
            vector=[1.0, 0.0, 0.0, 0.0], limit=5, query_filter=qf
        )
        assert len(results) == 2
        ids = {str(r["id"]) for r in results}
        expected = {str(self.make_point(1, [])["id"]), str(self.make_point(3, [])["id"])}
        assert ids == expected

    async def test_search_limit(self, store):
        points = [
            self.make_point(i, [1.0 - i * 0.01, 0.0, 0.0, 0.0])
            for i in range(10)
        ]
        await store.upsert_points(points)
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0], limit=3)
        assert len(results) == 3

    async def test_upsert_empty_list(self, store):
        result = await store.upsert_points([])
        assert result is True

    async def test_upsert_overwrite(self, store):
        p1 = self.make_point(1, [1.0, 0.0, 0.0, 0.0], text="original")
        p2 = self.make_point(1, [1.0, 0.0, 0.0, 0.0], text="updated")
        await store.upsert_points([p1, p2])
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0]["payload"]["text"] == "updated"

    async def test_scroll_basic(self, store):
        points = [
            self.make_point(1, [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            self.make_point(2, [0.9, 0.0, 0.0, 0.0], page_id="p1"),
            self.make_point(3, [0.0, 1.0, 0.0, 0.0], page_id="p2"),
        ]
        await store.upsert_points(points)
        results, next_offset = await store.scroll_points(filter_condition={})
        assert len(results) == 3
        assert next_offset is None

    async def test_scroll_with_filter(self, store):
        points = [
            self.make_point(1, [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            self.make_point(2, [0.9, 0.0, 0.0, 0.0], page_id="p1"),
            self.make_point(3, [0.0, 1.0, 0.0, 0.0], page_id="p2"),
        ]
        await store.upsert_points(points)
        fc = {"must": [{"key": "page_id", "match": {"value": "p1"}}]}
        results, _ = await store.scroll_points(filter_condition=fc)
        assert len(results) == 2

    async def test_scroll_pagination(self, store):
        points = [self.make_point(i, [1.0, 0.0, 0.0, 0.0]) for i in range(1, 6)]
        await store.upsert_points(points)
        page1, next_off = await store.scroll_points(
            filter_condition={}, limit=2
        )
        assert len(page1) == 2
        assert next_off is not None
        page2, next_off2 = await store.scroll_points(
            filter_condition={}, limit=2, offset=next_off
        )
        assert len(page2) == 2
        page3, next_off3 = await store.scroll_points(
            filter_condition={}, limit=2, offset=next_off2
        )
        assert len(page3) == 1
        assert next_off3 is None

    async def test_scroll_with_vectors(self, store):
        await store.upsert_points(
            [self.make_point(1, [1.0, 0.0, 0.0, 0.0])]
        )
        results, _ = await store.scroll_points(
            filter_condition={}, with_vectors=True
        )
        assert len(results) == 1
        vec = results[0].get("vector")
        assert vec is not None
        assert vec[0] == pytest.approx(1.0, abs=1e-5)

    async def test_scroll_without_vectors(self, store):
        await store.upsert_points(
            [self.make_point(1, [1.0, 0.0, 0.0, 0.0])]
        )
        results, _ = await store.scroll_points(filter_condition={})
        assert len(results) == 1
        assert "vector" not in results[0]

    async def test_delete_by_filter(self, store):
        points = [
            self.make_point(1, [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            self.make_point(2, [0.9, 0.0, 0.0, 0.0], page_id="p2"),
        ]
        await store.upsert_points(points)
        fc = {"must": [{"key": "page_id", "match": {"value": "p1"}}]}
        await store.delete_by_filter(fc)
        results, _ = await store.scroll_points(filter_condition={})
        assert len(results) == 1
        remaining = results[0]["payload"]["page_id"]
        assert remaining == "p2"

    async def test_delete_by_filter_any(self, store):
        points = [
            self.make_point(1, [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            self.make_point(2, [0.9, 0.0, 0.0, 0.0], page_id="p2"),
            self.make_point(3, [0.8, 0.0, 0.0, 0.0], page_id="p3"),
        ]
        await store.upsert_points(points)
        fc = {"must": [{"key": "page_id", "match": {"any": ["p1", "p3"]}}]}
        await store.delete_by_filter(fc)
        results, _ = await store.scroll_points(filter_condition={})
        assert len(results) == 1
        remaining = results[0]["payload"]["page_id"]
        assert remaining == "p2"

    async def test_delete_by_filter_none(self, store):
        await store.upsert_points(
            [self.make_point(1, [1.0, 0.0, 0.0, 0.0])]
        )
        result = await store.delete_by_filter({})
        assert result is True
        results, _ = await store.scroll_points(filter_condition={})
        assert len(results) == 1

    async def test_collection_exists(self, store):
        assert await store.collection_exists() is True

    async def test_get_collection_info(self, store):
        await store.upsert_points(
            [self.make_point(1, [1.0, 0.0, 0.0, 0.0])]
        )
        info = await store.get_collection_info()
        assert info["vectors_count"] == 1
        assert info["config"]["vector_size"] == 4
        assert info["config"]["distance"] == "Cosine"

    async def test_health_check_healthy(self, store):
        health = await store.health_check()
        assert health["status"] == "healthy"

    async def test_payload_all_fields_present(self, store):
        hh = ["Intro"]
        point = self.make_point(
            1,
            [1.0, 0.0, 0.0, 0.0],
            page_id="p42",
            page_title="My Page",
            chunk_type="table",
            chunk_index=2,
            page_version=3,
            parent_id="p10",
            block_index=0,
            block_part_id=5,
            updated_at="2025-01-15T12:00:00",
            heading_hierarchy=hh,
            text="Table cell content",
        )
        await store.upsert_points([point])
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0])
        assert len(results) == 1
        p = results[0]["payload"]
        assert p["chunk_id"] == point["payload"]["chunk_id"]
        assert p["chunk_type"] == "table"
        assert p["text"] == "Table cell content"
        assert p["page_id"] == "p42"
        assert p["page_title"] == "My Page"
        assert p["page_version"] == 3
        assert p["parent_id"] == "p10"
        assert p["block_index"] == 0
        assert p["block_part_id"] == 5
        assert p["chunk_index"] == 2
        assert p["updated_at"] == "2025-01-15T12:00:00"
        assert p["heading_hierarchy"] == hh



@pytest.fixture
async def lancedb_store():
    tmpdir = tempfile.mkdtemp()
    s = LanceDBClient(db_path=tmpdir, vector_size=4)
    await s.start()
    yield s
    await s.close()
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestLanceDBVectorDB(VectorDBTestSuite):
    @pytest.fixture
    def store(self, lancedb_store):
        return lancedb_store

    def make_point(self, num, vector, **kwargs):
        return _make_point_lance(num, vector, **kwargs)



_qdrant_skip = pytest.mark.skipif(
    not _qdrant_available(),
    reason="Qdrant server not available",
)


@_qdrant_skip
class TestQdrantVectorDB(VectorDBTestSuite):
    @pytest.fixture
    async def store(self):
        from atlassian_rag_mcp.config import settings
        from atlassian_rag_mcp.storage.vectordb.qdrant_client import QdrantClient

        coll_name = f"utest_{uuid.uuid4().hex[:12]}"
        config = QdrantConfig(
            host=settings.qdrant_host or "127.0.0.1",
            port=settings.qdrant_port,
            collection=coll_name,
            api_key=settings.qdrant_api_key,
            vector_size=4,
        )
        s = QdrantClient(
            host=config.host,
            port=config.port,
            api_key=config.api_key or "",
            collection=config.collection,
            vector_size=config.vector_size,
        )
        await s.start()
        await s.create_collection(vector_size=4)
        yield s
        try:
            from qdrant_client import AsyncQdrantClient
            client = AsyncQdrantClient(
                host=config.host, port=config.port, api_key=config.api_key, https=False
            )
            await client.delete_collection(collection_name=coll_name)
            await client.close()
        except Exception:
            pass
        await s.close()

    def make_point(self, num, vector, **kwargs):
        return _make_point_qdrant(num, vector, **kwargs)
