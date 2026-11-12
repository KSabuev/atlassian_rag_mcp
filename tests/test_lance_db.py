"""Tests for LanceDB storage backend using a real in-memory LanceDB instance."""

from __future__ import annotations

import shutil
import tempfile

import pytest

from atlassian_rag_mcp.storage.vectordb.lancedb_client import LanceDBClient


def _make_point(
    chunk_id,
    vector,
    page_id="p1",
    page_title="Test Page",
    chunk_type="section",
    chunk_index=0,
    page_version=1,
    parent_id="",
    block_index=-1,
    block_part_id=-1,
    updated_at="2024-01-01T00:00:00",
    heading_hierarchy=None,
    text="Sample chunk text",
    space_key=None,
):
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
            "space_key": space_key,
        },
    }


@pytest.fixture
async def store():
    tmpdir = tempfile.mkdtemp()
    s = LanceDBClient(db_path=tmpdir, vector_size=4)
    await s.start()
    yield s
    await s.close()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestLanceDBClientInit:

    @pytest.fixture
    async def fresh_store(self):
        tmpdir = tempfile.mkdtemp()
        s = LanceDBClient(db_path=tmpdir, vector_size=4)
        await s.start()
        yield s
        await s.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    async def test_creates_table_on_init(self, fresh_store):
        assert await fresh_store.collection_exists() is True
        info = await fresh_store.get_collection_info()
        assert info is not None
        assert info["vectors_count"] == 0

    async def test_reopens_existing_table(self, fresh_store):
        await fresh_store.upsert_points([_make_point("c1", [1.0, 0.0, 0.0, 0.0])])
        await fresh_store.close()

        tmpdir = fresh_store._db_path
        store2 = LanceDBClient(db_path=tmpdir, vector_size=4)
        await store2.start()
        assert await store2.collection_exists() is True
        info = await store2.get_collection_info()
        assert info["vectors_count"] == 1
        await store2.close()


class TestLanceDBClientUpsertSearch:

    async def test_upsert_and_search(self, store):
        v = [1.0, 0.0, 0.0, 0.0]
        await store.upsert_points([_make_point("c1", v)])
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0], limit=5)
        assert len(results) == 1
        assert results[0]["id"] == "c1"
        assert results[0]["score"] > 0.99

    async def test_search_ranking(self, store):
        points = [
            _make_point("c1", [1.0, 0.0, 0.0, 0.0]),
            _make_point("c2", [0.9, 0.1, 0.0, 0.0]),
            _make_point("c3", [0.0, 1.0, 0.0, 0.0]),
        ]
        await store.upsert_points(points)
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0], limit=3)
        assert len(results) == 3
        assert results[0]["id"] == "c1"
        assert results[2]["id"] == "c3"

    async def test_search_with_filter_value(self, store):
        points = [
            _make_point("c1", [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            _make_point("c2", [0.9, 0.0, 0.0, 0.0], page_id="p2"),
        ]
        await store.upsert_points(points)
        qf = {"must": [{"key": "page_id", "match": {"value": "p1"}}]}
        results = await store.search(
            vector=[1.0, 0.0, 0.0, 0.0], limit=5, query_filter=qf
        )
        assert len(results) == 1
        assert results[0]["id"] == "c1"

    async def test_search_with_filter_any(self, store):
        points = [
            _make_point("c1", [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            _make_point("c2", [0.9, 0.0, 0.0, 0.0], page_id="p2"),
            _make_point("c3", [0.8, 0.0, 0.0, 0.0], page_id="p3"),
        ]
        await store.upsert_points(points)
        qf = {"must": [{"key": "page_id", "match": {"any": ["p1", "p3"]}}]}
        results = await store.search(
            vector=[1.0, 0.0, 0.0, 0.0], limit=5, query_filter=qf
        )
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"c1", "c3"}

    async def test_upsert_empty_list(self, store):
        result = await store.upsert_points([])
        assert result is True

    async def test_upsert_overwrite(self, store):
        p1 = _make_point("c1", [1.0, 0.0, 0.0, 0.0], text="original")
        p2 = _make_point("c1", [1.0, 0.0, 0.0, 0.0], text="updated")
        await store.upsert_points([p1, p2])
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0])
        assert len(results) == 1
        assert results[0]["payload"]["text"] == "updated"

    async def test_search_limit(self, store):
        points = [
            _make_point(f"c{i}", [1.0 - i * 0.01, 0.0, 0.0, 0.0])
            for i in range(10)
        ]
        await store.upsert_points(points)
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0], limit=3)
        assert len(results) == 3


class TestLanceDBClientScrollDelete:

    async def test_scroll_basic(self, store):
        points = [
            _make_point("c1", [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            _make_point("c2", [0.9, 0.0, 0.0, 0.0], page_id="p1"),
            _make_point("c3", [0.0, 1.0, 0.0, 0.0], page_id="p2"),
        ]
        await store.upsert_points(points)
        results, next_offset = await store.scroll_points(filter_condition={})
        assert len(results) == 3
        assert next_offset is None

    async def test_scroll_with_filter(self, store):
        points = [
            _make_point("c1", [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            _make_point("c2", [0.9, 0.0, 0.0, 0.0], page_id="p1"),
            _make_point("c3", [0.0, 1.0, 0.0, 0.0], page_id="p2"),
        ]
        await store.upsert_points(points)
        fc = {"must": [{"key": "page_id", "match": {"value": "p1"}}]}
        results, _ = await store.scroll_points(filter_condition=fc)
        assert len(results) == 2
        ids = {r["id"] for r in results}
        assert ids == {"c1", "c2"}

    async def test_scroll_with_vectors(self, store):
        points = [_make_point("c1", [1.0, 0.0, 0.0, 0.0])]
        await store.upsert_points(points)
        results, _ = await store.scroll_points(
            filter_condition={}, with_vectors=True
        )
        assert len(results) == 1
        vec = results[0].get("vector")
        assert vec is not None
        assert vec[0] == pytest.approx(1.0)

    async def test_scroll_pagination(self, store):
        points = [_make_point(f"c{i}", [1.0, 0.0, 0.0, 0.0]) for i in range(5)]
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

    async def test_delete_by_filter(self, store):
        points = [
            _make_point("c1", [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            _make_point("c2", [0.9, 0.0, 0.0, 0.0], page_id="p2"),
        ]
        await store.upsert_points(points)
        fc = {"must": [{"key": "page_id", "match": {"value": "p1"}}]}
        await store.delete_by_filter(fc)
        results, _ = await store.scroll_points(filter_condition={})
        assert len(results) == 1
        assert results[0]["id"] == "c2"

    async def test_delete_by_filter_any(self, store):
        points = [
            _make_point("c1", [1.0, 0.0, 0.0, 0.0], page_id="p1"),
            _make_point("c2", [0.9, 0.0, 0.0, 0.0], page_id="p2"),
            _make_point("c3", [0.8, 0.0, 0.0, 0.0], page_id="p3"),
        ]
        await store.upsert_points(points)
        fc = {"must": [{"key": "page_id", "match": {"any": ["p1", "p3"]}}]}
        await store.delete_by_filter(fc)
        results, _ = await store.scroll_points(filter_condition={})
        assert len(results) == 1
        assert results[0]["id"] == "c2"

    async def test_delete_by_filter_none(self, store):
        await store.upsert_points([_make_point("c1", [1.0, 0.0, 0.0, 0.0])])
        result = await store.delete_by_filter({})
        assert result is True
        results, _ = await store.scroll_points(filter_condition={})
        assert len(results) == 1


class TestLanceDBClientPayloadSchema:

    async def test_payload_all_fields_present(self, store):
        hh = ["Intro"]
        point = _make_point(
            "c1",
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
        assert p["chunk_id"] == "c1"
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

    async def test_payload_heading_hierarchy_empty_list(self, store):
        point = _make_point("c1", [1.0, 0.0, 0.0, 0.0], heading_hierarchy=[])
        await store.upsert_points([point])
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0])
        assert results[0]["payload"]["heading_hierarchy"] == []

    async def test_payload_heading_hierarchy_none(self, store):
        point = _make_point("c1", [1.0, 0.0, 0.0, 0.0], heading_hierarchy=None)
        await store.upsert_points([point])
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0])
        assert results[0]["payload"]["heading_hierarchy"] == []

    async def test_payload_defaults_for_nullable_fields(self, store):
        point = _make_point("c1", [1.0, 0.0, 0.0, 0.0])
        point["payload"]["block_index"] = None
        point["payload"]["block_part_id"] = None
        await store.upsert_points([point])
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0])
        p = results[0]["payload"]
        assert p["block_index"] == -1
        assert p["block_part_id"] == -1


class TestLanceDBClientSpaceKey:

    async def test_space_key_round_trip(self, store):
        point = _make_point("c1", [1.0, 0.0, 0.0, 0.0], space_key="AAA")
        await store.upsert_points([point])
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0])
        assert results[0]["payload"]["space_key"] == "AAA"

    async def test_space_key_default_empty(self, store):
        point = _make_point("c1", [1.0, 0.0, 0.0, 0.0])
        await store.upsert_points([point])
        results = await store.search(vector=[1.0, 0.0, 0.0, 0.0])
        assert results[0]["payload"]["space_key"] == ""

    async def test_scroll_filter_by_space_value(self, store):
        points = [
            _make_point("c1", [1.0, 0.0, 0.0, 0.0], space_key="AAA"),
            _make_point("c2", [0.9, 0.0, 0.0, 0.0], space_key="EEE"),
        ]
        await store.upsert_points(points)
        fc = {"must": [{"key": "space_key", "match": {"value": "AAA"}}]}
        results, _ = await store.scroll_points(filter_condition=fc)
        assert [r["id"] for r in results] == ["c1"]

    async def test_scroll_filter_by_space_any(self, store):
        points = [
            _make_point("c1", [1.0, 0.0, 0.0, 0.0], space_key="AAA"),
            _make_point("c2", [0.9, 0.0, 0.0, 0.0], space_key="EEE"),
            _make_point("c3", [0.8, 0.0, 0.0, 0.0], space_key="FFF"),
        ]
        await store.upsert_points(points)
        fc = {"must": [{"key": "space_key", "match": {"any": ["AAA", "FFF"]}}]}
        results, _ = await store.scroll_points(filter_condition=fc)
        assert {r["id"] for r in results} == {"c1", "c3"}

    async def test_delete_by_space(self, store):
        points = [
            _make_point("c1", [1.0, 0.0, 0.0, 0.0], space_key="AAA"),
            _make_point("c2", [0.9, 0.0, 0.0, 0.0], space_key="EEE"),
        ]
        await store.upsert_points(points)
        fc = {"must": [{"key": "space_key", "match": {"value": "AAA"}}]}
        await store.delete_by_filter(fc)
        results, _ = await store.scroll_points(filter_condition={})
        assert [r["id"] for r in results] == ["c2"]


class TestLanceDBClientManagement:

    async def test_collection_exists(self, store):
        assert await store.collection_exists() is True

    async def test_get_collection_info(self, store):
        await store.upsert_points([_make_point("c1", [1.0, 0.0, 0.0, 0.0])])
        info = await store.get_collection_info()
        assert info["vectors_count"] == 1
        assert info["config"]["vector_size"] == 4
        assert info["config"]["distance"] == "Cosine"

    async def test_validate(self, store):
        await store.validate()

    async def test_health_check_healthy(self, store):
        health = await store.health_check()
        assert health["status"] == "healthy"
        assert "vectors_count" in health
        assert "db" in health

    async def test_create_collection_changes_vector_size(self, store):
        await store.create_collection(vector_size=128)
        info = await store.get_collection_info()
        assert info["config"]["vector_size"] == 128

    async def test_close_does_not_raise(self, store):
        await store.close()


class TestLanceDBClientFilterToSQL:
    """Test the _filter_to_sql static method directly."""

    def test_filter_empty(self):
        assert LanceDBClient._filter_to_sql({}) == ""

    def test_filter_none(self):
        assert LanceDBClient._filter_to_sql(None) == ""

    def test_filter_value(self):
        sql = LanceDBClient._filter_to_sql(
            {"key": "page_id", "match": {"value": "p123"}}
        )
        assert "page_id" in sql
        assert "p123" in sql

    def test_filter_any(self):
        sql = LanceDBClient._filter_to_sql(
            {"key": "page_id", "match": {"any": ["a", "b", "c"]}}
        )
        assert "IN" in sql.upper()
        assert "a" in sql
        assert "c" in sql

    def test_filter_text(self):
        sql = LanceDBClient._filter_to_sql(
            {"key": "page_title", "match": {"text": "документация"}}
        )
        assert "LIKE" in sql.upper()

    def test_filter_must_list(self):
        sql = LanceDBClient._filter_to_sql({
            "must": [
                {"key": "page_id", "match": {"value": "p1"}},
                {"key": "chunk_type", "match": {"value": "section"}},
            ]
        })
        assert "AND" in sql
