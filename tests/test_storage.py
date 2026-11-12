"""Unit test for JSON storage backends — fully isolated (no config/settings)."""

from __future__ import annotations

import json
import math
import os
import tempfile

import pytest


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class JsonStorage:
    def __init__(self, file_path: str) -> None:
        self._file_path = file_path
        import threading
        self._lock = threading.Lock()
        self._data = {}
        self._load()

    def _load(self):
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}

    def _save(self):
        os.makedirs(os.path.dirname(self._file_path) or ".", exist_ok=True)
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def upsert(self, data):
        with self._lock:
            self._data.update(data)
            self._save()

    def get(self, key, default=None):
        with self._lock:
            return self._data.get(key, default)

    def delete(self, keys):
        with self._lock:
            for k in keys:
                self._data.pop(k, None)
            self._save()

    def filter_keys(self, keys):
        with self._lock:
            return set(keys) - set(self._data.keys())

    def __len__(self):
        with self._lock:
            return len(self._data)

    def __contains__(self, key):
        with self._lock:
            return key in self._data


class JsonVectorStorage:
    def __init__(self, file_path: str, vector_size: int = 1024) -> None:
        self._file_path = file_path
        self._vector_size = vector_size
        import threading
        self._lock = threading.Lock()
        self._data = {}
        self._initialized = True
        self._load()

    def _load(self):
        if os.path.exists(self._file_path):
            try:
                with open(self._file_path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = {}

    def _save(self):
        os.makedirs(os.path.dirname(self._file_path) or ".", exist_ok=True)
        with open(self._file_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def upsert_points(self, points):
        if not points:
            return True
        with self._lock:
            for point in points:
                pid = point.get("id")
                if pid:
                    self._data[pid] = point
            self._save()
        return True

    def search(self, vector, limit=25, query_filter=None):
        with self._lock:
            data_copy = dict(self._data)

        scored = []
        for pid, point in data_copy.items():
            if query_filter and not self._apply_filter(point, query_filter):
                continue
            point_vector = point.get("vector")
            if not point_vector:
                continue
            score = cosine_similarity(vector, point_vector)
            scored.append({
                "id": pid,
                "score": score,
                "payload": point.get("payload", {}),
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def scroll_points(self, filter_condition, limit=10000, offset=None, with_vectors=False):
        with self._lock:
            data_copy = dict(self._data)

        all_matched = []
        for pid, point in data_copy.items():
            if filter_condition and not self._apply_filter(point, filter_condition):
                continue
            result = {"id": pid, "payload": point.get("payload", {})}
            if with_vectors:
                result["vector"] = point.get("vector")
            all_matched.append(result)

        offset_val = int(offset) if offset is not None else 0
        page = all_matched[offset_val:offset_val + limit]
        next_offset = (
            str(offset_val + len(page))
            if len(page) == limit and (offset_val + len(page)) < len(all_matched)
            else None
        )
        return page, next_offset

    def delete_by_filter(self, filter_condition):
        with self._lock:
            to_delete = [
                pid for pid, point in self._data.items()
                if self._apply_filter(point, filter_condition)
            ]
            for pid in to_delete:
                del self._data[pid]
            if to_delete:
                self._save()
        return True

    def get_collection_info(self):
        with self._lock:
            return {
                "vectors_count": len(self._data),
                "config": {"vector_size": self._vector_size, "distance": "Cosine"},
            }

    def health_check(self):
        try:
            with self._lock:
                count = len(self._data)
            return {"status": "healthy", "vectors_count": count, "file": self._file_path}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    @staticmethod
    def _apply_filter(point, filter_cond):
        must = filter_cond.get("must")
        should = filter_cond.get("should")
        if must is not None:
            return all(JsonVectorStorage._match_condition(point, c) for c in must)
        if should is not None:
            return any(JsonVectorStorage._match_condition(point, c) for c in should)
        return JsonVectorStorage._match_condition(point, filter_cond)

    @staticmethod
    def _match_condition(point, condition):
        key = condition.get("key")
        if not key:
            return True
        match = condition.get("match")
        range_cond = condition.get("range")
        payload = point.get("payload", {})
        value = payload.get(key)

        if match is not None:
            if "value" in match:
                return value == match["value"]
            if "any" in match:
                return value in match["any"]
            if "text" in match:
                if value is None:
                    return False
                return match["text"].lower() in str(value).lower()

        if range_cond is not None:
            if value is None:
                return False
            try:
                val = float(value) if isinstance(value, str) else value
                if "gte" in range_cond and val < float(range_cond["gte"]):
                    return False
                if "lte" in range_cond and val > float(range_cond["lte"]):
                    return False
                return True
            except (ValueError, TypeError):
                return False
        return True


@pytest.fixture
def tmp_json_file():
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    yield tmp.name
    if os.path.exists(tmp.name):
        os.remove(tmp.name)


def make_point(cid, vector, page_id="p1"):
    return {
        "id": cid,
        "vector": vector,
        "payload": {
            "page_id": page_id,
            "text": f"Chunk {cid}",
            "page_title": "Test Page",
            "chunk_type": "section",
            "chunk_index": 0,
            "heading_hierarchy": [],
            "block_index": None,
            "block_part_id": None,
            "updated_at": "2024-01-01T00:00:00",
        },
    }


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        v1 = [1.0, 0.0, 0.0]
        v2 = [-1.0, 0.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_zero_vector(self):
        v1 = [0.0, 0.0, 0.0]
        v2 = [1.0, 2.0, 3.0]
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_realistic_embedding(self):
        v1 = [0.1] * 128 + [0.2] * 128
        v2 = [0.15] * 128 + [0.15] * 128
        assert cosine_similarity(v1, v2) > 0.9


class TestJsonStorage:
    def test_upsert_and_get(self, tmp_json_file):
        store = JsonStorage(file_path=tmp_json_file)
        store.upsert({"a": 1, "b": 2})
        assert store.get("a") == 1
        assert store.get("b") == 2

    def test_delete(self, tmp_json_file):
        store = JsonStorage(file_path=tmp_json_file)
        store.upsert({"a": 1, "b": 2})
        store.delete(["a"])
        assert store.get("a") is None
        assert store.get("b") == 2

    def test_filter_keys(self, tmp_json_file):
        store = JsonStorage(file_path=tmp_json_file)
        store.upsert({"a": 1})
        assert store.filter_keys(["a", "b", "c"]) == {"b", "c"}

    def test_len_and_contains(self, tmp_json_file):
        store = JsonStorage(file_path=tmp_json_file)
        store.upsert({"a": 1, "b": 2})
        assert len(store) == 2
        assert "a" in store
        assert "c" not in store

    def test_persistence(self, tmp_json_file):
        store = JsonStorage(file_path=tmp_json_file)
        store.upsert({"key": "value"})
        store2 = JsonStorage(file_path=tmp_json_file)
        assert store2.get("key") == "value"


@pytest.fixture
def vector_store(tmp_json_file):
    return JsonVectorStorage(file_path=tmp_json_file, vector_size=3)


class TestJsonVectorStorage:
    def test_upsert_and_search(self, vector_store):
        v = [1.0, 0.0, 0.0]
        vector_store.upsert_points([make_point("c1", v)])
        results = vector_store.search(vector=[1.0, 0.0, 0.0], limit=5)
        assert len(results) == 1
        assert results[0]["id"] == "c1"
        assert results[0]["score"] == pytest.approx(1.0)

    def test_search_ranking(self, vector_store):
        points = [
            make_point("c1", [1.0, 0.0, 0.0]),
            make_point("c2", [0.9, 0.1, 0.0]),
            make_point("c3", [0.0, 1.0, 0.0]),
        ]
        vector_store.upsert_points(points)
        results = vector_store.search(vector=[1.0, 0.0, 0.0], limit=3)
        assert len(results) == 3
        assert results[0]["id"] == "c1"
        assert results[2]["id"] == "c3"

    def test_search_with_filter_page_ids(self, vector_store):
        points = [
            make_point("c1", [1.0, 0.0, 0.0], page_id="p1"),
            make_point("c2", [0.9, 0.1, 0.0], page_id="p2"),
        ]
        vector_store.upsert_points(points)
        query_filter = {"must": [{"key": "page_id", "match": {"value": "p1"}}]}
        results = vector_store.search(
            vector=[1.0, 0.0, 0.0], limit=5, query_filter=query_filter
        )
        assert len(results) == 1
        assert results[0]["id"] == "c1"

    def test_scroll_points(self, vector_store):
        points = [
            make_point("c1", [1.0, 0.0, 0.0], page_id="p1"),
            make_point("c2", [0.9, 0.1, 0.0], page_id="p1"),
            make_point("c3", [0.0, 1.0, 0.0], page_id="p2"),
        ]
        vector_store.upsert_points(points)
        filter_cond = {"must": [{"key": "page_id", "match": {"value": "p1"}}]}
        results, _ = vector_store.scroll_points(filter_condition=filter_cond, limit=10)
        assert len(results) == 2
        assert {r["id"] for r in results} == {"c1", "c2"}

    def test_scroll_with_vectors(self, vector_store):
        vector_store.upsert_points([make_point("c1", [1.0, 0.0, 0.0])])
        results, _ = vector_store.scroll_points(
            filter_condition={}, limit=10, with_vectors=True
        )
        assert len(results) == 1
        assert results[0]["vector"] == [1.0, 0.0, 0.0]

    def test_delete_by_filter(self, vector_store):
        points = [
            make_point("c1", [1.0, 0.0, 0.0], page_id="p1"),
            make_point("c2", [0.9, 0.1, 0.0], page_id="p2"),
        ]
        vector_store.upsert_points(points)
        filter_cond = {"must": [{"key": "page_id", "match": {"value": "p1"}}]}
        vector_store.delete_by_filter(filter_cond)
        results, _ = vector_store.scroll_points(filter_condition={}, limit=10)
        assert len(results) == 1
        assert results[0]["id"] == "c2"

    def test_collection_info(self, vector_store):
        vector_store.upsert_points([make_point("c1", [1.0, 0.0, 0.0])])
        info = vector_store.get_collection_info()
        assert info["vectors_count"] == 1
        assert info["config"]["vector_size"] == 3

    def test_health_check(self, vector_store):
        health = vector_store.health_check()
        assert health["status"] == "healthy"
        assert "file" in health

    def test_persistence(self, vector_store, tmp_json_file):
        vector_store.upsert_points([make_point("c1", [1.0, 0.0, 0.0])])
        store2 = JsonVectorStorage(file_path=tmp_json_file, vector_size=3)
        results = store2.search(vector=[1.0, 0.0, 0.0], limit=5)
        assert len(results) == 1
        assert results[0]["id"] == "c1"

    def test_filter_text_match(self, vector_store):
        points = [make_point("c1", [1.0, 0.0, 0.0])]
        points[0]["payload"]["page_title"] = "Документация ABC"
        vector_store.upsert_points(points)
        filter_cond = {"must": [{"key": "page_title", "match": {"text": "abc"}}]}
        results, _ = vector_store.scroll_points(filter_condition=filter_cond, limit=10)
        assert len(results) == 1

    def test_filter_range(self, vector_store):
        points = [
            make_point("c1", [1.0, 0.0, 0.0]),
            make_point("c2", [0.9, 0.1, 0.0]),
        ]
        points[0]["payload"]["chunk_index"] = 0
        points[1]["payload"]["chunk_index"] = 5
        vector_store.upsert_points(points)
        filter_cond = {"must": [{"key": "chunk_index", "range": {"gte": 0, "lte": 2}}]}
        results, _ = vector_store.scroll_points(filter_condition=filter_cond, limit=10)
        assert len(results) == 1
        assert results[0]["id"] == "c1"
