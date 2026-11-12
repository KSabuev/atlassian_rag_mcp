"""LanceDB vector storage — thread-safe, columnar, embedded, async wrapper."""

from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from atlassian_rag_mcp.confluence.models import ChunkPayload, make_payload
from atlassian_rag_mcp.exceptions import BaseError
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.storage.base import VectorDB

if TYPE_CHECKING:
    import pyarrow as pa

_COLUMN_MAP: dict[str, str] = {
    "page_id": "page_id",
    "page_version": "page_version",
    "updated_at": "updated_at",
    "page_title": "page_title",
    "chunk_index": "chunk_index",
    "block_index": "block_index",
    "parent_id": "parent_id",
    "chunk_type": "chunk_type",
    "block_part_id": "block_part_id",
    "heading_hierarchy": "heading_hierarchy",
    "space_key": "space_key",
    "text": "text",
}

_PAYLOAD_COLS = [
    "chunk_id", "chunk_type", "page_id", "page_title", "page_version",
    "parent_id", "block_index", "block_part_id", "chunk_index", "updated_at",
    "heading_hierarchy", "space_key", "text",
]


def _escape_sql(s: str) -> str:
    return s.replace("'", "''")


def _build_arrow_table(records: list[dict], dim: int) -> "pa.Table":
    import pyarrow as pa

    all_floats = [float(x) for r in records for x in r["vector"]]
    offsets = [0] + [(i + 1) * dim for i in range(len(records))]
    vec_arr = pa.ListArray.from_arrays(
        offsets=offsets,
        values=pa.array(all_floats, pa.float32()),
    )
    return pa.Table.from_arrays(
        [
            vec_arr,
            pa.array([r["chunk_id"] for r in records], pa.string()),
            pa.array([r["chunk_type"] for r in records], pa.string()),
            pa.array([r["page_id"] for r in records], pa.string()),
            pa.array([r["page_title"] for r in records], pa.string()),
            pa.array([r["page_version"] for r in records], pa.int64()),
            pa.array([r["parent_id"] for r in records], pa.string()),
            pa.array([r["block_index"] for r in records], pa.int64()),
            pa.array([r["block_part_id"] for r in records], pa.int64()),
            pa.array([r["chunk_index"] for r in records], pa.int64()),
            pa.array([r["updated_at"] for r in records], pa.string()),
            pa.array([r["heading_hierarchy"] for r in records], pa.string()),
            pa.array([r["space_key"] for r in records], pa.string()),
            pa.array([r["text"] for r in records], pa.string()),
        ],
        names=["vector", "chunk_id", "chunk_type", "page_id", "page_title",
               "page_version", "parent_id", "block_index", "block_part_id",
               "chunk_index", "updated_at", "heading_hierarchy", "space_key", "text"],
    )


class LanceDBClient(VectorDB):
    """LanceDB-backed vector storage. Async wrapper around sync LanceDB SDK."""

    COLLECTION_NAME: str = "vectors"

    @staticmethod
    def _nzv(val: Any, default: Any = "") -> Any:
        return default if val is None else val

    def __init__(self, db_path: str, vector_size: int = 2560) -> None:
        self._db_path = db_path
        self._vector_size = vector_size
        self._lock = threading.Lock()
        self._db = None
        self._table = None

    async def start(self) -> None:
        """Lazy init: connect to LanceDB and open/create the table."""
        if self._db is not None:
            return
        import lancedb

        os.makedirs(self._db_path, exist_ok=True)
        self._db = lancedb.connect(self._db_path)
        self._open_collection()

    def _list_tables(self) -> list[str]:
        return list(self._db.list_tables().tables)

    def _open_collection(self) -> None:
        import pyarrow as pa

        names = self._list_tables()
        if self.COLLECTION_NAME not in names:
            dim = self._vector_size
            schema = pa.schema([
                pa.field("vector", pa.list_(pa.float32(), list_size=dim)),
                pa.field("chunk_id", pa.string()),
                pa.field("chunk_type", pa.string()),
                pa.field("page_id", pa.string()),
                pa.field("page_title", pa.string()),
                pa.field("page_version", pa.int64()),
                pa.field("parent_id", pa.string()),
                pa.field("block_index", pa.int64()),
                pa.field("block_part_id", pa.int64()),
                pa.field("chunk_index", pa.int64()),
                pa.field("updated_at", pa.string()),
                pa.field("heading_hierarchy", pa.string()),
                pa.field("space_key", pa.string()),
                pa.field("text", pa.string()),
            ])
            self._db.create_table(self.COLLECTION_NAME, schema=schema)
            logger.info(f"LanceDB table created: dim={dim}, path={self._db_path}")

        self._table = self._db.open_table(self.COLLECTION_NAME)
        logger.info(f"LanceDB table opened: count={len(self._table)}")

        self._ensure_space_key_column()
        self._ensure_indices()

    def _ensure_space_key_column(self) -> None:
        """Add the ``space_key`` column to pre-existing tables (schema evolution).

        Tables created before ``space_key`` existed lack the column; without it
        filters/scrolls on ``space_key`` would fail. New nullable string column
        defaults to null for legacy rows (re-index fills the value).
        """
        try:
            col_names = {f.name for f in self._table.schema}
        except Exception:
            return
        if "space_key" in col_names:
            return
        try:
            import pyarrow as pa

            self._table.add_columns({"space_key": pa.string()})
            logger.info("LanceDB table migrated: added space_key column")
        except Exception as e:
            logger.warning(f"LanceDB could not add space_key column: {e}")

    def _ensure_indices(self) -> None:
        for col in ("page_id", "chunk_id"):
            try:
                self._table.create_scalar_index(col)
            except Exception:
                logger.debug(f"Index {col} already exists or failed")

    @staticmethod
    def _filter_to_sql(filter_cond: dict | None) -> str:
        if not filter_cond:
            return ""
        must_conds = [LanceDBClient._cond_to_sql(c) for c in filter_cond.get("must", []) if c]
        should_conds = [LanceDBClient._cond_to_sql(c) for c in filter_cond.get("should", []) if c]
        must_not_conds = [
            LanceDBClient._cond_to_sql(c)
            for c in filter_cond.get("must_not", [])
            if c
        ]
        if not must_conds and not should_conds and not must_not_conds:
            c = LanceDBClient._cond_to_sql(filter_cond)
            return c if c else ""
        parts = []
        if must_conds:
            parts.append("(" + " AND ".join(must_conds) + ")")
        if should_conds:
            parts.append("(" + " OR ".join(should_conds) + ")")
        if must_not_conds:
            parts.append("NOT (" + " AND ".join(must_not_conds) + ")")
        return " AND ".join(parts)

    @staticmethod
    def _cond_to_sql(cond: dict) -> str:
        key = _COLUMN_MAP.get(cond.get("key", ""), cond.get("key", ""))
        match = cond.get("match", {})
        rnge = cond.get("range", {})

        if "value" in match:
            v = match["value"]
            return f"{key} = '{_escape_sql(str(v))}'" if isinstance(v, str) else f"{key} = {v}"
        if "any" in match:
            escaped = [f"'{_escape_sql(str(x))}'" for x in match["any"]]
            return f"{key} IN ({', '.join(escaped)})"
        if "text" in match:
            return f"LOWER({key}) LIKE '%{match['text'].lower()}%'"
        gte, lte = rnge.get("gte"), rnge.get("lte")
        gt, lt = rnge.get("gt"), rnge.get("lt")
        parts = []
        if gte:
            parts.append(f"{key} >= '{_escape_sql(str(gte))}'")
        if lte:
            parts.append(f"{key} <= '{_escape_sql(str(lte))}'")
        if gt:
            parts.append(f"{key} > '{_escape_sql(str(gt))}'")
        if lt:
            parts.append(f"{key} < '{_escape_sql(str(lt))}'")
        return " AND ".join(parts) if parts else ""

    @staticmethod
    def _row_payload(row: dict) -> dict:
        """Rebuild a canonical chunk payload from a LanceDB table row.

        Uses the ChunkPayload model directly (same approach as every other
        store): only payload fields are read from the row, then the model
        validates/normalizes them and produces the canonical dict.
        """
        data = {key: row.get(key) for key in ChunkPayload.model_fields}
        return make_payload(data, row.get("chunk_id") or "").model_dump()

    @staticmethod
    def _row_to_result(row: dict, score: float) -> dict[str, Any]:
        return {
            "id": row.get("chunk_id", ""),
            "score": round(score, 6),
            "payload": LanceDBClient._row_payload(row),
        }

    async def _run_io(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def upsert_points(self, points: list[dict[str, Any]]) -> bool:
        """Upsert vector points into LanceDB table.

        Row identity is the LOGICAL chunk id (``payload["chunk_id"]``), not
        the point id: sub-points of a split composite share one logical
        chunk_id, and every downstream filter (``_delete_old`` exclusion,
        ``_rollback_vdb``, result dedup) matches on that logical id — the
        column must carry it, otherwise fresh composite sub-points get
        deleted by the post-write cleanup of the very run that wrote them.

        Points repeated within the batch (same point id) are collapsed,
        keeping the last occurrence; composite sub-points share a logical
        chunk_id but carry distinct point ids, so they all survive.
        """
        if not points:
            return True

        records = []
        logical_ids: list[str] = []
        point_ids: list[str] = []
        for point in points:
            pid_raw = point.get("id")
            vector = point.get("vector")
            if not pid_raw or not vector:
                continue
            payload = point.get("payload", {})
            logical_id = str(payload.get("chunk_id") or pid_raw)
            records.append({
                "vector": vector,
                "chunk_id": logical_id,
                "chunk_type": self._nzv(payload.get("chunk_type"), ""),
                "page_id": self._nzv(payload.get("page_id"), ""),
                "page_title": self._nzv(payload.get("page_title"), ""),
                "page_version": payload.get("page_version") or 0,
                "parent_id": self._nzv(payload.get("parent_id"), ""),
                "block_index": self._nzv(payload.get("block_index"), -1),
                "block_part_id": self._nzv(payload.get("block_part_id"), -1),
                "chunk_index": payload.get("chunk_index") or 0,
                "updated_at": self._nzv(payload.get("updated_at"), ""),
                "heading_hierarchy": json.dumps(
                    payload.get("heading_hierarchy", []) or [],
                    ensure_ascii=False,
                ),
                "space_key": self._nzv(payload.get("space_key"), ""),
                "text": self._nzv(payload.get("text"), ""),
            })
            logical_ids.append(logical_id)
            point_ids.append(str(pid_raw))

        if not records:
            return True

        last_index: dict[str, int] = {}
        for i, pid in enumerate(point_ids):
            last_index[pid] = i
        keep = set(last_index.values())
        records = [r for i, r in enumerate(records) if i in keep]

        new_table = _build_arrow_table(records, self._vector_size)

        def _do_write():
            with self._lock:
                unique_ids = sorted(set(logical_ids))
                if unique_ids:
                    escaped = [f"'{_escape_sql(c)}'" for c in unique_ids]
                    self._table.delete(f"chunk_id IN ({', '.join(escaped)})")
                self._table.add(new_table)

        await self._run_io(_do_write)
        logger.info(f"LanceDB upserted {len(records)} points")
        return True

    async def search(
        self,
        vector: list[float],
        limit: int = 25,
        query_filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search against LanceDB table."""
        where_sql = self._filter_to_sql(query_filter)

        def _search():
            q = self._table.search(vector).metric("cosine")
            if where_sql:
                q = q.where(where_sql)
            return q.limit(limit).to_list()

        rows = await self._run_io(_search)

        results = []
        for row in rows:
            dist = row.get("_distance", 0.0)
            score = max(0.0, min(1.0, 1.0 - dist))
            results.append(self._row_to_result(row, score))
        return results

    async def scroll_points(
        self,
        filter_condition: dict,
        limit: int = 10000,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Scroll through points matching filter."""
        where_sql = self._filter_to_sql(filter_condition)
        offset_val = int(offset) if offset is not None else 0

        logger.debug(f"filter={where_sql} limit={limit} offset={offset_val}")

        cols = list(_PAYLOAD_COLS)
        if with_vectors:
            cols.append("vector")

        def _scroll():
            tbl = self._table
            q = tbl.search(None).select(cols).limit(limit).offset(offset_val)
            if where_sql:
                q = q.where(where_sql)
            return q.to_list()

        rows = await self._run_io(_scroll)

        out: list[dict[str, Any]] = []
        for row in rows:
            rec: dict[str, Any] = {"id": row.get("chunk_id", "")}
            try:
                rec["payload"] = self._row_payload(row)
            except Exception:
                rec["payload"] = {}
            if with_vectors:
                vec = row.get("vector")
                if vec is not None:
                    if hasattr(vec, "tolist"):
                        rec["vector"] = vec.tolist()
                    else:
                        rec["vector"] = vec
            out.append(rec)

        next_offset = str(offset_val + len(out)) if len(out) == limit else None
        return out, next_offset

    async def delete_by_filter(self, filter_condition: dict) -> bool:
        """Delete points matching filter condition."""
        where_sql = self._filter_to_sql(filter_condition)
        if not where_sql:
            return True

        logger.debug(f"filter={where_sql}")

        def _do_delete():
            self._table.delete(where_sql)

        await self._run_io(_do_delete)
        logger.info(f"LanceDB deleted: filter={filter_condition}")
        return True

    async def create_collection(
        self, vector_size: int | None = None, distance: str = "Cosine"
    ) -> bool:
        if vector_size is not None:
            self._vector_size = vector_size
        return True

    async def collection_exists(self) -> bool:
        try:
            return self.COLLECTION_NAME in self._list_tables()
        except Exception:
            return False

    async def get_collection_info(self) -> dict[str, Any] | None:
        try:
            count = len(self._table)
            return {
                "vectors_count": count,
                "config": {"vector_size": self._vector_size, "distance": "Cosine"},
            }
        except Exception:
            return None

    async def validate(self) -> None:
        if self._table is None:
            raise BaseError(f"LanceDB table not initialized at {self._db_path}")
        try:
            count = len(self._table)
        except Exception as e:
            raise BaseError(f"Cannot access LanceDB table at {self._db_path}: {e}") from e
        logger.info(f"LanceDB OK — {self._db_path} (vectors: {count})")

    async def health_check(self) -> dict[str, Any]:
        """Check LanceDB table health and return status."""
        try:
            count = len(self._table)
            return {"status": "healthy", "vectors_count": count, "db": self._db_path}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    async def close(self) -> None:
        """Close LanceDB connection and release resources.

        Safe to call multiple times — idempotent under self._lock.
        """
        def _work():
            with self._lock:
                if self._db is None:
                    return
                self._db._conn.close()
                self._db = None
                self._table = None

        await self._run_io(_work)
