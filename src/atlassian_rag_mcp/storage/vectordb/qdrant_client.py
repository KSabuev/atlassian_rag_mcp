"""Qdrant vector storage via AsyncQdrantClient — implements VectorDB interface."""

from __future__ import annotations

from typing import Any

from qdrant_client import AsyncQdrantClient as _SDKClient
from qdrant_client.http import models

from atlassian_rag_mcp.exceptions import BaseError
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.storage.base import VectorDB


class QdrantClient(VectorDB):
    """Qdrant vector storage backed by AsyncQdrantClient.

    ``PAYLOAD_INDEX_FIELDS`` lists the payload fields that filtered reads
    (scroll/search filters on page_id, page_version, chunk_type, block_index,
    ...) hit on every search call — without indexes Qdrant full-scans the
    whole collection for each filter.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6333,
        api_key: str = "",
        collection: str = "atlassian_chunks",
        vector_size: int = 2560,
        *,
        timeout: int = 60,
        upsert_batch_size: int = 200,
    ) -> None:
        self.collection_name = collection
        self._vector_size = vector_size
        self._upsert_batch_size = upsert_batch_size

        self._client = _SDKClient(
            host=host,
            port=port,
            api_key=api_key or None,
            https=False,
            timeout=timeout,
        )
        logger.info(
            f"Qdrant client created: {host}:{port}"
        )

    async def start(self) -> None:
        """Qdrant client initializes in __init__ — nothing to lazy-init."""

    @staticmethod
    def _dict_to_filter(filter_dict: dict) -> models.Filter | None:
        if not filter_dict:
            return None

        must = [
            QdrantClient._condition_to_filter(c)
            for c in filter_dict.get("must", [])
            if c
        ]
        should = [
            QdrantClient._condition_to_filter(c)
            for c in filter_dict.get("should", [])
            if c
        ]
        must_not = [
            QdrantClient._condition_to_filter(c)
            for c in filter_dict.get("must_not", [])
            if c
        ]
        if not must and not should and not must_not:
            c = QdrantClient._condition_to_filter(filter_dict)
            if c:
                return models.Filter(must=[c])
            return None

        kwargs: dict[str, Any] = {}
        if must:
            kwargs["must"] = must
        if should:
            kwargs["should"] = should
        if must_not:
            kwargs["must_not"] = must_not
        return models.Filter(**kwargs)

    @staticmethod
    def _condition_to_filter(cond: dict) -> models.FieldCondition | None:
        key = cond.get("key")
        if not key:
            return None

        match = cond.get("match", {})
        rnge = cond.get("range", {})

        if "value" in match:
            return models.FieldCondition(
                key=key, match=models.MatchValue(value=match["value"])
            )
        if "any" in match:
            return models.FieldCondition(
                key=key, match=models.MatchAny(any=match["any"])
            )
        if "text" in match:
            return models.FieldCondition(
                key=key, match=models.MatchText(text=match["text"])
            )
        if rnge:
            return models.FieldCondition(
                key=key,
                range=models.Range(
                    gte=rnge.get("gte"),
                    lte=rnge.get("lte"),
                    gt=rnge.get("gt"),
                    lt=rnge.get("lt"),
                ),
            )
        return None

    PAYLOAD_INDEX_FIELDS = {
        "page_id": models.PayloadSchemaType.KEYWORD,
        "page_version": models.PayloadSchemaType.INTEGER,
        "chunk_type": models.PayloadSchemaType.KEYWORD,
        "chunk_id": models.PayloadSchemaType.KEYWORD,
        "block_index": models.PayloadSchemaType.INTEGER,
        "block_part_id": models.PayloadSchemaType.INTEGER,
        "space_key": models.PayloadSchemaType.KEYWORD,
    }

    async def ensure_payload_indexes(self) -> None:
        """Idempotently create payload indexes (skipped when already present).

        ``wait=False``: the build runs in the background so startup never blocks
        (queries full-scan until an index is ready, then speed up automatically).
        """
        for field, schema in self.PAYLOAD_INDEX_FIELDS.items():
            try:
                await self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                    wait=False,
                )
                logger.info(f"Payload index requested on '{field}' ({schema})")
            except Exception as e:
                if "already exists" in str(e).lower():
                    continue
                logger.warning(f"Failed to create payload index on '{field}': {e}")

    async def validate(self, host_info: str = "") -> None:
        try:
            info = await self._client.info()
            logger.info(f"Qdrant OK — {info}")

            exists = await self._client.collection_exists(self.collection_name)
            if exists:
                await self.ensure_payload_indexes()
                cinfo = await self.get_collection_info()
                count = cinfo.get("vectors_count", "?") if cinfo else "?"
                logger.info(
                    f"Collection '{self.collection_name}' exists (vectors: {count})"
                )
            else:
                logger.warning(
                    f"Collection '{self.collection_name}' not found — "
                    f"it will be created on first index operation"
                )
        except ConnectionError:
            raise BaseError(
                f"Cannot connect to Qdrant at {host_info}. "
                f"Check QDRANT_HOST and QDRANT_PORT."
            )
        except Exception as e:
            raise BaseError(f"Qdrant validation failed: {e}") from e

    async def collection_exists(self) -> bool:
        try:
            return await self._client.collection_exists(self.collection_name)
        except Exception:
            logger.debug("collection_exists failed, returning False")
            return False

    async def create_collection(
        self, vector_size: int | None = None, distance: str = "Cosine"
    ) -> bool:
        if vector_size is None:
            vector_size = self._vector_size

        if await self._client.collection_exists(self.collection_name):
            return True

        dist = models.Distance.COSINE
        if distance.upper() == "DOT":
            dist = models.Distance.DOT
        elif distance.upper() == "EUCLID":
            dist = models.Distance.EUCLID

        try:
            await self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, distance=dist
                ),
            )
            await self.ensure_payload_indexes()
            logger.info(
                f"Collection '{self.collection_name}' created with "
                f"size={vector_size}, distance={distance}"
            )
            return True
        except Exception as e:
            raise BaseError(
                f"Failed to create collection '{self.collection_name}' "
                f"size={vector_size}, distance={distance}: {e}"
            ) from e

    async def upsert_points(self, points: list[dict[str, Any]]) -> bool:
        """Upsert vector points into Qdrant collection (all-or-nothing).

        If a later batch fails after earlier batches were already written,
        the already-written points for this call are deleted (compensation)
        so a logical page is never left partially present in the collection.
        """
        if not points:
            return True

        sdk_points = [
            models.PointStruct(
                id=p["id"],
                vector=p.get("vector", []),
                payload=p.get("payload", {}),
            )
            for p in points
            if p.get("id") is not None and p.get("vector")
        ]
        if not sdk_points:
            return True

        chunk_ids = [p["id"] for p in points if p.get("id") is not None and p.get("vector")]
        written = 0
        try:
            for i in range(0, len(sdk_points), self._upsert_batch_size):
                batch = sdk_points[i : i + self._upsert_batch_size]
                await self._client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                )
                written += len(batch)
            logger.info(
                f"Upserted {len(sdk_points)} points to '{self.collection_name}' "
                f"(batch size {self._upsert_batch_size})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upsert points: {e!r}", exc_info=True)
            if written > 0:
                try:
                    await self._client.delete(
                        collection_name=self.collection_name,
                        points_selector=models.PointIdsList(
                            points=[str(c) for c in chunk_ids]
                        ),
                        wait=True,
                    )
                    logger.warning(
                        f"Rolled back {written} partially-upserted points "
                        f"from '{self.collection_name}'"
                    )
                except Exception as rb_e:
                    logger.error(
                        f"Rollback of partially-upserted points failed: {rb_e!r}",
                        exc_info=True,
                    )
            raise BaseError(
                f"Failed to upsert {len(points)} points to "
                f"'{self.collection_name}': {e}"
            ) from e

    async def search(
        self,
        vector: list[float],
        limit: int = 25,
        query_filter: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search against Qdrant collection."""
        qfilter = self._dict_to_filter(query_filter)

        try:
            resp = await self._client.query_points(
                collection_name=self.collection_name,
                query=vector,
                limit=limit,
                query_filter=qfilter,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

        results = []
        for point in resp.points:
            raw_score = point.score if point.score is not None else 0.0
            score = round((raw_score + 1.0) / 2.0, 6)
            results.append({
                "id": str(point.id) if point.id is not None else "",
                "score": score,
                "payload": point.payload or {},
            })
        return results

    async def delete_by_filter(self, filter_condition: dict) -> bool:
        """Delete points matching filter condition."""
        qfilter = self._dict_to_filter(filter_condition)
        if qfilter is None:
            logger.info("No filter condition — skipping deletion")
            return True

        try:
            await self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=qfilter),
            )
            logger.info(
                f"Deleted points by filter from '{self.collection_name}'"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete by filter: {e}")
            return False

    async def scroll_points(
        self,
        filter_condition: dict,
        limit: int = 10000,
        offset: str | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Scroll through points matching filter."""
        qfilter = self._dict_to_filter(filter_condition)

        try:
            records, next_offset = await self._client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
                scroll_filter=qfilter,
            )
        except Exception as e:
            logger.error(f"Failed to scroll points: {e}")
            return [], None

        points = []
        for record in records:
            p: dict[str, Any] = {
                "id": str(record.id) if record.id is not None else "",
                "payload": record.payload or {},
            }
            if with_vectors and record.vector is not None:
                vec = record.vector
                if hasattr(vec, "tolist"):
                    vec = vec.tolist()
                p["vector"] = vec
            points.append(p)

        return points, next_offset

    async def get_collection_info(self) -> dict[str, Any] | None:
        try:
            info = await self._client.get_collection(self.collection_name)
            return {
                "vectors_count": info.points_count or 0,
                "config": {
                    "vector_size": (
                        info.config.params.vectors.size
                        if info.config and info.config.params and info.config.params.vectors
                        else None
                    ),
                    "distance": (
                        str(info.config.params.vectors.distance)
                        if info.config and info.config.params and info.config.params.vectors
                        else None
                    ),
                },
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None

    async def health_check(self) -> dict[str, Any]:
        """Check Qdrant server and collection health."""
        try:
            info = await self._client.info()
            result: dict[str, Any] = {"status": "healthy", "version": str(info)}
            try:
                cinfo = await self.get_collection_info()
                if cinfo:
                    result["vectors_count"] = cinfo["vectors_count"]
            except Exception:
                pass
            return result
        except Exception as e:
            return {"status": "unhealthy", "version": None, "error": str(e)}

    async def close(self) -> None:
        await self._client.close()
