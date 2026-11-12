"""High-level service for vector indexing and search operations (async)."""

from __future__ import annotations

import uuid
from typing import Any, Callable

from atlassian_rag_mcp.embedding.base import EmbeddingProvider
from atlassian_rag_mcp.exceptions import BaseError
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.storage.base import VectorDB
from atlassian_rag_mcp.storage.fts.base import FTSIndex
from atlassian_rag_mcp.storage.utils import build_point, dedupe_chunks_by_id
from atlassian_rag_mcp.utils.text import html_to_text
from atlassian_rag_mcp.utils.tokenizer import (
    count_qwen3_tokens,
    get_qwen3_tokenizer,
    split_overlap_tokens,
)

from ..models import ChunkPayload, make_payload


class VectorService:
    """High-level service: indexing pages, search, delete via VectorDB (async).

    The token counter is the bundled Qwen3 tokenizer (fail fast on startup);
    tests override ``_token_counter`` directly.
    """

    def __init__(
        self,
        vector_db: VectorDB,
        embedding_client: EmbeddingProvider,
        fts_index: FTSIndex,
        batch_size: int = 50,
        *,
        scroll_limit: int = 1000,
        max_tokens: int = 8000,
        overlap_tokens: int = 200,
    ) -> None:
        self._scroll_limit = scroll_limit
        self.vector_db = vector_db
        self.embedding = embedding_client
        self.fts_index = fts_index
        self.batch_size = batch_size
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        get_qwen3_tokenizer()
        self._token_counter: Callable[[str], int] = count_qwen3_tokens

    async def _ensure_collection(self) -> None:
        dim = await self.embedding.get_embedding_dimension()
        if not await self.vector_db.create_collection(vector_size=dim):
            raise BaseError(
                f"Failed to create vector collection (vector_size={dim})"
            )

    async def index_page(
        self,
        page_id: str,
        chunks: list[ChunkPayload],
    ) -> bool:
        """Index a page's chunks into vector DB and FTS (all-or-nothing).

        Order:
          1. Preparation (no writes): embed the chunk texts, build the
             vector points and the FTS documents. If ANY of these fail the
             whole operation aborts and NOTHING is written.
          2. Write: upsert vectors to VectorDB, then index into FTS.
          3. Post-write cleanup: delete any stale chunks for this page
             (write-then-delete, so the page never disappears from search
             during a failed re-index).

        If preparation or write fails, the page's previous data is left
        intact and the causal error is raised to the caller.

        Stale-chunk deletion matches on the payload ``chunk_id`` (which
        carries the ``{base}_{i}`` suffix for split chunks), not on the
        Qdrant point IDs.
        """
        logger.info(f"Indexing page {page_id} with {len(chunks)} chunks")

        try:
            points, fts_docs = await self._prepare(page_id, chunks)
            new_chunk_ids = [p["payload"]["chunk_id"] for p in points]
            await self._write(page_id, points, fts_docs)
            await self._delete_old(page_id, new_chunk_ids)
            return True

        except Exception as e:
            logger.error(f"Failed to index page {page_id}: {e}", exc_info=True)
            raise

    async def _prepare(
        self,
        page_id: str,
        chunks: list[ChunkPayload],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Prepare (compute) all data in memory WITHOUT writing anything.

        A chunk is embedded whole (one vector) when its REAL token count
        (Qwen3 tokenizer, bundled with the project) fits ``max_tokens``.
        The chunker already enforces this budget, so only pathological
        units (one unbreakable paragraph/row over the limit) reach the
        split:
        overlapping segments of at most ``max_tokens`` tokens, each of which
        stores the SAME full ``text`` and the SAME ``chunk_id`` (the base
        uuid), differing only by its point id and vector — so any matched
        sub-chunk already carries the whole text. Sub-chunks of one composite
        get distinct valid backend point_ids (UUID).

        The embedding input is the HTML-stripped form of the chunk text —
        markup is noise for the model and eats the token budget; the stored
        payload keeps the full original text. One FTS doc is emitted per
        logical chunk: sub-chunks share chunk_id and the full text, so
        indexing each slice would only duplicate FTS hits.

        Returns (points, fts_docs). Raises with a causal message if any
        step (embedding or document building) fails before anything is
        committed to storage.
        """
        counter = self._token_counter
        budget = self._max_tokens

        expanded: list[tuple[ChunkPayload, str, str]] = []
        for c in chunks:
            plain = html_to_text(c.text) or c.text

            if counter(plain) <= budget:
                expanded.append((c, plain, c.chunk_id))
            else:
                segments = split_overlap_tokens(
                    plain,
                    counter,
                    max_tokens=budget,
                    overlap_tokens=self._overlap_tokens,
                )
                for i, seg in enumerate(segments):
                    sub = c.model_copy(update={"text": c.text})
                    point_id = str(uuid.uuid5(
                        uuid.NAMESPACE_URL, f"{c.chunk_id}_{i}"
                    ))
                    expanded.append((sub, seg, point_id))

        embed_texts = [et for _, et, _ in expanded]
        points: list[dict[str, Any]] = []
        fts_docs: list[dict[str, Any]] = []

        if embed_texts:
            if len(embed_texts) <= self.batch_size:
                vectors = await self.embedding.get_embeddings(embed_texts)
            else:
                vectors = []
                for i in range(0, len(embed_texts), self.batch_size):
                    batch = embed_texts[i:i + self.batch_size]
                    vectors.extend(await self.embedding.get_embeddings(batch))

            if not vectors or any(len(v) == 0 for v in vectors):
                raise RuntimeError(
                    f"Embedding returned empty vectors for page {page_id}"
                )

            if len(vectors) != len(embed_texts):
                raise RuntimeError(
                    f"Vector count mismatch for page {page_id}: "
                    f"{len(vectors)} vectors vs {len(embed_texts)} texts"
                )

            seen_chunk_ids: set[str] = set()
            for (chunk, _et, point_id), vector in zip(expanded, vectors):
                point = build_point(chunk, vector=vector, point_id=point_id)
                points.append(point)
                cid = point["payload"]["chunk_id"]
                if cid not in seen_chunk_ids:
                    seen_chunk_ids.add(cid)
                    fts_docs.append(point["payload"])

        return points, fts_docs

    async def _write(
        self,
        page_id: str,
        points: list[dict[str, Any]],
        fts_docs: list[dict[str, Any]],
    ) -> None:
        """Persist prepared points and FTS docs (all-or-nothing with rollback).

        Order: VectorDB first, then FTS. If the FTS write fails after the
        VectorDB write succeeded, the freshly-written VectorDB points are
        rolled back (compensation) — so a page is never left in a partial
        cross-store state. The page's previous data is untouched.
        """
        new_chunk_ids = [p["payload"]["chunk_id"] for p in points]

        if points and not await self.vector_db.upsert_points(points):
            raise RuntimeError(f"Failed to upsert points for page {page_id}")

        try:
            if not await self.fts_index.index_batch(fts_docs):
                raise RuntimeError(f"FTS index failed for page {page_id}")
        except Exception as e:
            logger.warning(
                f"FTS write failed for page {page_id}; rolling back "
                f"{len(new_chunk_ids)} fresh VectorDB points: {e}"
            )
            await self._rollback_vdb(page_id, new_chunk_ids)
            raise

    async def _rollback_vdb(
        self, page_id: str, chunk_ids: list[str]
    ) -> None:
        """Best-effort removal of freshly-written points (only these chunk_ids).

        Used as compensation so a failed page write never leaves partial data.
        Old chunks for the page are not affected.
        """
        if not chunk_ids:
            return
        try:
            filt = {
                "must": [
                    {"key": "chunk_id", "match": {"any": chunk_ids}}
                ],
            }
            await self.vector_db.delete_by_filter(filt)
        except Exception as e:
            logger.warning(
                f"VDB rollback failed for page {page_id} "
                f"({len(chunk_ids)} chunks): {e}",
                exc_info=True,
            )

    async def _delete_old(
        self, page_id: str, new_chunk_ids: list[str]
    ) -> None:
        """Post-write cleanup: remove stale chunks for the page (best-effort).

        Runs AFTER a successful write (write-then-delete), so a failed
        re-index never leaves the page missing from search.

        Deletes the page's chunks whose chunk_id is NOT in new_chunk_ids —
        i.e. only the stale data from previous runs — leaving the freshly
        written chunks intact.
        """
        if new_chunk_ids:
            filt = {
                "must": [{"key": "page_id", "match": {"value": page_id}}],
                "must_not": [
                    {"key": "chunk_id", "match": {"any": new_chunk_ids}}
                ],
            }
        else:
            filt = {"must": [{"key": "page_id", "match": {"value": page_id}}]}
        try:
            await self.vector_db.delete_by_filter(filt)
        except Exception as e:
            logger.warning(
                f"VDB post-write cleanup failed for page {page_id}: {e}",
                exc_info=True,
            )
        try:
            await self.fts_index.delete_by_filter(filt)
        except Exception as e:
            logger.warning(
                f"FTS post-write cleanup failed for page {page_id}: {e}",
                exc_info=True,
            )

    async def delete_page(self, page_id: str) -> bool:
        """Delete all chunks for a page from both VectorDB and FTS.
        Each store is written independently — reconciler handles any partial miss.
        """
        logger.debug(f"page={page_id}")
        filt = {"must": [{"key": "page_id", "match": {"value": page_id}}]}
        fts_ok = await self.fts_index.delete_by_filter(filt)
        vdb_ok = await self.vector_db.delete_by_filter(filt)
        if not fts_ok:
            logger.warning(f"FTS delete failed for page {page_id}")
        if not vdb_ok:
            logger.warning(f"VDB delete failed for page {page_id}")
        return fts_ok and vdb_ok

    async def get_chunks_by_filter(
        self,
        page_id: str | None = None,
        block_index: int | None = None,
        limit: int = 1000,
        offset: int = 0,
        in_work_versions: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Get chunks by filter with pagination.

        Split composites store the full ``text`` on every sub-chunk under the
        same ``chunk_id``, so entries are de-duplicated by ``chunk_id``.

        ``in_work_versions`` maps page_id -> version currently being indexed;
        chunks of such "in-work" versions are excluded so a page is always
        served from a single complete version.
        """
        scroll_limit = max(offset + limit, self._scroll_limit)
        conditions = []
        if page_id is not None:
            conditions.append({"key": "page_id", "match": {"value": page_id}})
        if block_index is not None:
            conditions.append({"key": "block_index", "match": {"value": block_index}})
        if not conditions:
            return []

        filter_cond = {"must": conditions}
        logger.debug(f"page={page_id} scroll limit={scroll_limit}")
        results, _ = await self.vector_db.scroll_points(
            filter_cond, limit=scroll_limit
        )

        enriched = []
        for r in results:
            payload = r.get("payload", {})
            if in_work_versions:
                pid = payload.get("page_id")
                ver = payload.get("page_version")
                if ver is not None and pid is not None:
                    in_work = in_work_versions.get(pid)
                    if in_work is not None and int(ver) == int(in_work):
                        continue
            enriched.append(make_payload(payload, r.get("id")).model_dump())

        enriched = dedupe_chunks_by_id(enriched)
        enriched.sort(key=lambda x: (x.get("chunk_index", 0), x.get("block_part_id", 0)))
        return enriched[offset : offset + limit]

    async def close(self) -> None:
        await self.vector_db.close()
        await self.embedding.close()
