"""Tantivy-based FTS index — native Okapi BM25 scoring engine, embedded, async wrapper."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
from collections.abc import Callable
from typing import Any

from atlassian_rag_mcp.exceptions import BaseError, FtsError
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.storage.fts.base import FTSIndex
from atlassian_rag_mcp.storage.fts.normalize import PATTERN_REGEX, normalize_index_text
from atlassian_rag_mcp.utils.text import html_to_text


def _proc_starttime() -> int:
    """This process's kernel starttime (field 22 of /proc/<pid>/stat)."""
    try:
        with open(f"/proc/{os.getpid()}/stat") as f:
            data = f.read()
        return int(data.split(") ")[1].split()[19])
    except (OSError, ValueError, IndexError):
        return 0


def _lock_holder_alive(lock_path: str) -> bool:
    """Return True if a lock file is held by a live process.

    Native tantivy lock files are empty; our ``_ensure_writer`` writes a
    ``"pid starttime"`` marker after acquiring a writer. A lock without a
    marker, or whose holder is dead (crash/shutdown), is stale.
    """
    try:
        with open(lock_path) as f:
            marker = f.read().strip()
    except OSError:
        return False
    parts = marker.split()
    if len(parts) != 2:
        return False
    try:
        pid, start = int(parts[0]), int(parts[1])
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
        if int(data.split(") ")[1].split()[19]) != start:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


class TantivyClient(FTSIndex):
    """Tantivy full-text search (FTS) index — FTS scoring, embedded storage.

    Schema v7: the document payload is stored as a single UTF-8 JSON blob —
    the exact same canonical chunk payload that lands in the VectorDB point —
    so FTS and vector results are byte-for-byte identical for the same chunk.
    Term fields (``chunk_id``, ``page``, ``space``) exist only for
    delete-by-term / query resolution (``page`` holds the payload's
    ``page_id`` value, ``space`` the payload's ``space_key`` value).
    ``text`` is the FTS field over the cleaned (HTML-stripped,
    Russian-stemmed) chunk text; ``title`` is the same normalization over
    the page title (used with a field boost). ``pattern`` is fed the raw
    cleaned text+title and indexed with a
    custom regex tokenizer (``PATTERNS`` in storage/fts/normalize) that emits
    only the whitelisted identifier families (case codes, Jira IDs, short
    codes, release versions) as single terms — so the exact code ``ABC.DEF.01``
    is distinct from related codes (``C_ABC.DEF.01``, ``ABC.DEF_GHI.01``) and
    noise (``OBJECT_IDENTIFIER``, dates, base64) is never indexed.
    """

    _SCHEMA_VERSION = "7"

    def __init__(self, index_dir: str) -> None:
        self._index_dir = index_dir
        self._index_path = os.path.join(index_dir, "tantivy")
        self._lock = threading.Lock()
        self._index: Any | None = None
        self._writer: Any | None = None

    def _build_schema(self, tantivy: Any) -> Any:
        """Build the Tantivy schema for the FTS index.

        The ``pattern`` field uses the custom ``pattern_tok`` regex
        tokenizer (PATTERNS from storage/fts/normalize): it emits only
        whitelisted identifiers, each as a single term, so a query for the
        exact code cannot be diluted by related codes and non-code noise is
        never indexed. Tokenizers are registered on the index in start()/
        reset() — they are in-memory per index object.
        """
        sb = tantivy.SchemaBuilder()
        sb.add_text_field("text", index_option="position")
        sb.add_text_field("title", index_option="position")
        sb.add_text_field("pattern", index_option="position", tokenizer_name="pattern_tok")
        sb.add_text_field(
            "chunk_id", stored=True,
            index_option="position", tokenizer_name="raw",
        )
        sb.add_text_field(
            "page", index_option="position", tokenizer_name="raw",
        )
        sb.add_text_field(
            "space", index_option="position", tokenizer_name="raw",
        )
        sb.add_bytes_field("payload", stored=True, indexed=False, fast=False)
        return sb.build()

    @staticmethod
    def _register_pattern_tokenizer(tantivy: Any, index: Any) -> None:
        """Register the ``pattern_tok`` regex tokenizer on the index.

        Tokenizer registrations live in the in-memory index object (not on
        disk), so they must be re-registered after every create/open.
        """
        index.register_tokenizer(
            "pattern_tok",
            tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.regex(PATTERN_REGEX)).build(),
        )

    @staticmethod
    def _read_marker(marker_path: str) -> str:
        try:
            with open(marker_path) as f:
                return f.read().strip()
        except OSError:
            return ""

    @staticmethod
    def _write_marker(marker_path: str, version: str) -> None:
        try:
            with open(marker_path, "w") as f:
                f.write(version)
        except OSError:
            pass

    async def start(self) -> None:
        """Lazy init: build schema and open/create the tantivy index.

        If an existing index was created with a different schema version
        (tracked via a ``schema_version`` marker next to the index dir), it is
        wiped and recreated — tantivy schemas are immutable. A persisted
        index dir that cannot be opened (partial write, killed process) is
        considered corrupt — it is wiped and recreated empty (rebuild via
        `sync` with `fts_rebuild: true`).
        """

        def _work() -> None:
            with self._lock:
                if self._index is not None:
                    return
                import tantivy

                os.makedirs(self._index_dir, exist_ok=True)
                schema = self._build_schema(tantivy)
                index_path = os.path.join(self._index_dir, "tantivy")
                os.makedirs(index_path, exist_ok=True)
                marker_path = os.path.join(self._index_dir, "schema_version")

                for lockfile in (".tantivy-writer.lock", ".tantivy-meta.lock"):
                    lock_path = os.path.join(index_path, lockfile)
                    try:
                        if _lock_holder_alive(lock_path):
                            logger.warning(
                                f"FTS lock {lockfile} is held by another live process"
                            )
                            continue
                        os.remove(lock_path)
                    except OSError:
                        pass

                if tantivy.Index.exists(path=index_path):
                    if self._read_marker(marker_path) != self._SCHEMA_VERSION:
                        logger.warning(
                            f"FTS index schema mismatch (marker="
                            f"{self._read_marker(marker_path) or 'absent'!r}, "
                            f"current={self._SCHEMA_VERSION}) — wiping legacy index"
                        )
                        shutil.rmtree(index_path)
                        os.makedirs(index_path, exist_ok=True)
                        self._index = tantivy.Index(schema=schema, path=index_path)
                    else:
                        try:
                            self._index = tantivy.Index.open(path=index_path)
                        except (ValueError, RuntimeError) as e:
                            logger.error(f"FTS index at {index_path} is corrupt ({e}) — wiping and recreating")
                            shutil.rmtree(index_path)
                            os.makedirs(index_path, exist_ok=True)
                            self._index = tantivy.Index(schema=schema, path=index_path)
                else:
                    self._index = tantivy.Index(schema=schema, path=index_path)
                self._write_marker(marker_path, self._SCHEMA_VERSION)
                self._register_pattern_tokenizer(tantivy, self._index)

        await self._run_io(_work)

    async def reset(self) -> None:
        """Wipe the index and recreate it with the current schema.

        Used to rebuild the FTS index from scratch (e.g. after a
        normalization change) — VectorDB is untouched, only the Tantivy
        directory is dropped and re-created.
        """

        def _work() -> None:
            with self._lock:
                if self._writer is not None:
                    try:
                        self._writer.close()
                    except Exception:
                        pass
                    self._writer = None
                self._index = None
                index_path = self._index_path
                if os.path.exists(index_path):
                    shutil.rmtree(index_path)
                os.makedirs(index_path, exist_ok=True)
                import tantivy

                self._index = tantivy.Index(schema=self._build_schema(tantivy), path=index_path)
                self._write_marker(os.path.join(self._index_dir, "schema_version"), self._SCHEMA_VERSION)
                self._register_pattern_tokenizer(tantivy, self._index)
                logger.info(f"TantivyClient reset (schema {self._SCHEMA_VERSION})")

        await self._run_io(_work)

    def _ensure_writer(self) -> Any:
        if self._writer is not None:
            return self._writer
        self._writer = self._index.writer()
        for lockfile in (".tantivy-writer.lock", ".tantivy-meta.lock"):
            try:
                with open(os.path.join(self._index_path, lockfile), "w") as f:
                    f.write(f"{os.getpid()} {_proc_starttime()}")
            except OSError:
                pass
        return self._writer

    def _ensure_searcher(self) -> Any:
        self._index.reload()
        return self._index.searcher()

    @staticmethod
    def _encode_doc(payload: dict[str, Any]) -> dict[str, Any]:
        """Encode a canonical chunk payload into a tantivy document.

        The payload is stored as a single UTF-8 JSON blob — the exact document
        that lands in the VectorDB point payload — so no per-field mapping or
        null sentinels are needed. Term fields serve deletion/querying only.
        ``text``/``title`` are normalized (HTML stripped, Russian stemmed) so
        FTS scores word content instead of markup and inflected forms match.
        ``pattern`` gets the same cleaned text+title raw: the field's
        regex tokenizer (PATTERNS) keeps only whitelisted identifiers as
        exact single terms.
        """
        text = html_to_text(str(payload.get("text") or ""))
        title = html_to_text(str(payload.get("page_title") or ""))
        return {
            "text": normalize_index_text(text),
            "title": normalize_index_text(title),
            "pattern": [text, title],
            "chunk_id": str(payload.get("chunk_id") or ""),
            "page": str(payload.get("page_id") or ""),
            "space": str(payload.get("space_key") or ""),
            "payload": json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        }

    @staticmethod
    def _decode_doc(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Decode a stored tantivy doc into (chunk_id, canonical payload).

        A corrupt or missing payload blob degrades to a minimal payload with
        the chunk_id preserved, so results stay usable.
        """
        cid = raw.get("chunk_id")
        cid = cid[0] if isinstance(cid, list) and cid else ""

        blob = raw.get("payload")
        blob = blob[0] if isinstance(blob, list) and blob else None
        if blob:
            try:
                text = blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob)
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return str(cid or parsed.get("chunk_id") or ""), parsed
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.warning("FTS payload blob is not valid JSON — using empty payload")
        return str(cid), {"chunk_id": cid} if cid else {}

    async def _run_io(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def index_batch(self, docs: list[dict[str, Any]]) -> bool:
        """Index a batch of documents into Tantivy (all-or-nothing).

        ``docs`` are canonical chunk payloads — the same dicts that land in
        the VectorDB point payload (see ``build_point``).

        If ANY document fails, the pending writer buffer is rolled back and
        a RuntimeError is raised — the batch is never partially persisted.
        """
        if not docs:
            return True

        def _work() -> bool:
            import tantivy

            with self._lock:
                writer = self._ensure_writer()
                added = 0
                try:
                    for d in docs:
                        td = tantivy.Document.from_dict(self._encode_doc(d))
                        writer.add_document(td)
                        added += 1
                    writer.commit()
                except Exception as e:
                    cid = docs[added - 1].get("chunk_id", "?") if added else "?"
                    try:
                        writer.rollback()
                    except Exception:
                        pass
                    logger.error(
                        f"Tantivy index_batch failed chunk={cid}, "
                        f"rolled back {added}/{len(docs)} staged docs: {e}",
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Tantivy index_batch failed chunk={cid}: {e}"
                    ) from e
                self._index.reload()
            logger.info(f"TantivyClient indexed {added}/{len(docs)} docs")
            return True

        return await self._run_io(_work)

    async def search(self, query: str, limit: int = 25) -> list[dict[str, Any]]:
        """Full-text search (FTS) via the Tantivy index.

        Raises FtsQueryError when the query is rejected by the parser (malformed
        syntax, unknown field) — the message carries the query and the parser's
        error text, so the caller can fix the query. Index/IO failures are
        logged and reported as an empty result set.
        """
        if not query:
            return []

        def _work() -> list[dict[str, Any]]:
            with self._lock:
                searcher = self._ensure_searcher()
                if searcher.num_docs == 0:
                    return []
                try:
                    query_obj = self._index.parse_query(query, ["text"])
                except Exception as e:
                    raise FtsError(f"FTS query failed: {query} — {e}") from e
                results = searcher.search(query_obj, limit)
            out: list[dict[str, Any]] = []
            for score, doc_addr in results.hits:
                doc = searcher.doc(doc_addr)
                cid, payload = self._decode_doc(doc.to_dict())
                out.append(
                    {
                        "id": cid,
                        "score": round(float(score), 6),
                        "payload": payload,
                    }
                )
            return out

        try:
            return await self._run_io(_work)
        except FtsError:
            raise
        except Exception as e:
            logger.warning(f"Tantivy search failed: {e}")
            return []

    async def delete_by_filter(self, filter_condition: dict) -> bool:
        """Delete documents matching filter condition."""
        if not filter_condition:
            return True

        def _work() -> bool:
            with self._lock:
                writer = self._ensure_writer()

                exclude = self._collect_must_not(filter_condition)
                if exclude is not None:
                    self._delete_terms_excluding(writer, filter_condition, exclude)
                else:
                    self._delete_terms(writer, filter_condition)

                writer.commit()
                self._index.reload()
            logger.info(f"TantivyClient delete_by_filter: {filter_condition}")
            return True

        return await self._run_io(_work)

    def _collect_must_not(self, filter_cond: dict) -> set[str] | None:
        """Return the set of excluded chunk_ids from a must_not clause.

        Returns None when there is no must_not clause to honor.
        """
        must_not = filter_cond.get("must_not")
        if not must_not:
            return None
        excluded: set[str] = set()
        for cond in must_not:
            key = cond.get("key", "")
            match = cond.get("match", {})
            if key != "chunk_id":
                continue
            if "any" in match:
                excluded.update(str(v) for v in match["any"])
            elif "value" in match:
                excluded.add(str(match["value"]))
        return excluded

    def _delete_terms_excluding(
        self, writer, filter_cond: dict, excluded: set[str]
    ) -> None:
        """Delete the page's docs whose chunk_id is NOT in `excluded`."""
        page_ids: set[str] = set()
        for cond in filter_cond.get("must", []):
            key = cond.get("key", "")
            match = cond.get("match", {})
            if key == "page_id":
                if "value" in match:
                    page_ids.add(str(match["value"]))
                elif "any" in match:
                    page_ids.update(str(v) for v in match["any"])
        if not page_ids:
            return

        searcher = self._ensure_searcher()
        all_chunk_ids: set[str] = set()
        for page_id in page_ids:
            query = self._index.parse_query(f'page:"{page_id}"', ["text"])
            results = searcher.search(query, 100_000)
            for _score, doc_addr in results.hits:
                doc = searcher.doc(doc_addr)
                raw = doc.to_dict()
                cid = raw.get("chunk_id")
                cid = (cid[0] if isinstance(cid, list) and cid else cid)
                if cid:
                    all_chunk_ids.add(str(cid))

        stale = all_chunk_ids - excluded
        for cid in stale:
            try:
                writer.delete_documents_by_term("chunk_id", cid)
            except Exception as e:
                logger.warning(f"Tantivy exclude-delete chunk_id={cid}: {e}")

    @staticmethod
    def _fts_field(key: str) -> str:
        """Map a shared (VectorDB-compatible) filter key to the FTS index field.

        Filters use the payload keys ``page_id`` / ``space_key`` (matching
        the VectorDB payload); the Tantivy index fields are ``page`` /
        ``space``.
        """
        return {"page_id": "page", "space_key": "space"}.get(key, key)

    def _delete_terms(self, writer: Any, filter_cond: dict) -> None:
        targets: list[tuple[str, Any]] = []

        for tag in ("must", "should"):
            conds = filter_cond.get(tag)
            if conds is None:
                continue
            for cond in conds:
                key = self._fts_field(cond.get("key", ""))
                match = cond.get("match", {})
                if "value" in match:
                    targets.append((key, match["value"]))
                elif "any" in match:
                    for v in match["any"]:
                        targets.append((key, v))

        if not targets:
            fm = filter_cond.get("match")
            if isinstance(fm, dict):
                key = self._fts_field(filter_cond.get("key", ""))
                if "value" in fm:
                    targets.append((key, fm["value"]))
                elif "any" in fm:
                    for v in fm["any"]:
                        targets.append((key, v))

        for field_name, value in targets:
            try:
                writer.delete_documents_by_term(field_name, str(value))
            except Exception as e:
                logger.warning(
                    f"Tantivy delete [{field_name}={value}]: {e}"
                )

    async def clear(self) -> bool:
        target = os.path.join(self._index_dir, "tantivy")

        def _work() -> bool:
            with self._lock:
                if os.path.exists(target):
                    shutil.rmtree(target)
                self._index = None
                self._writer = None
                logger.info("TantivyClient cleared")
                return True

        return await self._run_io(_work)

    async def validate(self) -> None:
        def _work() -> None:
            with self._lock:
                self._ensure_writer()
                searcher = self._ensure_searcher()
                num_docs = searcher.num_docs
                logger.info(f"TantivyClient OK — documents: {num_docs}")

        try:
            await self._run_io(_work)
        except Exception as e:
            if "LockBusy" in str(e):
                raise BaseError(
                    f"Tantivy FTS index is locked by another process "
                    f"(index dir: {self._index_dir}) — stop the other server instance"
                ) from e
            raise BaseError(f"Tantivy FTS index validation failed: {e}") from e

    async def close(self) -> None:
        def _work() -> None:
            with self._lock:
                if self._writer is not None:
                    try:
                        self._writer.commit()
                    except Exception as e:
                        logger.warning(
                            f"Tantivy commit during close failed: {e}",
                            exc_info=True,
                        )
                self._writer = None
                for lockfile in (".tantivy-writer.lock", ".tantivy-meta.lock"):
                    try:
                        os.remove(os.path.join(self._index_path, lockfile))
                    except OSError:
                        pass

        await self._run_io(_work)
