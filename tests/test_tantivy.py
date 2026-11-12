"""Smoke test for TantivyClient: index, search, delete, RRF."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

import pytest

from atlassian_rag_mcp.exceptions import FtsError


@pytest.fixture
def index_dir():
    tmp_dir = tempfile.mkdtemp()
    idx_dir = os.path.join(tmp_dir, "fts_test")
    yield idx_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


async def _create_index(index_dir: str):
    from atlassian_rag_mcp.storage.fts.tantivy_client import TantivyClient
    idx = TantivyClient(index_dir=index_dir)
    await idx.start()
    return idx


def _sample_docs() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "text": "Configuration and deployment pipeline for production systems",
            "page_id": "p1",
            "page_title": "Deployment Guide",
            "page_version": 3,
            "chunk_type": "section",
            "chunk_index": 0,
            "heading_hierarchy": ["Deployment", "Config"],
            "block_index": -1,
            "block_part_id": -1,
        },
        {
            "chunk_id": "c2",
            "text": "Debugging strategies for microservices architecture",
            "page_id": "p1",
            "page_title": "Deployment Guide",
            "page_version": 3,
            "chunk_type": "section",
            "chunk_index": 1,
            "heading_hierarchy": ["Deployment", "Debugging"],
            "block_index": -1,
            "block_part_id": -1,
        },
        {
            "chunk_id": "c3",
            "text": "Monitoring dashboards with Prometheus and Grafana stack",
            "page_id": "p2",
            "page_title": "Monitoring Setup",
            "page_version": 1,
            "chunk_type": "section",
            "chunk_index": 0,
            "heading_hierarchy": ["Monitoring", "Dashboards"],
            "block_index": -1,
            "block_part_id": -1,
        },
        {
            "chunk_id": "c4",
            "text": "Python configuration settings and environment variables",
            "page_id": "p2",
            "page_title": "Monitoring Setup",
            "page_version": 1,
            "chunk_type": "section",
            "chunk_index": 1,
            "heading_hierarchy": ["Monitoring", "Config"],
            "block_index": -1,
            "block_part_id": -1,
        },
        {
            "chunk_id": "c5",
            "text": "Testing microservices with integration test frameworks",
            "page_id": "p3",
            "page_title": "Testing Guide",
            "page_version": 2,
            "chunk_type": "section",
            "chunk_index": 0,
            "heading_hierarchy": ["Testing"],
            "block_index": -1,
            "block_part_id": -1,
        },
    ]


async def test_index_batch_succeeds(index_dir):
    idx = await _create_index(index_dir)
    docs = _sample_docs()
    result = await idx.index_batch(docs)
    assert result is True
    await idx.close()


async def test_index_batch_return_true_on_empty(index_dir):
    idx = await _create_index(index_dir)
    result = await idx.index_batch([])
    assert result is True
    await idx.close()


async def test_search_returns_results(index_dir):
    idx = await _create_index(index_dir)
    await idx.index_batch(_sample_docs())
    results = await idx.search("deployment")
    assert len(results) >= 1
    assert results[0]["id"] == "c1"
    await idx.close()


async def test_search_better_match_first(index_dir):
    idx = await _create_index(index_dir)
    await idx.index_batch(_sample_docs())
    results = await idx.search("prometheus grafana monitoring")
    assert results[0]["id"] == "c3"
    await idx.close()


async def test_exact_code_outranks_related_codes(index_dir):
    """An exact code (ABC.DEF.01) must outrank a dense changelog row that lists
    many RELATED codes (C_ABC.DEF.01, C_ABC.DEF_GHI.01, ...) — the
    pattern field makes the exact identifier a unique, boosted term."""
    from atlassian_rag_mcp.confluence.fts_builder import FtsSearchParams, build_fts_query

    idx = await _create_index(index_dir)
    docs = [
        {
            "chunk_id": "c-changelog",
            "text": (
                "Отключены C_ABC.DEF.01. Добавлены C_ABC.DEF_GHI.01 "
                "C_ABC.DEF_GHI.01 C_ABC.DEF_JKL.01 C_ABC.DEF_MNO.01 C_ABC.DEF_PQR.01. "
                "Изменены C_ABC.DEF.01 и C_ABC.DEF_MNO.02"
            ),
            "page_id": "p1",
            "page_title": "Changelog",
            "page_version": 1,
            "chunk_type": "table_row",
            "chunk_index": 0,
            "heading_hierarchy": [],
            "block_index": 1,
            "block_part_id": 1,
        },
        {
            "chunk_id": "c-exact",
            "text": "10 ABC.DEF.01 УТОЧНИТЬ Превышение установленных лимитов",
            "page_id": "p2",
            "page_title": "Счета для модуля",
            "page_version": 1,
            "chunk_type": "table_row",
            "chunk_index": 0,
            "heading_hierarchy": ["Счета для модуля"],
            "block_index": 1,
            "block_part_id": 11,
        },
    ]
    await idx.index_batch(docs)
    query = build_fts_query(FtsSearchParams(fts="ABC.DEF.01"))
    results = await idx.search(query, limit=10)
    assert len(results) >= 2
    assert results[0]["id"] == "c-exact"
    await idx.close()


async def test_exact_code_not_matched_by_related_code_query(index_dir):
    """Querying the exact code ABC.DEF.01 must not be satisfied by a doc that
    only contains the related code C_ABC.DEF.01 (different pattern
    terms)."""
    from atlassian_rag_mcp.confluence.fts_builder import FtsSearchParams, build_fts_query
    from atlassian_rag_mcp.storage.fts.normalize import extract_pattern_tokens

    idx = await _create_index(index_dir)
    docs = [
        {
            "chunk_id": "c-variant",
            "text": "Добавил проверки и изменил C_ABC.DEF.01 и C_ABC.DEF_GHI.01",
            "page_id": "p1",
            "page_title": "Changelog",
            "page_version": 1,
            "chunk_type": "table_row",
            "chunk_index": 0,
            "heading_hierarchy": [],
            "block_index": 1,
            "block_part_id": 1,
        },
    ]
    await idx.index_batch(docs)
    assert "ABC.DEF.01" not in extract_pattern_tokens(
        "Добавил проверки и изменил C_ABC.DEF.01 и C_ABC.DEF_GHI.01"
    )
    query = build_fts_query(FtsSearchParams(fts="ABC.DEF.01"))
    assert 'pattern:("ABC.DEF.01"^3)' in query
    results = await idx.search('pattern:"ABC.DEF.01"', limit=10)
    assert results == []
    await idx.close()


async def test_noise_not_indexed_in_pattern(index_dir):
    """Non-whitelisted identifiers (OBJECT_IDENTIFIER, APP_VERSION) and dates
    (23.01.2023) produce no pattern terms — an exact-term query finds
    nothing even though the words are present in the document text."""
    idx = await _create_index(index_dir)
    docs = [
        {
            "chunk_id": "c-noise",
            "text": "Макрос содержит OBJECT_IDENTIFIER и APP_VERSION, дата 23.01.2023",
            "page_id": "p1",
            "page_title": "Macro payload",
            "page_version": 1,
            "chunk_type": "table_row",
            "chunk_index": 0,
            "heading_hierarchy": [],
            "block_index": 1,
            "block_part_id": 1,
        },
    ]
    await idx.index_batch(docs)
    for term in ("OBJECT_IDENTIFIER", "APP_VERSION", "23.01.2023"):
        results = await idx.search(f'pattern:"{term}"', limit=10)
        assert results == [], f"{term} must not be indexed in pattern"
    await idx.close()


async def test_boosted_code_query_matches_code_doc(index_dir):
    """Per-term boost syntax: build_fts_query emits pattern:(CODE)^N (unquoted
    code, boost outside the parens) — it must parse against the live index and
    match only the doc that carries the exact code."""
    from atlassian_rag_mcp.confluence.fts_builder import FtsSearchParams, build_fts_query

    idx = await _create_index(index_dir)
    await idx.index_batch([
        {
            "chunk_id": "c-code",
            "text": "Справочник ABC.DEF.01 по кодированию записей",
            "page_id": "p1",
            "page_title": "Справочники",
            "page_version": 1,
            "chunk_type": "section",
            "chunk_index": 0,
            "heading_hierarchy": [],
            "block_index": 1,
            "block_part_id": 1,
        },
        {
            "chunk_id": "c-other",
            "text": "Configuration and deployment pipeline",
            "page_id": "p2",
            "page_title": "Deployment",
            "page_version": 1,
            "chunk_type": "section",
            "chunk_index": 0,
            "heading_hierarchy": [],
            "block_index": 1,
            "block_part_id": 1,
        },
    ])
    query = build_fts_query(FtsSearchParams(fts="ABC.DEF.01^4"))
    assert query == "(pattern:(ABC.DEF.01)^4)"
    results = await idx.search(query, limit=10)
    assert [r["id"] for r in results] == ["c-code"]

    query2 = build_fts_query(FtsSearchParams(fts="Справочники^2"))
    assert query2 == "((title:(справочник))^2)"
    results2 = await idx.search(query2, limit=10)
    assert [r["id"] for r in results2] == ["c-code"]
    await idx.close()


async def test_search_empty_query(index_dir):
    idx = await _create_index(index_dir)
    await idx.index_batch(_sample_docs())
    results = await idx.search("")
    assert results == []
    await idx.close()


async def test_search_no_index_yet(index_dir):
    idx = await _create_index(index_dir)
    results = await idx.search("deployment")
    assert results == []
    await idx.close()


async def test_search_limit(index_dir):
    idx = await _create_index(index_dir)
    await idx.index_batch(_sample_docs())
    results = await idx.search("configuration", limit=2)
    assert len(results) <= 2
    await idx.close()


async def test_search_payload_fields_present(index_dir):
    idx = await _create_index(index_dir)
    await idx.index_batch(_sample_docs())
    results = await idx.search("pipeline")
    assert len(results) >= 1
    payload = results[0].get("payload", {})
    assert payload.get("page_id") == "p1"
    assert "text" in payload
    await idx.close()


async def test_roundtrip_canonical_payload_exact(index_dir):
    """The stored FTS payload is the exact canonical chunk payload — the same
    dict that lands in the VectorDB point payload (byte-for-byte roundtrip)."""
    from atlassian_rag_mcp.confluence.models import ChunkPayload
    payload = ChunkPayload(
        chunk_id="c-1234",
        chunk_type="section",
        text="unique token aardvark configuration pipeline",
        page_id="p1",
        page_title="Deployment Guide",
        page_version=7,
        parent_id="p0",
        block_index=4,
        block_part_id=None,
        chunk_index=2,
        updated_at="2026-09-24T10:00:00",
        heading_hierarchy=["Общая информация", "3.3. Технические критерии"],
        space_key="PROJ",
    ).model_dump()
    idx = await _create_index(index_dir)
    await idx.index_batch([payload])
    results = await idx.search("aardvark")
    assert len(results) == 1
    assert results[0]["id"] == "c-1234"
    assert results[0]["payload"] == payload
    await idx.close()


async def test_none_fields_roundtrip(index_dir):
    """None-valued fields (block_part_id, parent_id, updated_at) roundtrip as
    None — matching Qdrant's null, not a storage sentinel."""
    idx = await _create_index(index_dir)
    doc = {
        "chunk_id": "n1",
        "text": "unique token quokka whole block chunk",
        "page_id": "p9",
        "page_title": "T", "page_version": 1,
        "chunk_type": "section", "chunk_index": 0,
        "heading_hierarchy": [],
        "block_index": 2, "block_part_id": None,
        "parent_id": None, "updated_at": None, "space_key": "PROJ",
    }
    await idx.index_batch([doc])
    results = await idx.search("quokka")
    assert len(results) == 1
    assert results[0]["payload"] == doc
    await idx.close()


def test_decode_doc_fallbacks():
    """Corrupt/missing payload blobs degrade to a minimal payload with the
    chunk_id preserved; legacy string blobs still parse."""
    from atlassian_rag_mcp.storage.fts.tantivy_client import TantivyClient
    decode = TantivyClient._decode_doc
    cid, payload = decode({"chunk_id": ["c1"], "payload": [b"not-json{{"]})
    assert cid == "c1"
    assert payload == {"chunk_id": "c1"}
    cid2, payload2 = decode({"chunk_id": ["c2"], "payload": []})
    assert cid2 == "c2"
    assert payload2 == {"chunk_id": "c2"}
    cid3, payload3 = decode({
        "chunk_id": ["c3"],
        "payload": ['{"chunk_id": "c3", "text": "x", "heading_hierarchy": ["a"]}'],
    })
    assert cid3 == "c3"
    assert payload3 == {"chunk_id": "c3", "text": "x", "heading_hierarchy": ["a"]}


async def test_schema_mismatch_wipes_legacy_index(index_dir):
    """A pre-existing index with an unknown schema (no schema_version marker)
    is wiped and recreated with the current schema on start()."""
    import subprocess
    import sys
    idx_path = os.path.join(index_dir, "tantivy")
    os.makedirs(idx_path, exist_ok=True)
    code = (
        "import tantivy; "
        "sb = tantivy.SchemaBuilder(); "
        "sb.add_text_field('x', index_option='position'); "
        f"tantivy.Index(schema=sb.build(), path={idx_path!r})"
    )
    subprocess.run([sys.executable, "-c", code], check=True)

    idx = await _create_index(index_dir)
    await idx.index_batch([
        {
            "chunk_id": "m1",
            "text": "unique token meerkat legacy schema wipe",
            "page_id": "p7",
            "page_title": "T", "page_version": 1,
            "chunk_type": "section", "chunk_index": 0,
            "heading_hierarchy": [], "block_index": None, "block_part_id": None,
            "space_key": "PROJ",
        },
    ])
    results = await idx.search("meerkat")
    assert len(results) == 1
    assert results[0]["payload"]["chunk_id"] == "m1"
    await idx.close()


async def test_search_fts_scores_descending(index_dir):
    idx = await _create_index(index_dir)
    await idx.index_batch(_sample_docs())
    results = await idx.search("architecture")
    scores = [r["score"] for r in results]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    await idx.close()


async def test_delete_by_page_id(index_dir):
    idx = await _create_index(index_dir)
    await idx.index_batch(_sample_docs())
    filter_cond = {
        "must": [{"key": "page_id", "match": {"value": "p1"}}]
    }
    await idx.delete_by_filter(filter_cond)
    results_after = await idx.search("architecture")
    ids_after = {r["id"] for r in results_after}
    assert "c1" not in ids_after
    assert "c2" not in ids_after
    await idx.close()


async def test_delete_empty_filter(index_dir):
    idx = await _create_index(index_dir)
    result = await idx.delete_by_filter({})
    assert result is True
    await idx.close()


async def test_delete_none_filter(index_dir):
    idx = await _create_index(index_dir)
    result = await idx.delete_by_filter(None)
    assert result is True
    await idx.close()


async def test_delete_by_must_not_keeps_excluded_chunks(index_dir):
    """Production chunks use hyphenated UUID ids; Tantivy's default tokenizer
    would split them on "-", so exact-term deletion of a stale id must still
    work — this exercises `_delete_terms_excluding` with hyphenated ids."""
    idx = await _create_index(index_dir)
    docs = [
        {
            "chunk_id": "aaaaaaaa-1111-2222-3333-000000000001",
            "text": "alpha configuration for page one",
            "page_id": "111",
            "page_title": "P1", "page_version": 2,
            "chunk_type": "section", "chunk_index": 0,
            "heading_hierarchy": [], "block_index": -1, "block_part_id": -1,
        },
        {
            "chunk_id": "bbbbbbbb-1111-2222-3333-000000000002",
            "text": "stale beta chunk that must be removed",
            "page_id": "111",
            "page_title": "P1", "page_version": 2,
            "chunk_type": "section", "chunk_index": 1,
            "heading_hierarchy": [], "block_index": -1, "block_part_id": -1,
        },
        {
            "chunk_id": "cccccccc-1111-2222-3333-000000000003",
            "text": "gamma config on another page untouched",
            "page_id": "222",
            "page_title": "P2", "page_version": 1,
            "chunk_type": "section", "chunk_index": 0,
            "heading_hierarchy": [], "block_index": -1, "block_part_id": -1,
        },
    ]
    await idx.index_batch(docs)

    new_ids = {docs[0]["chunk_id"]}
    filter_cond = {
        "must": [{"key": "page_id", "match": {"value": "111"}}],
        "must_not": [{"key": "chunk_id", "match": {"any": list(new_ids)}}],
    }
    await idx.delete_by_filter(filter_cond)

    assert docs[0]["chunk_id"] in {r["id"] for r in await idx.search("alpha", 10)}
    assert docs[1]["chunk_id"] not in {r["id"] for r in await idx.search("beta", 10)}, \
        "stale chunk should be deleted"
    assert docs[2]["chunk_id"] in {r["id"] for r in await idx.search("gamma", 10)}, \
        "other page's chunk must remain"
    await idx.close()


async def test_clear(index_dir):
    idx = await _create_index(index_dir)
    await idx.index_batch(_sample_docs())
    count_before = len(await idx.search("debugging"))
    assert count_before > 0
    await idx.clear()
    count_after = len(await idx.search("debugging"))
    assert count_after == 0
    await idx.close()


async def test_close_and_reopen(index_dir):
    idx1 = await _create_index(index_dir)
    await idx1.index_batch(_sample_docs())
    await idx1.close()
    idx2 = await _create_index(index_dir)
    results = await idx2.search("deployment")
    assert len(results) >= 1
    await idx2.close()


async def test_concurrent_search_and_index_batch(index_dir):
    idx = await _create_index(index_dir)
    await idx.index_batch(_sample_docs())

    async def search_task():
        for _ in range(5):
            await idx.search("deployment")

    tasks = [search_task() for _ in range(8)]
    batch_docs = _sample_docs() * 2
    batch_task = idx.index_batch(batch_docs)
    results = await asyncio.gather(*tasks, batch_task, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"Errors in concurrent search+index: {errors}"
    final = await idx.search("deployment")
    assert len(final) >= 1
    await idx.close()


async def test_concurrent_search_and_clear(index_dir):
    idx = await _create_index(index_dir)
    await idx.index_batch(_sample_docs())

    async def search_task():
        for _ in range(3):
            await idx.search("deployment")

    tasks = [search_task() for _ in range(6)]
    clear_task = idx.clear()
    results = await asyncio.gather(*tasks, clear_task, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"Errors in concurrent search+clear: {errors}"

    await idx.start()
    await idx.index_batch(_sample_docs())
    final = await idx.search("deployment")
    assert len(final) >= 1
    await idx.close()


class TestMalformedQuery:
    async def test_syntax_error_raises_fts_query_error(self, index_dir):
        idx = await _create_index(index_dir)
        await idx.index_batch(_sample_docs())
        with pytest.raises(FtsError, match="FTS query failed: text:\\(\\(\\("):
            await idx.search("text:(((")
        await idx.close()

    async def test_unknown_field_raises_fts_query_error(self, index_dir):
        idx = await _create_index(index_dir)
        await idx.index_batch(_sample_docs())
        with pytest.raises(FtsError, match="Field does not exist"):
            await idx.search("nosuchfield:foo")
        await idx.close()

    async def test_valid_search_still_works_after_rejection(self, index_dir):
        idx = await _create_index(index_dir)
        await idx.index_batch(_sample_docs())
        with pytest.raises(FtsError):
            await idx.search("text:(((")
        results = await idx.search("deployment")
        assert len(results) >= 1
        await idx.close()


async def test_concurrent_search_and_delete(index_dir):
    idx = await _create_index(index_dir)
    await idx.index_batch(_sample_docs())

    async def search_task():
        for _ in range(5):
            await idx.search("debugging")

    filter_cond = {"must": [{"key": "page_id", "match": {"value": "p1"}}]}
    tasks = [search_task() for _ in range(6)]
    delete_task = idx.delete_by_filter(filter_cond)
    results = await asyncio.gather(*tasks, delete_task, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"Errors in concurrent search+delete: {errors}"

    remaining = await idx.search("monitoring")
    assert len(remaining) > 0
    await idx.close()
