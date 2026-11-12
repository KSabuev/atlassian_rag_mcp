"""enrich_payload_results + RRF formula + response-size guard — isolated unit tests."""

from __future__ import annotations

import json

from atlassian_rag_mcp.confluence.models import _ensure_list, make_payload
from atlassian_rag_mcp.confluence.search.base import (
    OMITTED_TEXT_PREFIX,
    RRF_K,
    _payload_to_chunk_dict,
    _response_bytes,
    apply_response_size_guard,
    build_page_url,
    enrich_payload_results,
)


class TestRRFFormula:
    """The RRF formula is: 1 / (RRF_K + rank), summed across lists."""

    def test_single_result_rank_zero(self):
        score = 1.0 / (RRF_K + 0)
        assert round(score, 4) == 0.025

    def test_single_result_rank_one(self):
        score = 1.0 / (RRF_K + 1)
        assert round(score, 4) == 0.0244

    def test_sum_of_two_lists(self):
        s = 1.0 / (RRF_K + 0) + 1.0 / (RRF_K + 2)
        assert round(s, 4) == 0.0488

    def test_k_equals_one_extreme(self):
        k = 1
        assert 1.0 / (k + 0) == 1.0
        assert 1.0 / (k + 1) == 0.5

    def test_large_rank_diminishes(self):
        assert 1.0 / (RRF_K + 1) > 1.0 / (RRF_K + 100)

    def test_rank_zero_vs_rank_one(self):
        r0 = 1.0 / (RRF_K + 0)
        r1 = 1.0 / (RRF_K + 1)
        assert r0 > r1




class TestFieldMapping:
    def test_base_enrichment(self):
        results = [{
            "id": "c1",
            "_combined_score": 0.8,
            "_vector_score": 0.7,
            "_fts_score": 0.6,
            "payload": {
                "chunk_id": "c1", "text": "content", "page_id": "p1",
                "page_title": "Test", "heading_hierarchy": [],
                "chunk_type": "section", "chunk_index": 0,
            },
        }]
        enriched = enrich_payload_results(results, confluence_url="https://confluence.example.com")
        entry = enriched[0]
        assert entry["score"] == 0.8
        assert entry["vector_score"] == 0.7
        assert entry["fts_score"] == 0.6
        assert entry["chunk"]["text"] == "content"
        assert entry["page_id"] == "p1"
        assert entry["page_title"] == "Test"

    def test_custom_score_field(self):
        results = [{
            "id": "c1",
            "_custom_score": 0.95,
            "_vector_score": 0.7,
            "_fts_score": 0.6,
            "payload": {
                "chunk_id": "c1", "text": "t", "page_id": "p1",
                "page_title": "T", "heading_hierarchy": [],
                "chunk_type": "section", "chunk_index": 0,
            },
        }]
        enriched = enrich_payload_results(results, score_field="_custom_score",
                                          confluence_url="https://confluence.example.com")
        assert enriched[0]["score"] == 0.95

    def test_fts_score_none_omits_field(self):
        results = [{
            "id": "c1",
            "_combined_score": 0.8,
            "_vector_score": 0.7,
            "payload": {
                "chunk_id": "c1", "text": "t", "page_id": "p1",
                "page_title": "T", "heading_hierarchy": [],
                "chunk_type": "section", "chunk_index": 0,
            },
        }]
        enriched = enrich_payload_results(results, fts_score_field=None,
                                          confluence_url="https://confluence.example.com")
        assert enriched[0]["fts_score"] == 0.0

    def test_page_url_in_chunk(self):
        results = [{
            "id": "c1",
            "_combined_score": 0.8,
            "_vector_score": 0.7,
            "_fts_score": 0.6,
            "payload": {
                "chunk_id": "c1", "text": "t", "page_id": "p1",
                "page_title": "T", "heading_hierarchy": [],
                "chunk_type": "section", "chunk_index": 0,
            },
        }]
        enriched = enrich_payload_results(results, confluence_url="https://confluence.example.com")
        assert enriched[0]["page_url"] is not None
        assert "p1" in enriched[0]["page_url"]

    def test_missing_chunk_id_uses_fallback(self):
        results = [{
            "id": "fallback-id",
            "_combined_score": 0.8,
            "_vector_score": 0.7,
            "_fts_score": 0.6,
            "payload": {
                "text": "t", "page_id": "p1",
                "page_title": "T", "heading_hierarchy": [],
                "chunk_type": "section", "chunk_index": 0,
            },
        }]
        enriched = enrich_payload_results(results, confluence_url="https://confluence.example.com")
        assert enriched[0]["chunk"]["chunk_id"] == "fallback-id"

    def test_sort_by_score_descending(self):
        results = [
            {"id": "c1", "_combined_score": 0.3, "_vector_score": 0.3, "_fts_score": 0.0,
             "payload": {"chunk_id": "c1", "text": "t", "page_id": "p1", "page_title": "T",
                         "heading_hierarchy": [], "chunk_type": "section", "chunk_index": 0}},
            {"id": "c2", "_combined_score": 0.9, "_vector_score": 0.9, "_fts_score": 0.0,
             "payload": {"chunk_id": "c2", "text": "t", "page_id": "p1", "page_title": "T",
                         "heading_hierarchy": [], "chunk_type": "section", "chunk_index": 1}},
            {"id": "c3", "_combined_score": 0.6, "_vector_score": 0.6, "_fts_score": 0.0,
             "payload": {"chunk_id": "c3", "text": "t", "page_id": "p1", "page_title": "T",
                         "heading_hierarchy": [], "chunk_type": "section", "chunk_index": 2}},
        ]
        enriched = enrich_payload_results(results, confluence_url="https://confluence.example.com")
        scores = [e["score"] for e in enriched]
        assert scores == [0.9, 0.6, 0.3]

    def test_empty_results(self):
        enriched = enrich_payload_results([])
        assert enriched == []




class TestBuildPageUrl:
    def test_with_page_id(self):
        url = build_page_url("123", "https://c.example.com")
        assert url is not None
        assert "123" in url
        assert "c.example.com" in url

    def test_none_page_id(self):
        assert build_page_url(None, "https://c.example.com") is None

    def test_empty_base_url(self):
        assert build_page_url("123", "") is None


class TestMakePayload:
    def test_with_chunk_id(self):
        raw = {"chunk_id": "c1", "text": "t", "page_id": "p1", "page_title": "P",
               "heading_hierarchy": [], "chunk_type": "section", "chunk_index": 0}
        p = make_payload(raw)
        assert p.chunk_id == "c1"

    def test_fallback_id(self):
        raw = {"text": "t", "page_id": "p1", "page_title": "P",
               "heading_hierarchy": [], "chunk_index": 0}
        p = make_payload(raw, fallback_id="fallback")
        assert p.chunk_id == "fallback"

    def test_default_chunk_type(self):
        raw = {"chunk_id": "c1", "text": "t", "page_id": "p1", "page_title": "P",
               "heading_hierarchy": []}
        p = make_payload(raw)
        assert p.chunk_type == "section"

    def test_heading_hierarchy_as_json_string(self):
        raw = {"chunk_id": "c1", "text": "t", "page_id": "p1", "page_title": "P",
               "heading_hierarchy": '["A","B"]', "chunk_type": "section", "chunk_index": 0}
        p = make_payload(raw)
        assert p.heading_hierarchy == ["A", "B"]


class TestEnsureList:
    def test_passthrough_list(self):
        assert _ensure_list([1, 2, 3]) == [1, 2, 3]

    def test_json_string(self):
        assert _ensure_list('["a","b"]') == ["a", "b"]

    def test_non_json_string(self):
        assert _ensure_list("hello") == ["hello"]

    def test_empty_string(self):
        assert _ensure_list("") == []

    def test_none(self):
        assert _ensure_list(None) == []




class TestFullChunkText:
    def test_long_text_not_truncated(self):
        long_text = "слово " * 2000
        payload = make_payload({
            "chunk_id": "c1", "text": long_text, "page_id": "p1", "page_title": "P",
            "heading_hierarchy": [], "chunk_type": "section", "chunk_index": 0,
        })
        d = _payload_to_chunk_dict(payload)
        assert d["text"] == long_text
        assert "[truncated]" not in d["text"]

    def test_short_text_unchanged(self):
        payload = make_payload({
            "chunk_id": "c1", "text": "short", "page_id": "p1", "page_title": "P",
            "heading_hierarchy": [], "chunk_type": "section", "chunk_index": 0,
        })
        assert _payload_to_chunk_dict(payload)["text"] == "short"




def _omitted_call(pid: str, idx: int) -> str:
    return (
        f"{OMITTED_TEXT_PREFIX}confluence_get_page_content(what='chunks', "
        f"page_id='{pid}', chunk_index={idx})"
    )


def _is_omitted(c: dict) -> bool:
    return c["text"].startswith(OMITTED_TEXT_PREFIX)


def _result_with_chunks():
    def chunk(cid, idx, text):
        return {"chunk_id": cid, "text": text, "chunk_type": "section",
                "chunk_index": idx, "heading_hierarchy": []}

    big = "данные " * 800
    return {
        "sources": [
            {"page_id": "p1", "chunks": [chunk("c1", 0, big), chunk("c2", 1, big)]},
            {"page_id": "p2", "chunks": [chunk("c3", 0, big)]},
        ],
    }


class TestResponseBytes:
    def test_accounts_for_doubled_transport(self):
        """FastMCP sends the payload twice: pretty-printed in content[0].text
        and compact in structuredContent — the measure covers both copies."""
        r = _result_with_chunks()
        pretty = json.dumps(r, ensure_ascii=False, indent=2)
        compact = json.dumps(r, ensure_ascii=False)
        expected = len(json.dumps(pretty).encode("utf-8")) + len(compact.encode("utf-8")) + 65536
        assert _response_bytes(r) == expected
        assert _response_bytes(r) > len(compact.encode("utf-8")) * 2

    def test_grows_with_quotes(self):
        """HTML-like text (many quotes) inflates the escaped copy; the measure
        must track it, otherwise the guard underestimates the SSE body."""
        plain = {"sources": [{"page_id": "p", "chunks": [{"chunk_id": "c", "text": "abc" * 1000,
                                                           "chunk_type": "s", "chunk_index": 0,
                                                           "heading_hierarchy": []}]}]}
        quoted = {"sources": [{"page_id": "p", "chunks": [{"chunk_id": "c", "text": '"a" ' * 1000,
                                                           "chunk_type": "s", "chunk_index": 0,
                                                           "heading_hierarchy": []}]}]}
        assert _response_bytes(quoted) > _response_bytes(plain)


class TestResponseSizeGuard:
    def test_within_limit_untouched(self):
        r = _result_with_chunks()
        out = apply_response_size_guard(r, max_bytes=10_000_000)
        assert out is r
        assert all(c["text"] for s in r["sources"] for c in s["chunks"])
        assert all(not _is_omitted(c) for s in r["sources"] for c in s["chunks"])
        assert "warning" not in r

    def test_over_limit_replaces_tail_with_rerun_call(self):
        r = _result_with_chunks()
        size = _response_bytes(r)
        out = apply_response_size_guard(r, max_bytes=size - 1)

        p2c3 = out["sources"][1]["chunks"][0]
        assert p2c3["text"] == _omitted_call("p2", 0)
        assert _is_omitted(p2c3)
        for c in out["sources"][0]["chunks"]:
            assert c["text"] == "данные " * 800
            assert not _is_omitted(c)
        assert _response_bytes(out) <= size - 1
        assert "warning" not in out

    def test_deep_overflow_marks_every_chunk_with_its_own_call(self):
        r = _result_with_chunks()
        out = apply_response_size_guard(r, max_bytes=1000)
        for source, pid in zip(out["sources"], ("p1", "p2")):
            for c in source["chunks"]:
                assert _is_omitted(c)
                assert c["text"] == _omitted_call(pid, c["chunk_index"])

    def test_omission_order_is_tail_first(self):
        """If any p1 chunk is omitted, then every p2 chunk is omitted too —
        the tail is cleared first."""
        r = _result_with_chunks()
        size = _response_bytes(r)
        out = apply_response_size_guard(r, max_bytes=size - 40000)
        p1_omitted = [_is_omitted(c) for c in out["sources"][0]["chunks"]]
        p2_omitted = [_is_omitted(c) for c in out["sources"][1]["chunks"]]
        assert any(p2_omitted)
        if any(p1_omitted):
            assert all(p2_omitted)

    def test_reentry_is_noop(self):
        r = _result_with_chunks()
        out = apply_response_size_guard(r, max_bytes=1000)
        snapshot = json.dumps(out, sort_keys=True)
        out2 = apply_response_size_guard(out, max_bytes=1000)
        assert json.dumps(out2, sort_keys=True) == snapshot

    def test_empty_sources_noop(self):
        r = {"sources": []}
        out = apply_response_size_guard(r, max_bytes=10)
        assert out["sources"] == []
        assert "warning" not in r
