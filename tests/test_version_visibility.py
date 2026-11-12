"""Search must serve a single complete page version — never a mix.

Pages currently being (re)indexed have an "in-work" page_version; chunks
belonging to that in-work version must be hidden so search only returns the
last completed version.
"""

from __future__ import annotations

from atlassian_rag_mcp.confluence.search.base import (
    AlgorithmResult,
    _build_rag_result,
)
from atlassian_rag_mcp.confluence.search.vector_fts_fusion import VectorFtsFusionResult


def _item(page_id, version, chunk_id, text="txt"):
    return {
        "score": 1.0,
        "vector_score": 1.0,
        "fts_score": 0.0,
        "page_id": page_id,
        "page_title": "T",
        "page_url": "https://c.example.com/pages/viewpage.action?pageId=" + page_id,
        "page_version": version,
        "parent_id": None,
        "updated_at": None,
        "chunk": {
            "chunk_id": chunk_id,
            "text": text,
            "chunk_type": "text",
            "chunk_index": 0,
            "block_index": 0,
            "block_part_id": 0,
            "heading_hierarchy": [],
        },
    }


class TestInWorkVersionFiltering:
    def test_old_version_served_while_new_in_work(self):
        result = AlgorithmResult(results=[
            _item("P1", 2, "old1"),
            _item("P1", 2, "old2"),
            _item("P1", 3, "new1"),
        ])
        out = _build_rag_result(result, in_work_versions={"P1": 3}, result_model=VectorFtsFusionResult)
        sources = out["sources"]
        assert len(sources) == 1
        assert sources[0]["version"] == 2
        assert {c["chunk_id"] for c in sources[0]["chunks"]} == {"old1", "old2"}

    def test_new_version_served_after_no_longer_in_work(self):
        result = AlgorithmResult(results=[
            _item("P1", 3, "new1"),
            _item("P1", 3, "new2"),
        ])
        out = _build_rag_result(result, in_work_versions={}, result_model=VectorFtsFusionResult)
        sources = out["sources"]
        assert sources[0]["version"] == 3
        assert {c["chunk_id"] for c in sources[0]["chunks"]} == {"new1", "new2"}

    def test_mixed_versions_all_hidden_returns_empty(self):
        result = AlgorithmResult(results=[
            _item("P1", 3, "a"),
            _item("P1", 3, "b"),
        ])
        out = _build_rag_result(result, in_work_versions={"P1": 3}, result_model=VectorFtsFusionResult)
        assert out["sources"] == []

    def test_completed_version_not_hidden(self):
        result = AlgorithmResult(results=[_item("P1", 4, "c")])
        out = _build_rag_result(result, in_work_versions={"P1": 5}, result_model=VectorFtsFusionResult)
        assert len(out["sources"]) == 1
        assert out["sources"][0]["version"] == 4
        assert len(out["sources"][0]["chunks"]) == 1

    def test_other_pages_unaffected(self):
        result = AlgorithmResult(results=[
            _item("A", 1, "a1"),
            _item("P1", 3, "n1"),
            _item("P1", 2, "o1"),
        ])
        out = _build_rag_result(result, in_work_versions={"P1": 3}, result_model=VectorFtsFusionResult)
        ids = {s["page_id"] for s in out["sources"]}
        assert ids == {"A", "P1"}
        p1 = next(s for s in out["sources"] if s["page_id"] == "P1")
        assert p1["version"] == 2

    def test_no_in_work_versions_no_change(self):
        result = AlgorithmResult(results=[
            _item("P1", 2, "o1"),
            _item("P1", 3, "n1"),
        ])
        out = _build_rag_result(result, in_work_versions=None, result_model=VectorFtsFusionResult)
        assert len(out["sources"]) == 1
        assert len(out["sources"][0]["chunks"]) == 2
