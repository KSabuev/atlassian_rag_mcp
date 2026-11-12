"""Pydantic models — validation, defaults, and edge cases."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from atlassian_rag_mcp.api.health import HealthCheckResult, ServiceStatus
from atlassian_rag_mcp.confluence.models import (
    AttachmentInfo,
    Chunk,
    ChunkPayload,
    Page,
    PageContent,
)
from atlassian_rag_mcp.confluence.search.base import SearchResult
from atlassian_rag_mcp.confluence.search.cql_fts_fusion import (
    CqlFtsFusionResult,
    CqlFtsFusionSource,
)
from atlassian_rag_mcp.confluence.search.fulltext_search import (
    FullTextSearchResult,
    FullTextSearchSource,
)
from atlassian_rag_mcp.confluence.search.vector_fts_fusion import (
    VectorFtsFusionResult,
    VectorFtsFusionSource,
)


class TestPageContentModels:
    def test_page_content_result_defaults(self):
        r = PageContent(page_id="123")
        assert r.chunks == []
        assert r.files == []
        assert r.error is None
        assert r.version == 0

    def test_page_content_result_with_error(self):
        r = PageContent(page_id="123", error="what='table' requires block_index")
        assert r.error == "what='table' requires block_index"

    def test_page_content_result_serialization_roundtrip(self):
        r = PageContent(
            page_id="123", page_title="T", version=9,
            chunks=[Chunk(chunk_id="c1", text="t", chunk_type="table_row",
                          block_index=1, block_part_id=1)],
            files=[AttachmentInfo(title="a.xsd", size=10,
                                  download_url="https://x/download/attachments/123/a.xsd")],
        )
        r2 = PageContent.model_validate(r.model_dump())
        assert r2.chunks == r.chunks
        assert r2.files == r.files

    def test_attachment_info_defaults(self):
        a = AttachmentInfo(title="f.xml")
        assert a.media_type is None
        assert a.size is None
        assert a.version_number is None
        assert a.created is None
        assert a.download_url is None

    def test_attachment_info_requires_title(self):
        with pytest.raises(ValidationError):
            AttachmentInfo()


class TestChunkPayload:
    def test_minimal(self):
        cp = ChunkPayload(
            chunk_id="c1", chunk_type="section", text="content",
            page_id="p1", page_title="Title",
        )
        assert cp.chunk_id == "c1"
        assert cp.page_version is None
        assert cp.heading_hierarchy == []
        assert cp.chunk_index == 0

    def test_full(self):
        cp = ChunkPayload(
            chunk_id="c2", chunk_type="table_row", text="data",
            page_id="p2", page_title="T", page_version=3,
            parent_id="parent", block_index=1, block_part_id=2,
            chunk_index=5, updated_at="2024-01-01T00:00:00",
            heading_hierarchy=["H1", "H2"],
        )
        assert cp.page_version == 3
        assert cp.block_index == 1
        assert cp.heading_hierarchy == ["H1", "H2"]

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ChunkPayload(chunk_id="c1")


class TestServiceStatus:
    def test_success(self):
        s = ServiceStatus(status="ok", dimension=384, model="text-embedding-3-small")
        assert s.status == "ok"
        assert s.dimension == 384

    def test_error(self):
        s = ServiceStatus(status="error", error="Connection refused")
        assert s.error == "Connection refused"

    def test_minimal(self):
        s = ServiceStatus(status="unknown")
        assert s.dimension is None
        assert s.model is None
        assert s.error is None


class TestHealthCheckResult:
    def test_all_ok(self):
        h = HealthCheckResult(
            status="healthy",
            services={
                "confluence": ServiceStatus(status="ok"),
                "vector_db": ServiceStatus(status="ok"),
            },
        )
        assert h.status == "healthy"
        assert len(h.services) == 2
        assert "confluence" in h.services

    def test_degraded(self):
        h = HealthCheckResult(
            status="degraded",
            services={
                "confluence": ServiceStatus(status="ok"),
                "embedding": ServiceStatus(status="error", error="down"),
            },
        )
        assert h.status == "degraded"
        assert h.services["embedding"].error == "down"


class TestPageMetadata:
    def test_full(self):
        p = Page(
            page_id="123", page_title="My Page",
            page_url="https://confluence.com/pages/123",
            version=4, last_updated="2024-06-01",
        )
        assert p.page_id == "123"
        assert p.version == 4

    def test_no_chunks_field(self):
        p = Page(page_id="123")
        assert "chunks" not in p.model_dump()

    def test_error(self):
        p = Page(error="Page not found")
        assert p.error == "Page not found"
        assert p.page_id is None

    def test_all_defaults(self):
        p = Page()
        assert p.page_id is None
        assert p.version == 0


class TestChunk:
    def test_minimal(self):
        c = Chunk(chunk_id="c1", text="content", chunk_type="section")
        assert c.block_index is None
        assert c.heading_hierarchy == []

    def test_full(self):
        c = Chunk(
            chunk_id="c1", text="content", chunk_type="section",
            block_index=0, block_part_id=1,
            heading_hierarchy=["A", "B"],
        )
        assert c.block_index == 0
        assert c.heading_hierarchy == ["A", "B"]

    def test_missing_required(self):
        with pytest.raises(ValidationError):
            Chunk(chunk_id="c1")


class TestChunkSources:
    """vector_fts_fusion source: chunk-based, per-branch scores."""

    def test_defaults(self):
        d = VectorFtsFusionSource(page_id="123", page_title="P").model_dump()
        assert "score" not in d
        assert d["vector_score"] == 0.0
        assert d["fts_score"] == 0.0
        assert d["chunks"] == []

    def test_with_chunks(self):
        sp = VectorFtsFusionSource(
            page_id="123", page_title="P", page_url="https://c.com/p",
            version=2, parent_id="0", vector_score=0.9, fts_score=0.1,
            chunks=[Chunk(chunk_id="c1", text="t", chunk_type="section")],
        )
        assert sp.page_url == "https://c.com/p"
        assert sp.version == 2
        assert sp.chunks[0].chunk_id == "c1"
        assert "excerpt" not in sp.model_dump()


class TestFullTextSearchSource:
    """fulltext_search source: chunk-based, FTS score only (no vector branch)."""

    def test_defaults(self):
        d = FullTextSearchSource(page_id="123", page_title="P").model_dump()
        assert "score" not in d
        assert "vector_score" not in d
        assert d["fts_score"] == 0.0
        assert d["chunks"] == []

    def test_with_chunks(self):
        sp = FullTextSearchSource(
            page_id="123", page_title="P", page_url="https://c.com/p",
            version=2, fts_score=7.5,
            chunks=[Chunk(chunk_id="c1", text="t", chunk_type="section")],
        )
        assert sp.fts_score == 7.5
        assert sp.chunks[0].chunk_id == "c1"
        d = sp.model_dump()
        assert "excerpt" not in d
        assert "vector_score" not in d
        assert "score" not in d


class TestCqlFtsFusionSource:
    """cql_fts_fusion source: excerpt-based page with fusion and per-branch scores."""

    def test_defaults(self):
        d = CqlFtsFusionSource(page_id="123", page_title="P").model_dump()
        assert d["excerpt"] is None
        assert d["score"] == 0.0
        assert d["fts_score"] == 0.0
        assert d["cql_rank"] is None
        assert "chunks" not in d

    def test_with_excerpt(self):
        sp = CqlFtsFusionSource(
            page_id="123", page_title="P", page_url="https://c.com/p",
            version=2, excerpt="matched text",
            score=0.8, fts_score=4.2, cql_rank=3,
        )
        assert sp.excerpt == "matched text"
        assert sp.score == 0.8
        assert sp.fts_score == 4.2
        assert sp.cql_rank == 3
        d = sp.model_dump()
        assert "chunks" not in d
        assert "vector_score" not in d


class TestSearchResults:
    def test_empty_success(self):
        r = SearchResult(sources=[])
        assert r.error is None

    def test_with_sources(self):
        sp = CqlFtsFusionSource(page_id="123", page_title="P")
        r = SearchResult(sources=[sp])
        assert len(r.sources) == 1
        assert r.sources[0].page_id == "123"

    def test_with_error(self):
        r = SearchResult(sources=[], error="Query failed")
        assert r.error == "Query failed"

    @pytest.mark.parametrize("result_cls,source_cls", [
        (VectorFtsFusionResult, VectorFtsFusionSource),
        (FullTextSearchResult, FullTextSearchSource),
        (CqlFtsFusionResult, CqlFtsFusionSource),
    ])
    def test_algorithm_result(self, result_cls, source_cls):
        r = result_cls(sources=[source_cls(page_id="1", page_title="P")])
        assert len(r.sources) == 1
        assert type(r.sources[0]) is source_cls

