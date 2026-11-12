"""Shared fixtures — isolated unit test helpers, no live services."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atlassian_rag_mcp.confluence.models import ChunkPayload, CqlResult, PageContent
from atlassian_rag_mcp.confluence.search import SearchDeps


@pytest.fixture
def vector_db():
    db = MagicMock()
    db.search = AsyncMock(return_value=[])
    db.scroll_points = AsyncMock(return_value=([], None))
    return db


@pytest.fixture
def embedding():
    emb = MagicMock()
    emb.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    emb.get_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    return emb


@pytest.fixture
def confluence_client():
    conf = MagicMock()
    conf.cql = AsyncMock(return_value=CqlResult(results=[]))
    conf.get_page = AsyncMock(return_value=PageContent(
        page_title="Test Page",
        html="<p>test</p>",
        version=1,
        last_updated="2024-01-01",
    ))
    return conf


@pytest.fixture
def fts_index():
    idx = MagicMock()
    idx.search = AsyncMock(return_value=[])
    return idx


@pytest.fixture
def search_deps(vector_db, embedding, confluence_client, fts_index):
    return SearchDeps(
        vector_db=vector_db,
        embedding_client=embedding,
        confluence_client=confluence_client,
        fts_index=fts_index,
        top_k=25,
        confluence_space_key="TEST",
        confluence_url="https://c.example.com/",
    )



def make_payload(**overrides) -> dict:
    base = {
        "chunk_id": "chunk1",
        "chunk_type": "section",
        "text": "some text",
        "page_id": "p1",
        "page_title": "Page 1",
        "page_version": 1,
        "parent_id": "",
        "block_index": None,
        "block_part_id": None,
        "chunk_index": 0,
        "updated_at": "2024-01-01",
        "heading_hierarchy": [],
    }
    base.update(overrides)
    return ChunkPayload(**base).model_dump()
