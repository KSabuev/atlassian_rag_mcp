"""Factory for creating VectorDB instances."""

from __future__ import annotations

from atlassian_rag_mcp.storage.base import VectorDB
from atlassian_rag_mcp.storage.fts.base import FTSIndex


async def create_vector_db(
    backend: str = "lancedb",
    qdrant_host: str = "",
    qdrant_port: int = 6333,
    qdrant_api_key: str = "",
    qdrant_collection: str = "atlassian_chunks",
    lancedb_path: str = "./data/lancedb",
    vector_size: int = 2560,
    upsert_batch_size: int = 200,
) -> VectorDB:
    """Factory: create and start a VectorDB backend (lancedb/qdrant)."""
    if backend == "qdrant":
        from atlassian_rag_mcp.storage.vectordb.qdrant_client import QdrantClient
        db = QdrantClient(
            host=qdrant_host, port=qdrant_port, api_key=qdrant_api_key,
            collection=qdrant_collection, vector_size=vector_size,
            upsert_batch_size=upsert_batch_size,
        )
    else:
        from atlassian_rag_mcp.storage.vectordb.lancedb_client import LanceDBClient
        db = LanceDBClient(db_path=lancedb_path, vector_size=vector_size)
    await db.start()
    return db


async def create_fts_index(index_dir: str = "./data/fts_index") -> FTSIndex:
    """Factory: create and start a Tantivy FTS index."""
    from atlassian_rag_mcp.storage.fts.tantivy_client import TantivyClient
    idx = TantivyClient(index_dir=index_dir)
    await idx.start()
    return idx
