"""Confluence response models for MCP tool outputs."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class CqlResult(BaseModel):
    results: list[dict[str, Any]] = []
    total: int = 0
    query: str = ""
    error: str | None = None


class SpaceResult(BaseModel):
    key: str = ""
    name: str = ""
    homepage_id: str | None = None
    error: str | None = None


class SpacesResult(BaseModel):
    spaces: list[dict[str, Any]] = []
    total: int = 0
    error: str | None = None


class AttachmentInfo(BaseModel):
    title: str
    media_type: str | None = None
    size: int | None = None
    version_number: int | None = None
    created: str | None = None
    download_url: str | None = None


class AttachmentFile(AttachmentInfo):
    """Attachment with extracted text.

    ``status``: ``"ok"`` (text extracted), ``"skipped:<reason>"``
    (video/audio/image/archive/binary/too large), ``"error: <msg>"``
    (download/conversion failed), or ``"not_found"`` (requested title
    does not exist on the page).
    """

    status: str | None = None
    text: str | None = None
    truncated: bool = False


class Chunk(BaseModel):
    chunk_id: str
    text: str
    chunk_type: str
    chunk_index: int | None = None
    block_index: int | None = None
    block_part_id: int | None = None
    heading_hierarchy: list[str] = []
    space_key: str | None = None


class ChunkPayload(BaseModel):
    chunk_id: str
    chunk_type: str
    text: str
    page_id: str
    page_title: str
    page_version: int | None = None
    parent_id: str | None = None
    block_index: int | None = None
    block_part_id: int | None = None
    chunk_index: int = 0
    updated_at: str | None = None
    heading_hierarchy: list[str] = []
    space_key: str | None = None


class PageContext(BaseModel):
    """Page-level metadata shared by all chunks of a page."""

    model_config = ConfigDict(frozen=True)

    page_id: str
    page_title: str
    page_version: int | None = None
    updated_at: str | None = None
    parent_id: str | None = None


def _ensure_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            return [v] if v else []
    return []


def make_payload(raw_payload: dict[str, Any], fallback_id: str | None = None) -> ChunkPayload:
    data = dict(raw_payload)
    hh = data.get("heading_hierarchy")
    if hh is not None:
        data["heading_hierarchy"] = _ensure_list(hh)
    if "chunk_id" not in data and fallback_id:
        data["chunk_id"] = fallback_id
    if "chunk_type" not in data:
        data["chunk_type"] = "section"
    return ChunkPayload(**data)


class Page(BaseModel):
    page_id: str | None = None
    page_title: str | None = None
    page_url: str | None = None
    space_key: str | None = None
    version: int = 0
    last_updated: str | None = None
    parent_id: str | None = None
    error: str | None = None


class PageContent(Page):
    html: str | None = None
    files: list[AttachmentInfo] = []
    attachment_files: list[AttachmentFile] = []
    chunks: list[Chunk] = []


class StructureNode(BaseModel):
    """A single page node of the rendered structure tree."""
    page_id: str
    parent_id: str | None = None
    title: str = ""


class ReconcilerResult(BaseModel):
    """Result of a full index reconciliation."""
    status: str = ""
    pages_scanned: int = 0
    pages_queued: int = 0
    pages_deleted: int = 0
    fts_inconsistent: int = 0


class QueueStatus(BaseModel):
    """Snapshot of the indexing queue state."""
    queue_size: int = 0
    currently_indexing: list[str] = []
    total_indexed_today: int = 0
    empty_pages: int = 0
    last_errors: list[Any] = []
    in_work_versions: dict[str, int] = {}
    dead_letters: dict[str, Any] = {}
