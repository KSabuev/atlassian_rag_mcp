"""HTML chunker package (Confluence) — clean + chunk pipeline."""

from __future__ import annotations

from .html_chunker import HtmlChunker
from .pipeline import chunking_pipeline

__all__ = [
    "HtmlChunker",
    "chunking_pipeline",
]
