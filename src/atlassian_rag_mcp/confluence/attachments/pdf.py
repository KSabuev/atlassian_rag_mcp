"""PDF → text via pdfplumber."""

from __future__ import annotations

from io import BytesIO

import pdfplumber

from .registry import converter


@converter(name="pdf", exts=("pdf",), magic=lambda d: d[:5] == b"%PDF-", priority=10)
def convert(data: bytes, filename: str = "") -> str:
    parts: list[str] = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = (page.extract_text() or "").strip()
            if text:
                parts.append(f"--- page {i} ---\n{text}")
    return "\n".join(parts)
