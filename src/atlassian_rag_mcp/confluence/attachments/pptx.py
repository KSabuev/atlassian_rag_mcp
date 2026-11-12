"""PPTX (PowerPoint) → text: slide texts and tables."""

from __future__ import annotations

from io import BytesIO

from pptx import Presentation

from .registry import converter
from .sniff import has_part, is_zip


@converter(name="pptx", exts=("pptx",), magic=lambda d: is_zip(d) and has_part(d, "ppt/"), priority=20)
def convert(data: bytes, filename: str = "") -> str:
    prs = Presentation(BytesIO(data))
    parts: list[str] = []
    for i, slide in enumerate(prs.slides, 1):
        parts.append(f"--- Slide {i} ---")
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs).strip()
                    if text:
                        parts.append(text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append(" | ".join(cells))
    return "\n".join(parts)
