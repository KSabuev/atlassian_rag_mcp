"""DOCX (Word) → text: paragraphs and tables in document order.

Primary path uses python-docx (paragraphs + tables). If python-docx fails on
a particular document (malformed table grids, duplicated part references),
falls back to a lenient direct parse of ``word/document.xml``.
"""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from xml.etree import ElementTree as ET

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .errors import UnsupportedFormat
from .registry import converter
from .sniff import has_part, is_zip

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _via_python_docx(data: bytes) -> str:
    doc = Document(BytesIO(data))
    parts: list[str] = []
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            text = Paragraph(child, doc).text.strip()
            if text:
                parts.append(text)
        elif tag == "tbl":
            table = Table(child, doc)
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
    return "\n".join(parts)


def _via_raw_xml(data: bytes) -> str:
    """Lenient fallback: walk word/document.xml, collect <w:t> text with
    paragraph / cell / line-break structure. Ignores python-docx table-grid
    validation entirely."""
    try:
        zf = zipfile.ZipFile(BytesIO(data))
        xml = zf.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as e:
        raise UnsupportedFormat(f"not a readable docx: {e}") from e
    root = ET.fromstring(xml)
    lines: list[str] = []

    def para_text(p: ET.Element) -> str:
        return "".join(t.text or "" for t in p.iter(_W + "t"))

    def handle_container(el: ET.Element) -> None:
        for child in el:
            tag = child.tag
            if tag == _W + "p":
                txt = para_text(child)
                if txt.strip():
                    lines.append(txt.strip())
            elif tag == _W + "tbl":
                for tr in child:
                    if tr.tag != _W + "tr":
                        continue
                    cells: list[str] = []
                    for tc in tr:
                        if tc.tag != _W + "tc":
                            continue
                        cell_txt = " ".join(
                            para_text(p) for p in tc.iter(_W + "p")
                        ).strip()
                        cells.append(cell_txt)
                    if any(c for c in cells):
                        lines.append(" | ".join(cells))
            else:
                handle_container(child)

    handle_container(root)
    text = "\n".join(lines)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@converter(name="docx", exts=("docx",), magic=lambda d: is_zip(d) and has_part(d, "word/"), priority=20)
def convert(data: bytes, filename: str = "") -> str:
    try:
        return _via_python_docx(data)
    except Exception:
        return _via_raw_xml(data)
