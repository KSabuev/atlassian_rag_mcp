"""XLS (legacy Excel) → text via xlrd."""

from __future__ import annotations

import xlrd

from .registry import converter


@converter(name="xls", exts=("xls",))
def convert(data: bytes, filename: str = "") -> str:
    """Convert a legacy .xls (OLE compound) workbook.

    A misnamed XLSX (PK zip) with a .xls extension is routed to the XLSX
    converter.
    """
    if data[:4] == b"PK\x03\x04":
        from . import xlsx
        return xlsx.convert(data, filename)
    book = xlrd.open_workbook(file_contents=data)
    parts: list[str] = []
    for sheet in book.sheets():
        parts.append(f"--- Sheet: {sheet.name} ---")
        for rx in range(sheet.nrows):
            vals = [str(sheet.cell_value(rx, cx)).strip() for cx in range(sheet.ncols)]
            if any(vals):
                parts.append(" | ".join(vals))
    return "\n".join(parts)
