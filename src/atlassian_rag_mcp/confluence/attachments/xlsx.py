"""XLSX (Excel) → text: per-sheet rows.

Primary path uses openpyxl. Some workbooks carry style extensions (e.g.
``<patternFill><extLst>``) that openpyxl cannot parse, so a lenient
fallback reads the raw sheet / sharedStrings XML directly.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from io import BytesIO

import openpyxl

from .registry import converter
from .sniff import has_part, is_zip

_X = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _col_idx(ref: str) -> int:
    col = 0
    for ch in ref:
        if ch.isalpha():
            col = col * 26 + (ord(ch.upper()) - 64)
        else:
            break
    return col - 1


def _shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        xml = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(xml)
    return ["".join(t.text or "" for t in si.iter(_X + "t")) for si in root.findall(_X + "si")]


def _cell_value(c: ET.Element, shared: list[str]) -> str:
    t = c.get("t")
    if t == "inlineStr":
        return "".join(x.text or "" for x in c.iter(_X + "t"))
    v = c.find(_X + "v")
    if v is None or v.text is None:
        return ""
    if t == "s":
        try:
            return shared[int(v.text)]
        except (ValueError, IndexError):
            return ""
    if t == "b":
        return "TRUE" if v.text == "1" else "FALSE"
    return v.text


def _sheet_rows(zf: zipfile.ZipFile, sheet_path: str, shared: list[str]) -> list[str]:
    root = ET.fromstring(zf.read(sheet_path))
    lines: list[str] = []
    for row in root.iter(_X + "row"):
        cells: dict[int, str] = {}
        for c in row.findall(_X + "c"):
            ref = c.get("r") or ""
            idx = _col_idx(ref) if ref else len(cells)
            cells[idx] = _cell_value(c, shared)
        if not any(v.strip() for v in cells.values()):
            continue
        lines.append(" | ".join(cells.get(i, "") for i in range(max(cells) + 1)))
    return lines


def _via_raw_xml(data: bytes) -> str:
    zf = zipfile.ZipFile(BytesIO(data))
    shared = _shared_strings(zf)

    names: list[str] = []
    order: list[str] = []
    try:
        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {r.get("Id"): r.get("Target") for r in rels}
        for s in wb.find(_X + "sheets"):
            target = rid_to_target.get(s.get(_R + "id"), "")
            if target.startswith("/"):
                target = target.lstrip("/")
            elif target:
                target = "xl/" + target
            names.append(s.get("name", ""))
            order.append(target)
    except (KeyError, ET.ParseError):
        pass

    if not order:
        order = sorted(n for n in zf.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
        if not order:
            raise ValueError("no worksheets found in xlsx")
        names = [f"Sheet {i + 1}" for i in range(len(order))]

    parts: list[str] = []
    for i, path in enumerate(order):
        title = names[i] if i < len(names) and names[i] else f"Sheet {i + 1}"
        try:
            rows = _sheet_rows(zf, path, shared)
        except (KeyError, ET.ParseError):
            rows = []
        parts.append(f"--- Sheet: {title} ---")
        parts.extend(rows)
    return "\n".join(parts)


def _via_openpyxl(data: bytes) -> str:
    wb = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=True)
    try:
        parts: list[str] = []
        for ws in wb.worksheets:
            parts.append(f"--- Sheet: {ws.title} ---")
            for row in ws.iter_rows(values_only=True):
                vals = ["" if v is None else str(v) for v in row]
                if any(v.strip() for v in vals):
                    parts.append(" | ".join(vals))
    finally:
        wb.close()
    return "\n".join(parts)


@converter(name="xlsx", exts=("xlsx",), magic=lambda d: is_zip(d) and has_part(d, "xl/"), priority=20)
def convert(data: bytes, filename: str = "") -> str:
    """Convert an .xlsx (OOXML) workbook.

    A misnamed legacy .xls (OLE compound) with an .xlsx extension is routed
    to the XLS converter.
    """
    if data[:4] == b"\xd0\xcf\x11\xe0":
        from . import xls
        return xls.convert(data, filename)
    try:
        return _via_openpyxl(data)
    except Exception:
        return _via_raw_xml(data)
