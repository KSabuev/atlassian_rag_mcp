"""Drawio (mxGraph XML) → text: node labels and edge labels.

Handles plain XML ``<mxfile>`` (with embedded ``<mxGraphModel>``) and
diagrams stored as base64 + deflate inside ``<diagram>`` elements.
"""

from __future__ import annotations

import base64
import html as html_mod
import re
import urllib.parse
import xml.etree.ElementTree as ET
import zlib

from .errors import ConversionError
from .registry import converter
from .sniff import is_xml, starts_with


def _strip_html(value: str) -> str:
    value = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    value = re.sub(r"</(div|p|li|tr)>", "\n", value, flags=re.I)
    value = re.sub(r"<td[^>]*>", " | ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    value = html_mod.unescape(value)
    value = re.sub(r"[ \t]+\n", "\n", value)
    value = re.sub(r"\n{2,}", "\n", value)
    return value.strip()


def _inflate(raw: bytes) -> bytes | None:
    for wbits in (-15, 15, 31, 47):
        try:
            return zlib.decompress(raw, wbits)
        except zlib.error:
            continue
    return None


def _decode_diagram_inner(inner: str) -> str:
    """Decode a ``<diagram>`` text payload to XML text.

    Handles, in order:
      base64(deflate(percent-encoded xml))   — drawio default for compressed
      base64(deflate(xml))
      base64(xml)                             — base64 of plain (uncompressed)
      plain xml / percent-encoded xml
    """
    inner = inner.strip()
    result: str | None = None
    raw = None
    try:
        raw = base64.b64decode(inner, validate=True)
    except (ValueError, TypeError):
        raw = None

    if raw is not None:
        inflated = _inflate(raw)
        if inflated is not None:
            result = inflated.decode("utf-8", errors="replace")
        else:
            result = raw.decode("utf-8", errors="replace")
    else:
        result = inner

    result = result.strip()
    if result.startswith("%"):
        result = urllib.parse.unquote(result)
    return result


def _collect_cells(model: ET.Element, nodes: list[str], edges: list[str]) -> None:
    for cell in model.iter("mxCell"):
        value = _strip_html(cell.get("value") or "")
        if not value:
            continue
        if cell.get("edge") in ("1", "true"):
            edges.append(value)
        else:
            nodes.append(value)


def _models_from_xml(text: str) -> list[ET.Element]:
    root = ET.fromstring(text)
    if root.tag == "mxfile":
        models: list[ET.Element] = []
        for diagram in root.iter("diagram"):
            inner = (diagram.text or "").strip()
            if inner:
                models.append(ET.fromstring(_decode_diagram_inner(inner)))
            else:
                models.extend(diagram.iter("mxGraphModel"))
        return models
    return [root]


@converter(
    name="drawio",
    exts=("drawio",),
    magic=lambda d: starts_with(d, b"<mxfile", b"<mxGraphModel")
    or (is_xml(d) and (b"mxgraphmodel" in d.lower() or b"mxcell" in d.lower())),
    priority=30,
)
def convert(data: bytes, filename: str = "") -> str:
    text = data.decode("utf-8", errors="replace").strip()
    if not text.startswith("<"):
        text = _decode_diagram_inner(text)
        if not text.startswith("<"):
            raise ConversionError("not an mxfile/drawio file")
    try:
        models = _models_from_xml(text)
    except ET.ParseError as e:
        raise ConversionError(f"invalid drawio XML: {e}") from e

    nodes: list[str] = []
    edges: list[str] = []
    for m in models:
        _collect_cells(m, nodes, edges)
    if not nodes and not edges:
        return ""

    parts = [f"Diagram: {len(nodes)} nodes, {len(edges)} edges"]
    if nodes:
        parts.append("Nodes:\n" + "\n".join(f"- {n}" for n in nodes))
    if edges:
        parts.append("Edge labels:\n" + "\n".join(f"- {e}" for e in edges))
    return "\n".join(parts)
