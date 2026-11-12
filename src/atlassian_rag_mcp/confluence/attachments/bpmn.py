"""BPMN 2.0 XML → text: element names and text annotations."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from .errors import ConversionError
from .registry import converter
from .sniff import is_xml


@converter(name="bpmn", exts=("bpmn",), magic=lambda d: is_xml(d) and b"bpmn" in d.lower(), priority=40)
def convert(data: bytes, filename: str = "") -> str:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise ConversionError(f"invalid BPMN XML: {e}") from e

    names: list[str] = []
    for el in root.iter():
        name = el.get("name")
        if name:
            names.append(name)

    texts: list[str] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1] if isinstance(el.tag, str) else ""
        if tag == "text" and el.text and el.text.strip():
            texts.append(el.text.strip())

    parts: list[str] = []
    if names:
        parts.append(f"Elements ({len(names)}):\n" + "\n".join(f"- {n}" for n in names))
    if texts:
        parts.append("Text annotations:\n" + "\n".join(f"- {t}" for t in texts))
    return "\n\n".join(parts)
