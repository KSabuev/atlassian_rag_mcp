"""SVG → text: titles and text labels (normalized whitespace)."""

from __future__ import annotations

import re
from xml.etree import ElementTree as etree

from .registry import converter
from .sniff import is_xml


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


@converter(
    name="svg",
    exts=("svg",),
    magic=lambda d: is_xml(d) and (b"<svg" in d or (b"xmlns" in d and b"svg" in d)),
    priority=50,
)
def convert(data: bytes, filename: str = "") -> str:
    """Extract title and text content from an SVG document.

    itertext() already includes nested <tspan> text — no double count.
    """
    root = etree.fromstring(data)
    parts: list[str] = []
    for el in root.iter():
        tag = el.tag.split("}")[-1] if isinstance(el.tag, str) else ""
        if tag == "title":
            parts.append(_clean(el.text or ""))
        elif tag == "text":
            parts.append(_clean(" ".join(el.itertext())))
    return "\n".join(p for p in parts if p)
