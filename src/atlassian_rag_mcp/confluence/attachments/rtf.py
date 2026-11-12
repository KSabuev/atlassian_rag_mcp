"""RTF → text via striprtf.

Handles the common non-standard case where Cyrillic (or other) text is
stored as raw bytes in the file rather than as ``\\uN`` / ``\\'xx``
escapes. Such files declare the byte encoding via ``\\ansicpgNNNN``; we
detect high bytes and decode with that code page before stripping.
"""

from __future__ import annotations

import re

from striprtf.striprtf import rtf_to_text

from .registry import converter

_ANSICPG_TO_CODEC = {
    1251: "cp1251",
    1252: "cp1252",
    1250: "cp1250",
    1253: "cp1253",
    1254: "cp1254",
    1255: "cp1255",
    1256: "cp1256",
    1257: "cp1257",
    1258: "cp1258",
    874: "cp874",
    932: "cp932",
    936: "cp936",
    949: "cp949",
    950: "cp950",
    1200: "utf-16-le",
    1201: "utf-16-be",
    65001: "utf-8",
    20127: "utf-8",
}


def _detect_codec(data: bytes) -> str:
    head = data[:4000].decode("latin-1", errors="replace")
    m = re.search(r"\\ansicpg(\d+)", head)
    if m:
        return _ANSICPG_TO_CODEC.get(int(m.group(1)), "cp1251")
    return "cp1251"


@converter(name="rtf", exts=("rtf",), magic=lambda d: d[:5] == b"{\\rtf", priority=10)
def convert(data: bytes, filename: str = "") -> str:
    """Convert RTF to text.

    Proper 7-bit RTF has only ASCII bytes; "raw-byte" RTF (common for
    Cyrillic) stores high bytes directly and declares \\ansicpgNNNN — the
    codec is then detected from that directive (default cp1251).
    """
    if any(b >= 0x80 for b in data):
        text = data.decode(_detect_codec(data), errors="replace")
    else:
        text = data.decode("latin-1", errors="replace")
    return rtf_to_text(text)
