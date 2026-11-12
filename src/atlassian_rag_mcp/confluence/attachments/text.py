"""Plain text formats: txt, xml, csv, sql, json, yaml, wsdl, pem, java, …"""

from __future__ import annotations

from .registry import converter

_ENCODINGS = ("utf-8", "cp1251", "latin-1")


@converter(
    name="text",
    exts=(
        "txt", "log", "md", "markdown", "rst", "tex", "csv", "tsv", "json",
        "xml", "yaml", "yml", "toml", "ini", "cfg", "conf", "env",
        "properties", "srt", "vtt",
    ),
)
def convert(data: bytes, filename: str = "") -> str:
    """Decode bytes to text (utf-8 → cp1251 → latin-1)."""
    for enc in _ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")
