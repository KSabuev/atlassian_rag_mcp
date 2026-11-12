"""Attachment text extraction.

Converters are self-describing modules that register themselves with the
``@converter`` decorator (declared extensions + optional content signature +
priority). The package auto-discovers them on import, so adding a converter is
a single new file — nothing central is edited. ``extract_text(filename, data)``
routes a file to a converter by extension first, then by content signature,
with Office-zip / OLE container handling and text fallbacks.
"""

from __future__ import annotations

import pkgutil
from importlib import import_module

from . import text
from .classify import classify
from .download import download_attachment
from .errors import AttachmentError, AttachmentTooLarge, ConversionError, UnsupportedFormat
from .registry import by_ext, by_name, with_magic
from .sniff import is_ole, zip_names

__all__ = [
    "extract_text",
    "classify",
    "download_attachment",
    "AttachmentError",
    "AttachmentTooLarge",
    "ConversionError",
    "UnsupportedFormat",
]

for _module in pkgutil.iter_modules(__path__):
    import_module(f"{__name__}.{_module.name}")


def _ext(filename: str) -> str:
    name = filename or ""
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def _safe(fn, data: bytes, filename: str) -> str:
    """Call a converter, wrapping unexpected errors as ConversionError."""
    try:
        return fn(data, filename)
    except AttachmentError:
        raise
    except Exception as e:
        raise ConversionError(f"conversion failed for {filename or '(no name)'}: {e}") from e


def extract_text(filename: str, data: bytes) -> str:
    """Extract text from attachment bytes, routing by extension then content."""
    conv = by_ext(_ext(filename))
    if conv is not None:
        return _safe(conv.convert, data, filename)

    for c in with_magic():
        if c.magic(data):
            return _safe(c.convert, data, filename)

    if zip_names(data) is not None:
        return "zip archive (skipped)"

    if is_ole(data):
        last: ConversionError | None = None
        for name in ("doc", "xls"):
            try:
                return _safe(by_name(name).convert, data, filename)
            except ConversionError as e:
                last = e
        raise UnsupportedFormat(f"OLE binary {filename or '(no name)'}: conversion failed") from last

    if data[:100].lstrip().startswith(b"<?xml"):
        return _safe(text.convert, data, filename)
    if b"\x00" in data:
        raise UnsupportedFormat(f"binary file {filename or '(no name)'}")
    return _safe(text.convert, data, filename)
