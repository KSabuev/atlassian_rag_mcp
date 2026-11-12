"""Shared byte-signature helpers for converter ``magic`` predicates."""

from __future__ import annotations

import zipfile
from io import BytesIO

OLE_MAGIC = b"\xd0\xcf\x11\xe0"


def starts_with(data: bytes, *prefixes: bytes) -> bool:
    return any(data.lstrip().startswith(p) for p in prefixes)


def is_zip(data: bytes) -> bool:
    return data[:4] == b"PK\x03\x04"


def zip_names(data: bytes) -> list[str] | None:
    """Return the zip member names, or None if ``data`` is not a readable zip."""
    if not is_zip(data):
        return None
    try:
        return zipfile.ZipFile(BytesIO(data)).namelist()
    except Exception:
        return None


def has_part(data: bytes, part: str) -> bool:
    """True if the zip container holds a member starting with ``part``."""
    names = zip_names(data)
    return names is not None and any(n.startswith(part) for n in names)


def is_ole(data: bytes) -> bool:
    return data[:4] == OLE_MAGIC


def is_xml(data: bytes) -> bool:
    return data.lstrip()[:5] == b"<?xml"
