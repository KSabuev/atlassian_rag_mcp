"""DOC (legacy Word) → text via antiword, with catdoc fallback.

Also cross-detects misnamed files: an RTF payload with a ``.doc`` name is
routed to the RTF converter.
"""

from __future__ import annotations

import shutil
import subprocess

from .errors import ConversionError
from .registry import converter


def _run(cmd: list[str], data: bytes) -> str:
    proc = subprocess.run(cmd, input=data, capture_output=True, timeout=120)
    if proc.returncode != 0:
        raise ConversionError(
            f"{cmd[0]} failed: {proc.stderr.decode(errors='replace').strip()[:200]}"
        )
    return proc.stdout.decode(errors="replace")


@converter(name="doc", exts=("doc",))
def convert(data: bytes, filename: str = "") -> str:
    """Convert a legacy .doc file.

    A misnamed RTF with a .doc extension is routed to the RTF converter;
    otherwise the first available of antiword/catdoc is used.
    """
    if data.lstrip()[:5] == b"{\\rtf":
        from . import rtf
        return rtf.convert(data, filename)

    tried: list[str] = []
    for tool in ("antiword", "catdoc"):
        if shutil.which(tool) is None:
            continue
        cmd = [tool, "-m", "UTF-8"] if tool == "antiword" else [tool, "-d", "utf-8"]
        try:
            return _run(cmd, data)
        except ConversionError as e:
            tried.append(str(e))
    if tried:
        raise ConversionError("doc: all converters failed: " + " | ".join(tried))
    raise ConversionError("doc: neither antiword nor catdoc is installed")
