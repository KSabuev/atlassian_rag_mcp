"""HTML → text via BeautifulSoup (scripts/styles dropped)."""

from __future__ import annotations

from bs4 import BeautifulSoup

from .registry import converter


@converter(name="html", exts=("html", "htm"))
def convert(data: bytes, filename: str = "") -> str:
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)
