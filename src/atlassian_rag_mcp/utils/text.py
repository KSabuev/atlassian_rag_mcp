"""Text normalization utilities (HTML -> plain text).

Shared by every layer that consumes chunk text for model input (FTS index,
vector embeddings) while the stored chunk payload keeps the full
HTML.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Comment

_WS_RE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    """Convert an HTML fragment to plain text (tags stripped, comments removed).

    Plain (non-HTML) input passes through unchanged apart from whitespace
    collapsing. Empty/None input yields "".
    """
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return _WS_RE.sub(" ", html).strip()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()
    text = soup.get_text(" ")
    return _WS_RE.sub(" ", text).strip()
