"""Qwen3 tokenizer — bundled with the project, real-token counting/splitting.

The tokenizer file (``qwen3_tokenizer.json``) ships inside the package, so
no runtime download and no cache dir are needed. ``get_qwen3_tokenizer``
loads it once (cached in-process) and raises when it is missing — there is
no byte fallback: chunk and embedding budgets are measured in REAL tokens
only. ``count_qwen3_tokens`` counts actual subword tokens and
``split_overlap_tokens`` cuts text by a real token budget.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_TOKENIZER_PATH = Path(__file__).parent / "qwen3_tokenizer.json"

_tokenizer: Any = None
_tokenizer_lock = threading.Lock()


def get_qwen3_tokenizer() -> Any:
    """Load the Qwen3 tokenizer from the bundled file (cached in-process).

    Raises RuntimeError when the file is missing or unreadable — the
    budgets are measured in real tokens only, there is no fallback.
    """
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    with _tokenizer_lock:
        if _tokenizer is not None:
            return _tokenizer
        try:
            from tokenizers import Tokenizer

            _tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load the bundled Qwen3 tokenizer from "
                f"{_TOKENIZER_PATH}: {e}"
            ) from e
        return _tokenizer


def count_qwen3_tokens(text: str) -> int:
    """Exact subword token count of *text* under the Qwen3 tokenizer."""
    return len(get_qwen3_tokenizer().encode(text).ids)


def split_overlap_tokens(
    text: str,
    counter: Any,
    max_tokens: int = 8000,
    overlap_tokens: int = 200,
) -> list[str]:
    """Split *text* into segments of at most ``max_tokens`` REAL tokens.

    ``counter(text)`` must be monotone in the text (more chars -> no fewer
    tokens), which holds for any real tokenizer; the split points are found
    by binary search over character boundaries. Adjacent segments share a
    trailing window of at most ``overlap_tokens`` tokens.

    Returns ``[text]`` when the text already fits. When even a single
    character exceeds the budget (one unbreakable unit — e.g. a 50k-char
    base64 blob), the segment is cut at a character count of
    ``max_tokens``: no split can make it fit, so the caller's embedding
    failure surfaces instead of silent infinite recursion.
    """
    if counter(text) <= max_tokens:
        return [text]

    n = len(text)
    segments: list[str] = []
    start = 0
    while start < n:
        lo, hi, end = start + 1, n, -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if counter(text[start:mid]) <= max_tokens:
                end = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if end <= start:
            end = min(n, start + max(1, max_tokens))
        segments.append(text[start:end])
        if end >= n:
            break
        lo, hi, nxt = start + 1, end, end
        while lo <= hi:
            mid = (lo + hi) // 2
            if counter(text[mid:end]) <= overlap_tokens:
                nxt = mid
                hi = mid - 1
            else:
                lo = mid + 1
        if nxt <= start:
            nxt = start + 1
        start = nxt
    return segments
