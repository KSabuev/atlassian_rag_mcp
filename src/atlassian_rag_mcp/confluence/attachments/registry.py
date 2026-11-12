"""Self-describing, self-registering converter registry.

A converter is a ``convert(data: bytes, filename: str) -> str`` function
decorated with :func:`converter`. The decorator records the converter's
name, the file extensions it owns, an optional content-sniffing predicate
(``magic``) and a dispatch priority. Adding a new converter is a single file
that auto-registers on import; nothing central needs to be edited.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

ConverterFn = Callable[[bytes, str], str]
MagicFn = Callable[[bytes], bool]


@dataclass(frozen=True)
class Converter:
    """Registered attachment converter; a lower priority is tried earlier
    during magic dispatch."""
    name: str
    convert: ConverterFn
    exts: tuple[str, ...] = ()
    magic: MagicFn | None = None
    priority: int = 100


_REGISTRY: list[Converter] = []


def converter(
    *,
    name: str,
    exts: Sequence[str] = (),
    magic: MagicFn | None = None,
    priority: int = 100,
) -> Callable[[ConverterFn], ConverterFn]:
    """Register ``convert`` as a named converter.

    ``exts`` are the lowercase file extensions the converter handles.
    ``magic`` is an optional predicate over raw bytes used when the file has
    no (matching) extension. ``priority`` orders magic matches (lower first).
    """

    def deco(fn: ConverterFn) -> ConverterFn:
        if any(c.name == name for c in _REGISTRY):
            raise ValueError(f"duplicate converter name: {name!r}")
        _REGISTRY.append(
            Converter(name=name, convert=fn, exts=tuple(exts), magic=magic, priority=priority),
        )
        return fn

    return deco


def all_converters() -> list[Converter]:
    return sorted(_REGISTRY, key=lambda c: c.priority)


def by_ext(ext: str) -> Converter | None:
    return next((c for c in _REGISTRY if ext in c.exts), None)


def by_name(name: str) -> Converter | None:
    return next((c for c in _REGISTRY if c.name == name), None)


def with_magic() -> list[Converter]:
    return [c for c in all_converters() if c.magic is not None]
