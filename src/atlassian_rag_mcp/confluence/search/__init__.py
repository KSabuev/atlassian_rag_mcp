"""Search algorithms — self-registering classes, auto-discovered per file.

Adding an algorithm is a single new module in this package: it registers
itself with the shared ``Registry`` on import (``@register`` + ``descriptor()``),
mirroring the attachments converter registry.
"""

from __future__ import annotations

import pkgutil
from importlib import import_module

from .base import (
    MethodDescriptor,
    SearchDeps,
    build_page_url,
    execute_algorithm,
    get_algorithms,
    resolve_search_params,
)

for _module in pkgutil.iter_modules(__path__):
    import_module(f"{__name__}.{_module.name}")

__all__ = [
    "MethodDescriptor",
    "SearchDeps",
    "build_page_url",
    "execute_algorithm",
    "get_algorithms",
    "resolve_search_params",
]
