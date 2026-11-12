"""Shared method-registry infrastructure.

Used by the Jira methods registry, the Zephyr methods registry and the
search-algorithms registry. Each method/algorithm is a class with:
  - descriptor() -> MethodDescriptor (id, description, parameter schema)
  - execute(params, *args) -> result   (sync or async — dispatched automatically)

Registration via the ``Registry.register`` decorator. A registry provides:
  - get_methods()  -> list[MethodDescriptor]
  - invoke(cls, params, *args) -> result (sync runs in a worker thread)
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from pydantic import BaseModel, Field


class MethodParameter(BaseModel):
    name: str
    type: str = Field(description='Parameter type: "string", "array", "integer", "boolean", "object"')
    description: str = ""
    required: bool = False


class MethodDescriptor(BaseModel):
    id: str
    description: str = ""
    parameters: list[MethodParameter] = []


class Registry:
    """Name -> method-class registry with sync/async dispatch."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._methods: dict[str, type] = {}

    def register(self, cls: type) -> type:
        desc = cls.descriptor()
        if desc.id in self._methods:
            raise ValueError(f"duplicate method id: {desc.id!r}")
        self._methods[desc.id] = cls
        return cls

    def ids(self) -> list[str]:
        return list(self._methods)

    def __len__(self) -> int:
        return len(self._methods)

    def __contains__(self, method_id: str) -> bool:
        return method_id in self._methods

    def __getitem__(self, method_id: str) -> type:
        return self._methods[method_id]

    def get(self, method_id: str) -> type | None:
        return self._methods.get(method_id)

    def get_methods(self) -> list[MethodDescriptor]:
        """Return list of registered MethodDescriptors."""
        return [cls.descriptor() for cls in self._methods.values()]

    @staticmethod
    async def invoke(cls: type, params: dict[str, Any], *args: Any) -> Any:
        """Run ``cls.execute`` — awaiting async classes, offloading sync ones to a thread."""
        execute = cls.execute
        if inspect.iscoroutinefunction(execute):
            return await execute(params, *args)
        return await asyncio.to_thread(execute, params, *args)



DICT_PARAMS_DESC = (
    "Key-value dict. Only pass parameters the chosen method declares; "
    "unknown keys are rejected with an error naming the field."
)


def projects_example() -> str:
    """Configured Jira project keys rendered as a quoted list for tool descriptions."""
    from atlassian_rag_mcp.config import settings  # deferred: avoids import cycles

    return "[" + ", ".join(f'"{k}"' for k in settings.jira_project_keys) + "]"


def substitute_placeholders(text: str, placeholders: dict[str, str]) -> str:
    """Replace {key} placeholder patterns in ``text`` (e.g. descriptor examples)."""
    if not text or not placeholders:
        return text
    for key, value in placeholders.items():
        text = text.replace(key, value)
    return text


def missing_required(descriptor: MethodDescriptor, params: dict[str, Any]) -> list[str]:
    """Names of required parameters absent from ``params``."""
    return [
        p.name for p in descriptor.parameters
        if p.required and params.get(p.name) in (None, "", [])
    ]


def error_message(e: Exception) -> str:
    """Best-effort human-readable message for an exception (HTTP-aware)."""
    msg = str(e)
    if not msg:
        resp = getattr(e, "response", None)
        msg = f"HTTP {resp.status_code}" if resp is not None else type(e).__name__
    return msg
