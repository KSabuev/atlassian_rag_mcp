"""Shared method Registry: registration, duplicate handling, sync/async dispatch."""

from __future__ import annotations

import pytest

from atlassian_rag_mcp.registry import MethodDescriptor, MethodParameter, Registry, missing_required


def _make_cls(mid: str, sync: bool = False):
    if sync:
        execute = classmethod(lambda cls_, params, *args: {"kind": "sync", "params": params})
    else:
        async def _execute(cls_, params, *args):
            return {"kind": "async", "params": params}

        execute = classmethod(_execute)

    cls = type(mid.upper(), (), {})
    cls.descriptor = classmethod(lambda cls_: MethodDescriptor(id=mid))
    cls.execute = execute
    return cls


class TestRegistry:
    def test_register_and_lookup(self):
        reg = Registry("t")
        reg.register(_make_cls("alpha"))

        assert "alpha" in reg
        assert reg.get("alpha") is not None
        assert reg.get("nope") is None
        assert reg.ids() == ["alpha"]
        assert len(reg) == 1

    def test_duplicate_id_raises(self):
        reg = Registry("t")
        reg.register(_make_cls("alpha"))

        with pytest.raises(ValueError, match="duplicate method id"):
            reg.register(_make_cls("alpha"))

    def test_get_methods(self):
        reg = Registry("t")
        reg.register(_make_cls("alpha"))

        methods = reg.get_methods()
        assert [m.id for m in methods] == ["alpha"]

    async def test_invoke_async_execute(self):
        reg = Registry("t")
        cls = _make_cls("alpha")
        reg.register(cls)

        result = await reg.invoke(cls, {"a": 1}, "dep")
        assert result == {"kind": "async", "params": {"a": 1}}

    async def test_invoke_sync_execute(self):
        reg = Registry("t")
        cls = _make_cls("alpha", sync=True)
        reg.register(cls)

        result = await reg.invoke(cls, {"a": 1}, "dep")
        assert result == {"kind": "sync", "params": {"a": 1}}


class TestMissingRequired:
    def _descriptor(self) -> MethodDescriptor:
        return MethodDescriptor(
            id="x",
            parameters=[
                MethodParameter(name="a", type="string", required=True),
                MethodParameter(name="b", type="array", required=True),
                MethodParameter(name="c", type="string"),
            ],
        )

    def test_absent_and_empty_count_as_missing(self):
        d = self._descriptor()
        assert missing_required(d, {}) == ["a", "b"]
        assert missing_required(d, {"a": "", "b": []}) == ["a", "b"]
        assert missing_required(d, {"a": None, "b": None}) == ["a", "b"]

    def test_present_values_pass(self):
        d = self._descriptor()
        assert missing_required(d, {"a": "x", "b": ["y"]}) == []
        assert missing_required(d, {"a": "x", "b": ["y"], "c": ""}) == []

    def test_optional_never_missing(self):
        d = self._descriptor()
        assert missing_required(d, {"a": "x", "b": ["y"]}) == []
