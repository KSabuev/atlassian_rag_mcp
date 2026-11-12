"""Tests for logging setup (setup_logging)."""

from __future__ import annotations

import logging

import pytest

from atlassian_rag_mcp.log.logger import setup_logging


@pytest.fixture(autouse=True)
def _cleanup_handlers():
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_arag_managed", False):
            root.removeHandler(h)
            h.close()


def _managed_handlers():
    return [
        h for h in logging.getLogger().handlers
        if getattr(h, "_arag_managed", False)
    ]


class TestSetupLogging:
    def test_level_mapping(self):
        setup_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING

    def test_level_lowercase_accepted(self):
        setup_logging("info")
        assert logging.getLogger().level == logging.INFO

    def test_invalid_level_falls_back_to_debug(self):
        setup_logging("BOGUS")
        assert logging.getLogger().level == logging.DEBUG

    def test_stdout_handler_added(self):
        setup_logging("DEBUG")
        handlers = _managed_handlers()
        assert len(handlers) == 1
        assert isinstance(handlers[0], logging.StreamHandler)

    def test_idempotent_no_duplicate_handlers(self):
        setup_logging("DEBUG")
        setup_logging("DEBUG")
        assert len(_managed_handlers()) == 1

    def test_existing_unmanaged_handler_preserved(self):
        custom = logging.StreamHandler()
        root = logging.getLogger()
        root.addHandler(custom)
        try:
            setup_logging("DEBUG")
            handlers = _managed_handlers()
            assert len(handlers) == 1
            assert custom in root.handlers
        finally:
            root.removeHandler(custom)

    def test_file_handler_created(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        setup_logging("INFO", log_file="enabled")
        handlers = _managed_handlers()
        assert len(handlers) == 2
        file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
        assert (tmp_path / "logs").is_dir()
        log_files = list((tmp_path / "logs").glob("atlassian_rag_mcp_*.log"))
        assert len(log_files) == 1

    def test_third_party_loggers_suppressed(self):
        setup_logging("DEBUG")
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING
        assert logging.getLogger("atlassian").level == logging.WARNING
