"""Logging — configuration and global logger."""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime


def setup_logging(level: str = "DEBUG", log_file: str | None = None) -> None:
    """Configure root logger: stdout + optional daily rotating file.

    Idempotent: handlers from a previous call are replaced, so calling it
    from both ``__main__`` and the FastAPI lifespan does not duplicate logs.
    """
    level_name = level.upper()
    level = getattr(logging, level_name, logging.DEBUG)

    log_format = "%(asctime)s [%(levelname)-7s] %(name)s:%(message)s"
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(level)
    stdout_handler._arag_managed = True

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for h in list(root_logger.handlers):
        if getattr(h, "_arag_managed", False):
            root_logger.removeHandler(h)
    root_logger.addHandler(stdout_handler)

    if log_file:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_filename = f"{log_dir}/atlassian_rag_mcp_{datetime.now().strftime('%Y-%m-%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        file_handler._arag_managed = True
        root_logger.addHandler(file_handler)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("atlassian").setLevel(logging.WARNING)


logger = logging.getLogger("atlassian_rag_mcp")
