"""Jira plugin integrations — creates test-management plugin instances by JIRA_PLUGINS."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from atlassian import Jira

from atlassian_rag_mcp.log.logger import logger

_PLUGIN_MODULES: dict[str, tuple[str, str]] = {
    "zephyr": (".zephyr.client", "ZephyrClient"),
}


def create_plugins(
    jira: Jira,
    plugins_str: str,
    base_url: str = "",
    username: str = "",
    password: str = "",
) -> dict[str, Any]:
    """Factory: instantiate test-management plugins named in plugins_str."""
    plugins: dict[str, Any] = {}
    for name in (n.strip() for n in plugins_str.split(",") if n.strip()):
        entry = _PLUGIN_MODULES.get(name)
        if entry is None:
            logger.warning(f"Unknown jira_plugin={name}, skipping")
            continue
        mod_path, cls_name = entry
        try:
            mod = import_module(mod_path, package="atlassian_rag_mcp.jira.plugins")
            cls = getattr(mod, cls_name)
            plugins[name] = cls(
                jira, base_url=base_url, username=username, password=password,
            )
            logger.info(f"Plugin '{name}' initialized ({cls_name})")
        except Exception as e:
            logger.error(f"Failed to init plugin '{name}': {e}")
    return plugins
