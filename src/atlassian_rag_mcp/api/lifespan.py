"""Resource initialization and cached factory.

Clients are created once via _init() and shared across all
MCP sessions and REST API handlers via an internal cache.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from atlassian_rag_mcp.config import settings
from atlassian_rag_mcp.confluence import ConfluenceClient, create_confluence_client
from atlassian_rag_mcp.confluence.search import SearchDeps
from atlassian_rag_mcp.confluence.services.check_relevant import check_relevant
from atlassian_rag_mcp.confluence.services.indexer import IndexQueue
from atlassian_rag_mcp.confluence.services.reconciler import reconcile
from atlassian_rag_mcp.confluence.services.vector_service import VectorService
from atlassian_rag_mcp.embedding import create_embedding_client
from atlassian_rag_mcp.embedding.base import EmbeddingProvider
from atlassian_rag_mcp.exceptions import BaseError
from atlassian_rag_mcp.jira import JiraClient, create_jira_client
from atlassian_rag_mcp.jira.methods import fetch_jira_value_maps
from atlassian_rag_mcp.jira.plugins.zephyr import ZephyrClient
from atlassian_rag_mcp.log.logger import logger
from atlassian_rag_mcp.storage.base import VectorDB
from atlassian_rag_mcp.storage.factory import create_fts_index, create_vector_db

_ctx: ServerContext | None = None

_INJECT_TIMEOUT = 20.0


async def _await_with_timeout(awaitable: Any, timeout: float = _INJECT_TIMEOUT) -> Any:
    """Await a coroutine with a hard timeout so a black-holed dependency host
    (unreachable Confluence/Jira/embedding) can never block startup forever."""
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError:
        raise BaseError(f"Timed out after {timeout}s") from None


def _apply_preamble_to_descriptions(
    tool_names: tuple[str, ...],
    preamble: str,
) -> list[str]:
    """Prepend `preamble` (a header block) to MCP tool descriptions by name.

    Updates both the function docstring and the registered Tool object — the
    MCP server resolves tool descriptions at decoration time, so mutating the
    docstring alone is not visible to MCP clients. Returns the names actually
    updated.

    Imports api.tools lazily to avoid a circular import (api.tools imports lifespan).
    """
    import atlassian_rag_mcp.api.tools as tools
    from atlassian_rag_mcp.api.mcp_server import mcp as mcp_server

    updated: list[str] = []
    for func_name in tool_names:
        func = getattr(tools, func_name, None)
        if func is None:
            continue
        old = func.__doc__ or ""
        func.__doc__ = preamble + old
        tool = mcp_server._tool_manager.get_tool(func_name)
        if tool is not None:
            tool.description = preamble + (tool.description or "")
        updated.append(func_name)
    return updated


@dataclass
class ServerContext:
    """Container for all server-level resources.

    ``relevant_filter``: async ``(list[dict]) -> list[dict]``, the relevance
    pre-filter for search results. ``reconcile_fn``: async
    ``(space_key, deep, force, delete_stale) -> dict`` — the unified index sync.
    """

    confluence_client: ConfluenceClient
    jira_client: JiraClient
    vector_db: VectorDB
    vector_service: VectorService
    embedding: EmbeddingProvider
    index_queue: IndexQueue
    search_deps: SearchDeps
    relevant_filter: Any
    reconcile_fn: Any

    async def aclose(self) -> None:
        """Close all resources in reverse creation order.

        Safe to call multiple times — missing/None attributes are skipped.
        """
        errors: list[tuple[str, Exception]] = []

        async def _safe(name: str, closer: Callable[[], Any]) -> Any:
            try:
                return await closer()
            except Exception as e:
                errors.append((name, e))
                return None

        await _safe("index_queue.flush_to_disk", self.index_queue.flush_to_disk)

        for pname, plugin in self.jira_client.plugins.items():
            aclose = getattr(plugin, "aclose", None)
            if aclose is not None:
                await _safe(f"plugin.{pname}", aclose)

        await _safe("embedding", self.embedding.close)
        await _safe("fts_index", self.search_deps.fts_index.close)
        await _safe("vector_db", self.vector_db.close)

        if errors:
            for name, exc in errors:
                logger.warning(f"Resource close error [{name}]: {exc}")


async def _inject_metadata(
    domain: str,
    tool_name: str,
    sections: list[tuple[str, Callable[[], Any]]],
) -> None:
    """Fetch per-domain metadata sections and inject them into a tool description.

    ``sections`` is a list of ``(label, fetch)``: ``fetch`` is a zero-arg
    callable (sync runs in a thread) returning a formatted line or ``None``.
    Each section fails independently — a warning is logged and the remaining
    sections still apply, so a transient backend failure never blocks startup.

    Shared scaffold for the per-domain injectors below (Confluence / Jira /
    Zephyr); adding a new domain is a list of section factories, not a copy.
    """
    lines: list[str] = []
    for label, fetch in sections:
        try:
            text = await _await_with_timeout(
                asyncio.to_thread(fetch) if not asyncio.iscoroutinefunction(fetch) else fetch()
            )
            if text:
                lines.extend(str(text).splitlines())
        except Exception as e:
            logger.warning(f"Could not fetch {domain} {label}: {e}")
    if not lines:
        return
    preamble = "\n".join(lines) + "\n"
    updated = _apply_preamble_to_descriptions((tool_name,), preamble)
    logger.info(
        f"Injected {domain} metadata ({len(lines)} sections) into {len(updated)} tool descriptions"
    )


async def _inject_confluence_metadata(
    confluence_client: Any,
    space_keys: list[str] | None = None,
) -> None:
    """Fetch Confluence metadata (spaces) and inject into tool docstring."""
    space_keys = space_keys or []

    async def _all_spaces() -> str | None:
        spaces_data = await confluence_client.get_all_spaces(limit=settings.confluence_spaces_limit)
        if not spaces_data.spaces:
            return None
        space_list = [f"{s.get('key')} ({s.get('name', '')})" for s in spaces_data.spaces]
        return "all spaces: " + ", ".join(space_list)

    async def _configured_space(key: str) -> str | None:
        space_info = await confluence_client.get_space(key, expand="description,homepage")
        name = space_info.name
        return f"{key} ({name})" if name else key

    sections: list[tuple[str, Callable[[], Any]]] = [("all spaces", _all_spaces)]
    sections.extend(
        (f"space info [{key}]", lambda key=key: _configured_space(key)) for key in space_keys
    )
    await _inject_metadata("Confluence", "confluence_execute_algorithm", sections)


def build_embedding_client(s: Any = None) -> Any:
    """Create the embedding client from settings (shared by app and tests)."""
    from atlassian_rag_mcp.config import settings as _s
    s = s or _s
    return create_embedding_client(
        url=s.embedding_url,
        api_key=s.embedding_api_key or "",
        model=s.embedding_model,
        dim=s.embedding_dim,
        binding=s.embedding_binding,
        http_timeout=s.embedding_timeout,
    )


async def _init() -> ServerContext:
    """Create client once, validate service, return singleton context."""
    global _ctx
    if _ctx is not None:
        return _ctx

    confluence_client = create_confluence_client(
        url=settings.confluence_url,
        username=settings.confluence_username,
        password=settings.confluence_password,
        timeout=settings.confluence_timeout,
    )
    jira_keys = settings.jira_project_keys
    jira_client = create_jira_client(
        url=settings.jira_url,
        username=settings.jira_username,
        password=settings.jira_password,
        timeout=settings.jira_timeout,
        project_key=jira_keys[0] if jira_keys else "",
        plugins=settings.jira_plugins,
    )
    vector_db = await create_vector_db(
        backend=settings.vector_db,
        qdrant_host=settings.qdrant_host or "",
        qdrant_port=settings.qdrant_port,
        qdrant_api_key=settings.qdrant_api_key or "",
        qdrant_collection=settings.qdrant_collection,
        lancedb_path=settings.working_dir + "/lancedb",
        vector_size=settings.embedding_dim,
        upsert_batch_size=settings.upsert_batch_size,
    )
    embedding_client = build_embedding_client(settings)
    fts_index = await create_fts_index(
        index_dir=settings.working_dir + "/fts_index",
    )
    vector_service = VectorService(
        vector_db=vector_db,
        embedding_client=embedding_client,
        fts_index=fts_index,
        batch_size=settings.embedding_batch_size,
        scroll_limit=settings.scroll_limit,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    index_queue = IndexQueue(
        settings.working_dir,
        persist_interval=settings.persist_interval,
        max_stored_errors=settings.max_stored_errors,
        max_retries=settings.index_max_retries,
    )

    async def _relevant_filter(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await check_relevant(results, confluence_client, index_queue, batch_size=settings.stale_check_batch_size)

    _ctx = ServerContext(
        confluence_client=confluence_client,
        jira_client=jira_client,
        vector_db=vector_db,
        vector_service=vector_service,
        embedding=embedding_client,
        index_queue=index_queue,
        relevant_filter=_relevant_filter,
        reconcile_fn=lambda space_key=None, deep=False, force=False, delete_stale=True: reconcile(
            confluence_client=confluence_client,
            vector_service=vector_service,
            index_queue=index_queue,
            space_key=space_key or settings.confluence_space_key or "",
            cql_batch_size=settings.confluence_cql_limit,
            deep=deep,
            force=force,
            delete_stale=delete_stale,
        ),
        search_deps=SearchDeps(
            vector_db=vector_db,
            embedding_client=embedding_client,
            confluence_client=confluence_client,
            fts_index=fts_index,
            top_k=settings.top_k,
            confluence_space_key=settings.confluence_space_key or "",
            confluence_url=settings.confluence_url or "",
            get_in_work_versions=index_queue.get_in_work_versions,
        ),
    )

    try:
        await vector_service._ensure_collection()
    except BaseError as e:
        logger.error(f"VectorDB collection init FAILED: {e}")
        raise

    await _inject_confluence_metadata(confluence_client, space_keys=settings.confluence_space_keys)
    await _inject_jira_metadata(jira_client, project_keys=jira_keys or None)
    await _inject_zephyr_metadata(jira_client, project_key=jira_keys[0] if jira_keys else None)

    conf_url = settings.confluence_url
    jira_url = settings.jira_url
    checks = [
        ("Confluence", True, lambda: confluence_client.validate(url=conf_url), False),
        ("Jira", True, lambda: jira_client.validate(url=jira_url), False),
        ("Embedding", False, lambda: embedding_client.validate(), False),
        ("VectorDB", False, lambda: vector_db.validate(), False),
        ("FTSIndex", False, lambda: fts_index.validate(), True),
    ]
    for name, is_sync, method, fatal in checks:
        try:
            if is_sync:
                await _await_with_timeout(asyncio.to_thread(method))
            else:
                await _await_with_timeout(method())
        except BaseError as e:
            logger.error(f"    {name:<20s}  FAILED: {e}")
            if fatal:
                raise
        else:
            logger.info(f"    {name:<20s}  OK")

    return _ctx


async def _inject_jira_metadata(jira_client: Any, project_keys: list[str] | None = None) -> None:
    """Sync SDK calls wrapped via fetch_jira_value_maps in a thread."""
    if not project_keys:
        return

    def _value_maps() -> str | None:
        raw = fetch_jira_value_maps(jira_client, project_keys)
        if not raw.get("projects") and not raw.get("priorities") and not raw.get("resolutions"):
            return None
        lines: list[str] = []
        for key, values in raw["projects"].items():
            types = values.get("types") or []
            if types:
                names = [f"{name} ({pid})" for name, pid in types]
                lines.append(f"issue types ({key}): {', '.join(names)}  (use numeric ID in JQL)")
        priorities = raw.get("priorities") or []
        if priorities:
            names = [f"{name} ({pid})" for name, pid in priorities]
            lines.append(f"priorities: {', '.join(names)}  (use numeric ID in JQL)")
        resolutions = raw.get("resolutions") or []
        if resolutions:
            names = [f"{name} ({pid})" for name, pid in resolutions]
            lines.append(f"resolutions: {', '.join(names)}  (use numeric ID in JQL)")
        for key, values in raw["projects"].items():
            statuses = values.get("statuses") or []
            if statuses:
                names = [f"{name} ({pid})" for name, pid in statuses]
                lines.append(f"statuses ({key}): {', '.join(names)}  (use numeric ID in JQL)")
        return "\n".join(lines) or None

    await _inject_metadata("Jira", "jira_execute_method", [("value maps", _value_maps)])


async def _inject_zephyr_metadata(jira_client: Any, project_key: str | None = None) -> None:
    """Inject Zephyr project metadata (statuses, priorities) into tool descriptions.

    A transient failure (DNS down etc.) must not crash the startup: the
    injection is skipped and availability is re-checked lazily later.
    """
    if not project_key:
        return

    plugin = (getattr(jira_client, "plugins", None) or {}).get("zephyr")
    if not isinstance(plugin, ZephyrClient):
        return
    try:
        available = await _await_with_timeout(plugin.is_available())
    except Exception as e:
        logger.warning(f"Zephyr availability check failed: {e} — skipping metadata injection")
        return
    if not available:
        return

    async def _project() -> str | None:
        projects = await plugin.get_projects()
        proj = next((p for p in projects if p.get("key") == project_key), None)
        if not proj:
            return None
        return f"configured Zephyr project: {project_key} (id {proj.get('id')})"

    async def _statuses() -> str | None:
        statuses = await plugin.get_test_case_statuses(project_key)
        if not statuses:
            return None
        names = [f"{s.get('name')} ({s.get('id')})" for s in statuses]
        return f"test case statuses: {', '.join(names)}"

    async def _priorities() -> str | None:
        priorities = await plugin.get_test_case_priorities(project_key)
        if not priorities:
            return None
        names = [f"{p.get('name')} ({p.get('id')})" for p in priorities]
        return f"test case priorities: {', '.join(names)}"

    async def _result_statuses() -> str | None:
        result_statuses = await plugin.get_test_result_statuses(project_key)
        if not result_statuses:
            return None
        names = [f"{s.get('name')} ({s.get('id')})" for s in result_statuses]
        return f"test result statuses: {', '.join(names)}"

    sections: list[tuple[str, Callable[[], Any]]] = [
        ("project", _project),
        ("test case statuses", _statuses),
        ("test case priorities", _priorities),
        ("test result statuses", _result_statuses),
    ]
    await _inject_metadata("Zephyr", "zephyr_execute_method", sections)


def reset_ctx() -> None:
    """Reset the cached context (for tests)."""
    global _ctx
    _ctx = None
