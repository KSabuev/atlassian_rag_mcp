"""Structure service — builds page tree from vector DB metadata (async)."""

from __future__ import annotations

from typing import Any

from atlassian_rag_mcp.confluence.models import StructureNode
from atlassian_rag_mcp.storage.base import VectorDB


class StructureService:
    """Build Confluence page structure from vector DB payload metadata.

    Scans all indexed chunks, extracts unique {page_id, parent_id, title},
    builds a parent→children tree, and renders it as Markdown.
    """

    def __init__(self, vector_db: VectorDB, space_key: str = "") -> None:
        self.vector_db = vector_db
        self.space_key = space_key
        keys = [k.strip() for k in space_key.split(",") if k.strip()]
        self._space_label = ", ".join(keys) if keys else "all"

    async def get_all_pages(self) -> list[dict[str, Any]]:
        """Scroll all points with pagination and return deduplicated page list."""
        seen: dict[str, dict[str, Any]] = {}
        offset: str | None = None
        page_size = 1000
        while True:
            points, next_offset = await self.vector_db.scroll_points(
                filter_condition={}, limit=page_size, offset=offset
            )
            if not points:
                break
            for p in points:
                pay = p.get("payload", {})
                pid = pay.get("page_id")
                if pid and pid not in seen:
                    seen[pid] = StructureNode(
                        page_id=pid,
                        parent_id=pay.get("parent_id"),
                        title=pay.get("page_title", ""),
                    ).model_dump()
            if not next_offset:
                break
            offset = next_offset
        return list(seen.values())

    async def render_tree(
        self,
        page_id: str | None = None,
        children: bool = False,
        parents: bool = False,
    ) -> str:
        """Render structure as markdown."""
        pages = await self.get_all_pages()
        if not pages:
            return "No indexed pages found in vector store."

        children_map: dict[str | None, list[dict]] = {}
        for p in pages:
            parent = p.get("parent_id") or None
            children_map.setdefault(parent, []).append(p)
        for k in children_map:
            children_map[k].sort(key=lambda x: x["title"].lower())

        if page_id is None:
            return self._render_full_tree(pages, children_map)

        page_map = {p["page_id"]: p for p in pages}
        if page_id not in page_map:
            return f"Page {page_id} not found in vector store"

        page = page_map[page_id]

        if not children and not parents:
            indent = "  " * self._compute_depth(page_id, page_map)
            hashes = "#" * (self._compute_depth(page_id, page_map) + 1)
            return f"{indent}{hashes} {page['title']} ({page['page_id']})"

        if children and not parents:
            descendants = self._get_descendants(page_id, children_map)
            subset = [page] + descendants
            return self._render_subset(subset, children_map)

        if parents and not children:
            ancestor_chain = self._get_ancestors(page_id, page_map)
            subset = ancestor_chain + [page]
            return self._render_subset(subset, children_map)

        ancestor_chain = self._get_ancestors(page_id, page_map)
        descendants = self._get_descendants(page_id, children_map)
        subset = ancestor_chain + [page] + descendants
        return self._render_subset(subset, children_map)

    def _render_full_tree(self, pages, children_map) -> str:
        lines = [f"# Confluence Structure — {self._space_label}"]
        seen: set[str] = set()

        def render(parent_id: str | None, depth: int = 0) -> None:
            for node in children_map.get(parent_id, []):
                pid = node["page_id"]
                if pid in seen:
                    continue
                seen.add(pid)
                indent = "  " * depth
                hashes = "#" * min(depth + 1, 6)
                lines.append(f"{indent}{hashes} {node['title']} ({pid})")
                render(pid, depth + 1)

        render(None)
        return "\n".join(lines)

    def _render_subset(self, subset, children_map) -> str:
        lines = [f"# Confluence Structure — {self._space_label}"]
        seen: set[str] = set()
        subset_ids = {p["page_id"] for p in subset}

        roots = [p for p in subset if p.get("parent_id") not in subset_ids]

        def render(parent_id: str | None, depth: int = 0) -> None:
            for node in children_map.get(parent_id, []):
                if node["page_id"] not in subset_ids:
                    continue
                pid = node["page_id"]
                if pid in seen:
                    continue
                seen.add(pid)
                indent = "  " * depth
                hashes = "#" * min(depth + 1, 6)
                lines.append(f"{indent}{hashes} {node['title']} ({pid})")
                render(pid, depth + 1)

        for root in roots:
            render(root.get("parent_id") or None)
        return "\n".join(lines)

    @staticmethod
    def _get_descendants(page_id: str, children_map) -> list[dict]:
        result = []
        stack = list(children_map.get(page_id, []))
        while stack:
            node = stack.pop(0)
            result.append(node)
            stack.extend(children_map.get(node["page_id"], []))
        return result

    @staticmethod
    def _get_ancestors(page_id: str, page_map) -> list[dict]:
        chain = []
        current = page_map.get(page_id)
        while current and (current.get("parent_id") or None):
            parent = page_map.get(current["parent_id"] or None)
            if parent:
                chain.insert(0, parent)
                current = parent
            else:
                break
        return chain

    @staticmethod
    def _compute_depth(page_id: str, page_map) -> int:
        depth = 0
        current = page_map.get(page_id)
        while current and (current.get("parent_id") or None):
            depth += 1
            current = page_map.get(current["parent_id"] or None)
        return depth
