# atlassian-rag-mcp

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![MCP](https://img.shields.io/badge/MCP-server-green)
![FastAPI](https://img.shields.io/badge/FastAPI-server-009688)
![Docker](https://img.shields.io/badge/docker-ready-2496ED)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**atlassian-rag-mcp** is an MCP (Model Context Protocol) server that gives LLM agents
hybrid RAG search over your Confluence wiki, plus read-only access to Jira and
Zephyr Scale — all behind a single HTTP endpoint.

The codebase is project-agnostic (no instance-specific data in the code) and is
designed for **medical and banking** knowledge bases — all deployment data lives
in `.env` and the benchmark data.

- **Confluence RAG** — hybrid search combining semantic vectors, FTS keyword matching
  (Tantivy) and Confluence CQL structural filters, fused with Reciprocal Rank Fusion.
- **Jira** — 35 read-only methods (issues, trees, comments, changelog, worklog,
  boards, epics, sprints, components, versions, statuses, fields, users, groups).
- **Zephyr Scale** — 24 read-only test-management methods (test cases, runs, plans,
  folders, trace links, attachments).

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Search algorithms](#search-algorithms)
- [Requirements](#requirements)
- [Installation](#installation)
  - [From source](#from-source)
  - [Docker](#docker)
- [MCP tools](#mcp-tools)
- [Connecting an MCP client](#connecting-an-mcp-client)
- [REST API](#rest-api)
- [Indexing](#indexing)
- [Development](#development)
- [License](#license)

## Features

- **Hybrid Confluence search** — three pluggable algorithms (CQL+FTS fusion, pure
  FTS, vector+FTS fusion) with per-request tuning (`top_k`, `rrf_k`).
- **Full indexing pipeline** — CQL space scan → Confluence HTML cleaning →
  heading-aware chunking (real Qwen3 token budget) → embeddings → vector store
  (Qdrant or LanceDB) + FTS index (Tantivy).
- **Attachment text extraction** — PDF, DOCX, DOC, XLSX, XLS, PPTX, RTF, EML, SVG,
  DrawIO, BPMN, HTML and plain text attachments are downloaded, classified and
  indexed alongside page content.
- **Page structure as Markdown** — full or scoped page tree (children/parents) from
  indexed metadata.
- **Stale-page handling** — version comparison against Confluence, re-index queue
  with retries, dead-letter sweep and periodic persistence.
- **Read-only Jira & Zephyr Scale** — exposed as self-describing method registries
  (`*_get_methods` / `*_execute_method`), so agents discover schemas at runtime.
- **REST API** — index sync / reconcile / rebuild endpoints with Swagger UI and
  optional `X-API-Key` protection.
- **Health checks** — one tool reports VectorDB, Confluence, Jira, plugins and
  embedding service status.
- **Docker** — multi-stage Dockerfile and a reference docker-compose deployment
  with Qdrant.

## Architecture

```
┌──────────────┐  pages + attachments    ┌───────────────────────────────────────────┐
│  Confluence  │ ───────────────────────► │  atlassian-rag-mcp (FastAPI + uvicorn)    │
└──────────────┘                          │                                           │
┌──────────────┐  read-only               │  indexer: CQL scan → HTML clean →         │
│     Jira     │ ───────────────────────► │  chunker → embeddings ──► VectorDB       │
└──────────────┘      (35 methods)        │  (Qdrant / LanceDB)  +  Tantivy (FTS)    │
┌──────────────┐  read-only               │                                           │
│   Zephyr     │ ───────────────────────► │  agents:  /mcp      – MCP (streamable HTTP)
└──────────────┘  (Jira plugin, 24 meth.) │            /api/v1  – REST index API      │
                                           │            /health  – liveness probe      │
                                           │            /docs    – Swagger UI         │
                                           └───────────────────────────────────────────┘
```

The server boots a shared `ServerContext` (Confluence client, Jira client, vector
store, FTS index, embedding provider, index queue) and runs N async indexer
workers (`EMBEDDING_MAX_WORKERS`) that pull pages from a persistent queue, chunk
them, embed them and upsert into the vector store + FTS index.

## Search algorithms

Confluence search is exposed through a single tool — `confluence_execute_algorithm` —
with a registry of algorithms. Call `confluence_get_algorithms` for the exact
parameter schema of each.

| Algorithm | Branches | Returns | Best for |
|---|---|---|---|
| `cql_fts_fusion` | Confluence CQL + FTS (RRF) | up to `top_k` **pages** with a plain-text `excerpt` | queries mixing identifiers with natural language, or needing structural filters (title / label / author / date / space). Overall leader — prefer it when in doubt |
| `fulltext_search` | FTS (Tantivy) only | up to `top_k` **chunks** with full text | pure lexical queries: acronyms, short domain words, exact keywords |
| `vector_fts_fusion` | Semantic vectors + FTS (RRF) | up to `top_k` **chunks** with full text | conceptual natural-language questions, no filters (safe default) |

Fusion notes:

- Results are merged with **Reciprocal Rank Fusion** (`RRF_K = 40`); each branch
  builds a `top_k × 10` candidate pool before the final cut.
- `score` is per-algorithm and **not comparable** across algorithms. An out-of-domain
  query returns low-scoring noise rather than an empty list — treat a low score as
  "not found" and switch algorithm.
- Every algorithm accepts an optional `search_params` object with per-request
  overrides: `{"top_k": int, "rrf_k": int}`.
- CQL quirks are documented in the algorithm description: multi-token `text ~ "..."`
  phrases use AND semantics, so use one distinctive term; `title` is a fuzzy
  substring match; ISO dates only.


## Requirements

- **Python** ≥ 3.10 (3.12 recommended).
- **Confluence** Server / Data Center with the REST API enabled.
- **Jira** Server / Data Center (read-only API access). Zephyr Scale plugin is
  optional — controlled by `JIRA_PLUGINS`.
- **Embedding endpoint** — any OpenAI-compatible `/v1/embeddings` server
  (vLLM, TEI, OpenAI, …) or Ollama (`EMBEDDING_BINDING=ollama`).
- **Vector store** — LanceDB (embedded, default) or Qdrant (`VECTOR_DB=qdrant`).
- ~8 GB of RAM is comfortable for the embedded stack; more for large spaces.

## Installation

### From source

```bash
git clone https://github.com/KSabuev/confluens_to_lightrag.git atlassian_rag_mcp
cd atlassian_rag_mcp

python -m venv .venv
source .venv/bin/activate

pip install .            # or: pip install -e ".[dev]" for development
```

Create a `.env` file with your Confluence/Jira/embedding credentials (all variables
are documented in [`config/settings.py`](../https:/github.com/KSabuev/atlassian_rag_mcp/src/atlassian_rag_mcp/config/settings.py)),
then start the server:

```bash
atlassian-rag-mcp                # console script
# or
python -m atlassian_rag_mcp
```

On startup you'll see:

```
MCP endpoint:  http://127.0.0.1:8000/mcp
REST API:      http://127.0.0.1:8000/api/v1/index
Swagger:       http://127.0.0.1:8000/docs
```

Then trigger an initial index sync:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/index/sync"
```

### Docker

> Note: the Dockerfile installs heavy native dependencies (lancedb, tantivy,
> simsimd, tokenizers, …) from pinned wheels in `build/wheels/`. Make sure that
> directory is populated with the pinned wheels before building, or adjust the
> builder stage to a regular `pip install`.

```bash
docker build -t atlassian-rag-mcp .
docker run --env-file .env \
  -p 8000:8000 \
  -v rag-data:/app/data \
  atlassian-rag-mcp
```

A reference deployment with Qdrant is provided in
[`docker-compose.GPU.yml`](docker-compose.GPU.yml) (Qdrant + service, tuned for a
GPU-hosted embedding endpoint — review and adapt the host-specific settings before
use).


## MCP tools

Nine tools are exposed over the MCP endpoint (`/mcp/`):

| Tool | Description |
|---|---|
| `health_check` | Health of all backends: VectorDB, Confluence, Jira, Jira plugins, embedding service |
| `confluence_get_algorithms` | List available Confluence search algorithms with full parameter schemas |
| `confluence_execute_algorithm` | Execute a search algorithm by ID (see [Search algorithms](#search-algorithms)) |
| `confluence_get_structure` | Page tree as Markdown — full tree, or scoped subtree (`children` / `parents`) |
| `confluence_get_page_content` | Any part of a page: `info` (metadata), `html` (cleaned live HTML), `chunks` (indexed chunks, filterable by block/chunk index), `table` (all rows of one block), `files` (attachment list with download links), `attachments` (attachment text extraction) |
| `jira_get_methods` | List all 35 Jira methods with descriptions and parameter schemas |
| `jira_execute_method` | Execute a Jira method by ID (read-only: issue data & context, search, counts, project metadata, users & groups) |
| `zephyr_get_methods` | List all 24 Zephyr Scale methods with descriptions and parameter schemas |
| `zephyr_execute_method` | Execute a Zephyr Scale method by ID (read-only: test cases, runs, plans, folders, trace links, attachments) |

The `*_get_methods` / `*_get_algorithms` tools make the server **self-describing**:
an agent discovers exact names, types and required flags at runtime, so new
methods/algorithms are picked up automatically.

## Connecting an MCP client

The server speaks **streamable HTTP** MCP at `/mcp/`.

Claude Desktop / any URL-based client:

```json
{
  "mcpServers": {
    "atlassian-rag-mcp": {
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

opencode (`.opencode/opencode.json`):

```json
{
  "mcp": {
    "atlassian-rag-mcp": {
      "type": "url",
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

## REST API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe (`{"status": "ok"}`) |
| `POST`/`GET`/`DELETE` | `/mcp/` | MCP streamable-HTTP endpoint |
| `POST` | `/api/v1/index/sync` | The single index-sync method. CQL scan → version compare → enqueue stale/missing pages. JSON body, all flags optional: `force` — re-enqueue every scanned page (full re-index, use after a chunker change); `spaces` — restrict to these space keys (omitted/empty = configured spaces, `CONFLUENCE_SPACE_KEY`); `delete_stale` — delete orphans (pages removed from Confluence) within the scanned scope; `deep` — per-page verification against live Confluence (chunk presence + version) for every indexed page in scope, skipped when `force` is set, costs one `get_page` per indexed page; `fts_rebuild` — rebuild the Tantivy FTS index from stored VectorDB payloads of the target spaces (no re-embedding, VectorDB and Confluence untouched), scope follows `spaces`, requires at least one space, runs after the scan so orphan deletions are already reflected |
| `POST` | `/api/v1/index/page/{page_id}` | Enqueue a single page for re-indexing |
| `DELETE` | `/api/v1/index/page/{page_id}` | Remove a page from the vector store and FTS index |
| `GET` | `/api/v1/index/status` | Index queue status (pending, processed, errors, …) |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/openapi.json` | OpenAPI schema |

Examples:

```bash
# Incremental sync (all configured spaces)
curl -X POST "http://localhost:8000/api/v1/index/sync" \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{}'

# Forced re-index of specific spaces
curl -X POST "http://localhost:8000/api/v1/index/sync" \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"force": true, "spaces": ["PROJ"]}'

# Sync + orphan cleanup + full index audit
curl -X POST "http://localhost:8000/api/v1/index/sync" \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"delete_stale": true, "deep": true}'

# Rebuild the FTS index from stored payloads (no re-embedding)
curl -X POST "http://localhost:8000/api/v1/index/sync" \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"fts_rebuild": true, "spaces": ["PROJ"]}'

# Queue status
curl "http://localhost:8000/api/v1/index/status" -H "X-API-Key: $API_KEY"
```

When `API_KEY` is set, every `/api/v1/*` call requires the `X-API-Key` header.

## Indexing

1. **Discovery** — pages are discovered via CQL (`type=page AND space.key IN …`),
   batched by `CONFLUENCE_CQL_LIMIT`.
2. **Versioning** — each page's Confluence version number is compared against the
   stored payload; unchanged pages are skipped. `force=true` re-indexes everything
   (use after a chunker change).
3. **Queue** — stale/missing page IDs go into a persistent async queue with
   retries (`INDEX_MAX_RETRIES`) and a dead-letter sweep.
4. **Workers** — N workers fetch page content + attachments, clean the HTML,
   chunk it (sections and tables, heading hierarchy, token budget), extract
   attachment text, embed the chunks and upsert into the vector store + FTS index.
5. **Maintenance** — `POST /index/sync` with `{"delete_stale": true}` removes
   pages deleted in Confluence (orphans) and re-queues stale ones;
   `{"fts_rebuild": true}` rebuilds the FTS index from stored VectorDB
   payloads without re-embedding (scoped to `spaces`).

## Development

```bash
pip install -e ".[dev]"     # includes pytest, ruff

pytest                      # run the test suite (48 test files)
pytest tests/test_chunker.py -v   # single file
ruff check .                # lint (E, F, I, W; line length 120)
```

## License

This project is licensed under the **MIT License**.
