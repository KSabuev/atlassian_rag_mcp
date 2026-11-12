"""Unified configuration via pydantic-settings.

Single source of truth for all environment variables.
Replaces: config.py (dataclass), confluence/config.py, and scattered dotenv_values() calls.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env / environment variables."""

    confluence_url: str = Field(validation_alias="CONFLUENCE_URL")
    confluence_username: str = Field(validation_alias="CONFLUENCE_USERNAME")
    confluence_password: str = Field(validation_alias="CONFLUENCE_PASSWORD")
    confluence_space_key: str = Field(default="PROJ", validation_alias="CONFLUENCE_SPACE_KEY")
    confluence_timeout: int = Field(default=30, validation_alias="CONFLUENCE_TIMEOUT")

    @property
    def confluence_space_keys(self) -> list[str]:
        return [k.strip() for k in self.confluence_space_key.split(",") if k.strip()]

    jira_url: str = Field(validation_alias="JIRA_URL")
    jira_username: str = Field(validation_alias="JIRA_USERNAME")
    jira_password: str = Field(validation_alias="JIRA_PASSWORD")
    jira_timeout: int = Field(default=30, validation_alias="JIRA_TIMEOUT")
    jira_plugins: str = Field(default="zephyr", validation_alias="JIRA_PLUGINS")
    jira_project_key: str = Field(default="PROJ", validation_alias="JIRA_PROJECT_KEYS")

    @property
    def jira_project_keys(self) -> list[str]:
        return [k.strip() for k in self.jira_project_key.split(",") if k.strip()]

    embedding_binding: str = Field(default="openai_compat", validation_alias="EMBEDDING_BINDING")
    embedding_url: str = Field(validation_alias="EMBEDDING_URL")
    embedding_api_key: str | None = Field(default=None, validation_alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(validation_alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=2560, validation_alias="EMBEDDING_DIM")
    embedding_batch_size: int = Field(default=50, validation_alias="EMBEDDING_BATCH_SIZE")

    chunk_max_tokens: int = Field(default=8000, validation_alias="CHUNK_MAX_TOKENS")
    chunk_overlap_tokens: int = Field(default=200, validation_alias="CHUNK_OVERLAP_TOKENS")

    attachment_max_size_mb: int = Field(default=20, validation_alias="ATTACHMENT_MAX_SIZE_MB")
    attachment_text_max_chars: int = Field(default=20000, validation_alias="ATTACHMENT_TEXT_MAX_CHARS")
    attachment_max_files: int = Field(default=100, validation_alias="ATTACHMENT_MAX_FILES")

    vector_db: str = Field(default="lancedb", validation_alias="VECTOR_DB")

    working_dir: str = Field(default="./data", validation_alias="WORKING_DIR")

    qdrant_host: str | None = Field(default=None, validation_alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, validation_alias="QDRANT_PORT")
    qdrant_collection: str = Field(default="atlassian_chunks", validation_alias="QDRANT_COLLECTION")
    qdrant_api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_file: bool = Field(default=False, validation_alias="LOG_FILE")

    api_key: str | None = Field(default=None, validation_alias="API_KEY")

    host: str = Field(default="127.0.0.1", validation_alias="HOST")
    port: int = Field(default=8000, validation_alias="PORT")

    top_k: int = Field(default=20, validation_alias="TOP_K")
    max_response_bytes: int = Field(default=900_000, validation_alias="MAX_RESPONSE_BYTES")
    scroll_limit: int = Field(default=10000, validation_alias="SCROLL_LIMIT")
    upsert_batch_size: int = Field(default=100, validation_alias="UPSERT_BATCH_SIZE")
    embedding_timeout: int = Field(default=60, validation_alias="EMBEDDING_TIMEOUT")
    persist_interval: float = Field(default=5.0, validation_alias="PERSIST_INTERVAL")
    worker_idle_sleep: float = Field(default=5.0, validation_alias="WORKER_IDLE_SLEEP")
    confluence_cql_limit: int = Field(default=50, validation_alias="CONFLUENCE_CQL_LIMIT")
    confluence_spaces_limit: int = Field(default=100, validation_alias="CONFLUENCE_SPACES_LIMIT")
    stale_check_batch_size: int = Field(default=10, validation_alias="STALE_CHECK_BATCH_SIZE")
    max_stored_errors: int = Field(default=100, validation_alias="MAX_STORED_ERRORS")
    index_max_retries: int = Field(default=3, validation_alias="INDEX_MAX_RETRIES")
    embedding_max_workers: int = Field(default=1, validation_alias="EMBEDDING_MAX_WORKERS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


settings = Settings()
