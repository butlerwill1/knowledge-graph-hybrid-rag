from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = Field(default="Knowledge Graph MVP", alias="APP_NAME")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    raw_data_dir: Path = Field(default=Path("data/raw_pdfs"), alias="RAW_DATA_DIR")
    parsed_data_dir: Path = Field(default=Path("data/parsed"), alias="PARSED_DATA_DIR")
    extracted_data_dir: Path = Field(default=Path("data/extracted"), alias="EXTRACTED_DATA_DIR")
    vector_store_path: Path = Field(default=Path("data/vector_store.json"), alias="VECTOR_STORE_PATH")
    graph_backend: str = Field(default="in_memory", alias="GRAPH_BACKEND")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="knowledgegraph", alias="NEO4J_PASSWORD")
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"),
    )
    llm_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "OPENROUTER_BASE_URL", "OPENAI_BASE_URL"),
    )
    llm_model: str = Field(
        default="openai/gpt-5",
        validation_alias=AliasChoices("LLM_MODEL", "OPENROUTER_MODEL", "OPENAI_MODEL"),
    )
    openrouter_http_referer: str | None = Field(default=None, alias="OPENROUTER_HTTP_REFERER")
    openrouter_title: str | None = Field(default=None, alias="OPENROUTER_TITLE")
    enable_llm_extraction: bool = Field(default=False, alias="ENABLE_LLM_EXTRACTION")
    llm_extraction_max_chunks: int = Field(default=25, alias="LLM_EXTRACTION_MAX_CHUNKS")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    def ensure_directories(self) -> None:
        for path in (self.raw_data_dir, self.parsed_data_dir, self.extracted_data_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def llm_extraction_ready(self) -> bool:
        return self.enable_llm_extraction and bool(self.llm_api_key)


settings = Settings()
settings.ensure_directories()
