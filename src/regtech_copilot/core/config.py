"""Centralized application settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "RegTech Copilot"
    app_env: Literal["development", "test", "production"] = "development"
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)

    streamlit_host: str = "0.0.0.0"
    streamlit_port: int = Field(default=8501, ge=1, le=65535)
    api_base_url: str = "http://localhost:8000"

    llm_provider: Literal["ollama", "nvidia", "huggingface"] = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_model: str
    nvidia_api_key: SecretStr | None = None
    huggingface_api_key: SecretStr | None = None

    embedding_provider: Literal["ollama", "nvidia", "huggingface"] = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int | None = Field(default=None, gt=0)

    vector_store_path: str = "data/vector_store"
    documents_path: str = "data/documents"

    chunk_size_tokens: int = Field(default=700, gt=0)
    chunk_overlap_tokens: int = Field(default=100, ge=0)
    retrieval_top_k: int = Field(default=5, gt=0)

    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    llm_max_tokens: int = Field(default=1200, gt=0)

    max_upload_size_mb: int = Field(default=20, gt=0)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        """Ensure chunk overlap is smaller than chunk size."""
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "CHUNK_OVERLAP_TOKENS must be smaller than CHUNK_SIZE_TOKENS."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""
    return Settings()
