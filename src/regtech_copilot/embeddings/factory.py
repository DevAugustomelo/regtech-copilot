"""Factory for creating embedding providers."""

from regtech_copilot.core.config import Settings, get_settings
from regtech_copilot.embeddings.base import EmbeddingProvider
from regtech_copilot.embeddings.ollama import OllamaEmbeddingClient


class UnsupportedEmbeddingProviderError(Exception):
    """Raised when a requested embedding provider is not implemented."""


class EmbeddingFactory:
    """Create embedding providers based on application settings."""

    @staticmethod
    def create(settings: Settings | None = None) -> EmbeddingProvider:
        """Create the configured embedding provider."""

        config = settings or get_settings()

        if config.embedding_provider == "ollama":
            return OllamaEmbeddingClient(
                base_url=config.llm_base_url,
                model=config.embedding_model,
                expected_dimensions=config.embedding_dimensions,
            )

        raise UnsupportedEmbeddingProviderError(
            f"Embedding provider '{config.embedding_provider}' is not implemented yet."
        )
