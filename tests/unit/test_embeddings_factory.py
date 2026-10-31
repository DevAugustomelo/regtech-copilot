"""Tests for the embedding provider factory."""

import pytest

from regtech_copilot.core.config import Settings
from regtech_copilot.embeddings.factory import (
    EmbeddingFactory,
    UnsupportedEmbeddingProviderError,
)
from regtech_copilot.embeddings.ollama import OllamaEmbeddingClient


def create_settings(**overrides: object) -> Settings:
    """Create test settings with configurable values."""

    return Settings(
        _env_file=None,
        llm_model="test-llm",
        **overrides,
    )


def test_factory_creates_ollama_embedding_client() -> None:
    settings = create_settings(
        embedding_provider="ollama",
        embedding_model="test-embedding-model",
        embedding_dimensions=768,
    )

    provider = EmbeddingFactory.create(settings)

    assert isinstance(provider, OllamaEmbeddingClient)
    assert provider.model == "test-embedding-model"
    assert provider.expected_dimensions == 768
    assert provider.base_url == "http://localhost:11434"


def test_factory_creates_ollama_without_expected_dimensions() -> None:
    settings = create_settings(
        embedding_provider="ollama",
        embedding_dimensions=None,
    )

    provider = EmbeddingFactory.create(settings)

    assert isinstance(provider, OllamaEmbeddingClient)
    assert provider.expected_dimensions is None


def test_factory_rejects_unsupported_provider() -> None:
    settings = create_settings(
        embedding_provider="nvidia",
    )

    with pytest.raises(
        UnsupportedEmbeddingProviderError,
        match="nvidia",
    ):
        EmbeddingFactory.create(settings)


def test_factory_uses_application_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = create_settings(
        embedding_provider="ollama",
        embedding_model="test-embedding-model",
    )

    monkeypatch.setattr(
        "regtech_copilot.embeddings.factory.get_settings",
        lambda: settings,
    )

    provider = EmbeddingFactory.create()

    assert isinstance(provider, OllamaEmbeddingClient)
    assert provider.model == "test-embedding-model"
