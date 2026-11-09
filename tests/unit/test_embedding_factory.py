"""Tests for the embedding provider factory."""

import pytest

from regtech_copilot.core.config import Settings
from regtech_copilot.embeddings.factory import EmbeddingFactory
from regtech_copilot.embeddings.ollama import OllamaEmbeddingClient


@pytest.mark.asyncio
async def test_factory_uses_embedding_base_url() -> None:
    settings = Settings(
        _env_file=None,
        llm_base_url="https://integrate.api.nvidia.com/v1",
        llm_model="test-model",
        embedding_base_url="http://localhost:11434",
        embedding_provider="ollama",
    )

    provider = EmbeddingFactory.create(settings)

    assert isinstance(provider, OllamaEmbeddingClient)
    assert provider.base_url == "http://localhost:11434"

    await provider.aclose()
