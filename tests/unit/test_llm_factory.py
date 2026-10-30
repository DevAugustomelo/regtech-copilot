"""Tests for the LLM factory."""

import pytest

from regtech_copilot.core.config import Settings
from regtech_copilot.llm.factory import (
    LLMFactory,
    UnsupportedLLMProviderError,
)
from regtech_copilot.llm.ollama import OllamaClient


def create_settings(
    llm_provider: str = "ollama",
) -> Settings:
    """Create isolated settings for factory tests."""

    return Settings(
        _env_file=None,
        llm_model="test-model",
        llm_provider=llm_provider,
    )


def test_factory_creates_ollama_provider() -> None:
    settings = create_settings()

    provider = LLMFactory.create(settings)

    assert isinstance(provider, OllamaClient)
    assert provider.model == "test-model"


def test_factory_rejects_unsupported_provider() -> None:
    settings = create_settings("nvidia")

    with pytest.raises(
        UnsupportedLLMProviderError,
        match="not implemented yet",
    ):
        LLMFactory.create(settings)


def test_factory_uses_application_settings() -> None:
    settings = create_settings()

    provider = LLMFactory.create(settings)

    assert provider.model == settings.llm_model
    assert provider.base_url == settings.llm_base_url
