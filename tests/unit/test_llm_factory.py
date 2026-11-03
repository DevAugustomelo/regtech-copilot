"""Tests for the LLM factory."""

import pytest

from regtech_copilot.core.config import Settings
from regtech_copilot.llm.factory import (
    LLMFactory,
    MissingNvidiaAPIKeyError,
    UnsupportedLLMProviderError,
)
from regtech_copilot.llm.nvidia import NvidiaClient
from regtech_copilot.llm.ollama import OllamaClient


def create_settings(
    llm_provider: str = "ollama",
    nvidia_api_key: str | None = None,
) -> Settings:
    return Settings(
        _env_file=None,
        llm_model="test-model",
        llm_provider=llm_provider,
        nvidia_api_key=nvidia_api_key,
    )


def test_factory_creates_ollama_provider() -> None:
    settings = create_settings()
    provider = LLMFactory.create(settings)

    assert isinstance(provider, OllamaClient)
    assert provider.model == "test-model"


def test_factory_creates_nvidia_provider() -> None:
    settings = create_settings(
        llm_provider="nvidia",
        nvidia_api_key="test-api-key",
    )

    provider = LLMFactory.create(settings)

    assert isinstance(provider, NvidiaClient)
    assert provider.model == "test-model"
    assert provider.base_url == settings.llm_base_url


def test_factory_rejects_nvidia_without_api_key() -> None:
    settings = create_settings(llm_provider="nvidia")

    with pytest.raises(
        MissingNvidiaAPIKeyError,
        match="NVIDIA_API_KEY is required",
    ):
        LLMFactory.create(settings)


def test_factory_rejects_empty_nvidia_api_key() -> None:
    settings = create_settings(
        llm_provider="nvidia",
        nvidia_api_key="   ",
    )

    with pytest.raises(
        MissingNvidiaAPIKeyError,
        match="NVIDIA_API_KEY cannot be empty",
    ):
        LLMFactory.create(settings)


def test_factory_rejects_unsupported_provider() -> None:
    settings = create_settings(llm_provider="huggingface")

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
