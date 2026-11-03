"""Factory for creating LLM providers."""

from regtech_copilot.core.config import Settings, get_settings
from regtech_copilot.llm.base import LLMProvider
from regtech_copilot.llm.nvidia import NvidiaClient
from regtech_copilot.llm.ollama import OllamaClient


class UnsupportedLLMProviderError(Exception):
    """Raised when a requested LLM provider is not implemented."""


class MissingNvidiaAPIKeyError(Exception):
    """Raised when NVIDIA is selected without an API key."""


class LLMFactory:
    """Create LLM providers based on application settings."""

    @staticmethod
    def create(settings: Settings | None = None) -> LLMProvider:
        """Create the configured LLM provider."""

        config = settings or get_settings()

        if config.llm_provider == "ollama":
            return OllamaClient(
                base_url=config.llm_base_url,
                model=config.llm_model,
            )

        if config.llm_provider == "nvidia":
            if config.nvidia_api_key is None:
                raise MissingNvidiaAPIKeyError(
                    "NVIDIA_API_KEY is required when NVIDIA is selected."
                )

            api_key = config.nvidia_api_key.get_secret_value().strip()

            if not api_key:
                raise MissingNvidiaAPIKeyError(
                    "NVIDIA_API_KEY cannot be empty when NVIDIA is selected."
                )

            return NvidiaClient(
                api_key=api_key,
                base_url=config.llm_base_url,
                model=config.llm_model,
            )

        raise UnsupportedLLMProviderError(
            f"LLM provider '{config.llm_provider}' is not implemented yet."
        )
