"""Tests for application settings."""

import pytest
from pydantic import ValidationError

from regtech_copilot.core.config import Settings


def test_settings_load_default_values() -> None:
    settings = Settings(
        _env_file=None,
        llm_model="test-model",
    )

    assert settings.app_name == "RegTech Copilot"
    assert settings.llm_provider == "ollama"
    assert settings.chunk_size_tokens == 700
    assert settings.chunk_overlap_tokens == 100


def test_settings_reject_invalid_port() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            llm_model="test-model",
            api_port=70000,
        )


def test_settings_reject_overlap_equal_to_chunk_size() -> None:
    with pytest.raises(ValidationError, match="CHUNK_OVERLAP_TOKENS"):
        Settings(
            _env_file=None,
            llm_model="test-model",
            chunk_size_tokens=100,
            chunk_overlap_tokens=100,
        )


def test_settings_reject_invalid_temperature() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            llm_model="test-model",
            llm_temperature=3.0,
        )
