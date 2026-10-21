"""Shared interface and data models for LLM providers."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Parameters controlling text generation."""

    temperature: float
    top_p: float
    max_tokens: int


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized response returned by an LLM provider."""

    content: str
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLMProvider(Protocol):
    """Contract that every LLM provider must implement."""

    async def generate(
        self,
        prompt: str,
        *,
        options: GenerationOptions,
        system_prompt: str | None = None,
    ) -> LLMResponse:
        """Generate a response from a prompt."""
        ...
