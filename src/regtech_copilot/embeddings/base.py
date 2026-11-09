"""Shared interface and data models for embedding providers."""

from dataclasses import dataclass
from typing import Literal, Protocol


class EmbeddingProviderError(Exception):
    """Base exception for embedding provider failures."""


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """Normalized response returned by an embedding provider."""

    embeddings: list[list[float]]
    model: str
    dimensions: int


class EmbeddingProvider(Protocol):
    """Contract that every embedding provider must implement."""

    async def embed(
        self,
        texts: list[str],
        *,
        input_type: Literal["document", "query"] = "document",
    ) -> EmbeddingResponse:
        """Generate vector embeddings for a collection of texts."""
        ...

    async def aclose(self) -> None:
        """Close resources owned by the provider."""
        ...
