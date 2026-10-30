"""Shared interface and data models for embedding providers."""

from dataclasses import dataclass
from typing import Protocol


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
    ) -> EmbeddingResponse:
        """Generate vector embeddings for a collection of texts."""
        ...
