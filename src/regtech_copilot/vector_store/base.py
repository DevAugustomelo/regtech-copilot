"""Contracts for vector storage providers."""

from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from regtech_copilot.schemas.documents import DocumentChunk


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """A chunk retrieved through vector similarity search."""

    chunk: DocumentChunk
    distance: float


class VectorStore(Protocol):
    """Contract for vector storage implementations."""

    def add(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store document chunks and their corresponding embeddings."""
        ...

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[VectorSearchResult]:
        """Retrieve the most similar chunks for a query embedding."""
        ...

    def search_by_article(
        self,
        query_embedding: list[float],
        document_id: UUID,
        article_reference: str,
    ) -> list[VectorSearchResult]:
        """Retrieve chunks belonging to a specific article."""
        ...

    def delete(self, document_id: UUID) -> None:
        """Delete all chunks belonging to a document."""
        ...

    def count(self) -> int:
        """Return the total number of stored chunks."""
        ...
