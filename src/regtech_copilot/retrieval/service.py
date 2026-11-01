"""Retrieve relevant document chunks using semantic similarity."""

import asyncio

from regtech_copilot.embeddings.base import EmbeddingProvider
from regtech_copilot.vector_store.base import VectorSearchResult, VectorStore


class RetrievalServiceError(Exception):
    """Base exception for retrieval errors."""


class RetrievalEmbeddingError(RetrievalServiceError):
    """Raised when query embedding generation returns invalid results."""


class RetrievalService:
    """Generate query embeddings and retrieve relevant document chunks."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[VectorSearchResult]:
        """Retrieve the most relevant chunks for a user query."""

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        result_limit = self.top_k if top_k is None else top_k

        if result_limit <= 0:
            raise ValueError("top_k must be greater than zero.")

        embedding_response = await self.embedding_provider.embed([query])

        if len(embedding_response.embeddings) != 1:
            raise RetrievalEmbeddingError(
                "Expected exactly one embedding for the query."
            )

        query_embedding = embedding_response.embeddings[0]

        if (
            embedding_response.dimensions <= 0
            or len(query_embedding) != embedding_response.dimensions
        ):
            raise RetrievalEmbeddingError("The query embedding has invalid dimensions.")

        return await asyncio.to_thread(
            self.vector_store.search,
            query_embedding,
            result_limit,
        )
